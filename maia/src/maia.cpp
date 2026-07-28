// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only


#include "maia.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <fftw3-mpi.h>
#include <string>
#include "INCLUDE/maialikwid.h"
#include "compiler_config.h"
#include "environment.h"
#include "globals.h"

// GCC parallel STL uses the Intel TBB library
// Check that it can be included here.
#if defined(MAIA_GCC_COMPILER) && defined(MAIA_PSTL)
#define TBB_USE_EXCEPTIONS 0
#include <tbb/parallel_for.h>
#endif

#ifdef WITH_PHYDLL_DIRECT
#include "ML/mlCouplerPhyDLL.h"
#endif


#ifdef WITH_SCOREP
#include <scorep/SCOREP_User.h>

SCOREP_USER_REGION_DEFINE(maiaRunRegion);
#endif

using namespace std;

Environment* mEnvironment = nullptr;

namespace {
bool usesPhydllProvider()
{
  const char* raw = std::getenv("CPP_ML_INTERFACE_PROVIDER_ENV");
  if(raw == nullptr || *raw == '\0') {
    return false;
  }
  std::string provider(raw);
  std::transform(provider.begin(), provider.end(), provider.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return provider == "PHYDLL";
}
} // namespace

int main(int argc, char* argv[]) {
  // Create MAIA instance
  MAIA maia(argc, argv);

  // Run main loop and save exit code
  const int exitCode = maia.run();

  // Return exit code to OS
  return exitCode;
}

/// Main controlling method for MAIA. Calls everything else.
int MAIA::run() {
    std::cerr << "[MAIA] entering MAIA::run()" << std::endl;
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_BEGIN(maiaRunRegion, "m-AIA-Solver::run", SCOREP_USER_REGION_TYPE_FUNCTION);
    #endif

  // Initialize MPI communication
  // Profiling uses MPI_Wtime, i.e. the first DEBUG call requires MPI to be initialized already
#ifdef _OPENMP

  /* If OMP_NUM_THREADS is not set, limit number of threads to one, unless another environment
  // variable is set appropriately
  if (!getenv("OMP_NUM_THREADS")) {
    const char* env = getenv("MAIA_AUTO_NUM_THREADS");
    if (!env || MString(env) != MString("1")) {
      omp_set_num_threads(1);
      m_log << "Set number of openMP-Threads to 1" << endl;
      cerr0 << "Set number of openMP-Threads to 1" << endl;
    }
  }
  */

  int provided;
  MPI_Init_thread(&m_argc, &m_argv, MPI_THREAD_FUNNELED, &provided);
  std::cerr << "[MAIA] initialized MPI thread" << std::endl;
  // The check for the provided thread level is omitted until OpenMPI reports
  // the correct level and not just MPI_THREAD_SINGLE no matter what
  // if (provided < MPI_THREAD_FUNNELED) {
  //   cerr << "OpenMP requires a threaded MPI library, but the used library is "
  //        << "not." << endl;
  //   MPI_Finalize();
  //   return EXIT_FAILURE;
  // }
#else
  MPI_Init(&m_argc, &m_argv);
#endif
  int domainId, noDomains;

  /*
   * note(Fabian Orland, 20th December 2023):
   * for the coupling of m-AIA with PhyDLL in the context of CoE RAISE project,
   * m-AIA will be launched in an MPMD fashion like:
   *   mpirun -np X maia : -np Y python3 phydll_DLEngine.py
   * Thus, the MPI_COMM_WORLD communicator will also contain Python processes 
   * that execute the DL script. 
   * As soon as m-AIA calls into collective MPI operations on MPI_COMM_WORLD,
   * a deadlock will occur as the Python DL processes are not part of that.
   * As a solution we create an application-owned solver communicator before
   * MAIA performs any collectives and store it in mpiInformation. The PhyDLL
   * DL clients participate in this same split with MPI_UNDEFINED. PhyDLL then
   * performs its own separate split when CMI initializes it later.
   */
  MPI_Comm maiaCommWorld = MPI_COMM_WORLD;
  bool ownsMaiaCommWorld = false;
#ifdef MLCOUPLING_WITH_PHYDLL
  if(usesPhydllProvider()) {
    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    std::cerr << "[MAIA " << worldRank << "] entering MPI_Comm_split..." << std::endl;
    MPI_Comm_split(MPI_COMM_WORLD, 0, worldRank, &maiaCommWorld);
    std::cerr << "[MAIA " << worldRank << "] completed MPI_Comm_split!" << std::endl;
    ownsMaiaCommWorld = true;
  }
#endif

  MPI_Comm_size(maiaCommWorld, &noDomains);
  MPI_Comm_rank(maiaCommWorld, &domainId);
  g_mpiInformation.init(domainId, noDomains, maiaCommWorld);

  // Set MPI error handling (return error and handle in code)
  if(!usesPhydllProvider()) {
    PMPI_Comm_set_errhandler(globalMaiaCommWorld(), MPI_ERRORS_RETURN);
  }

#ifndef MAIA_WINDOWS
  fftw_mpi_init();
#endif
  // MPI_Comm_rank(MPI_COMM_WORLD, &domainId);
  // MPI_Comm_size(MPI_COMM_WORLD, &noDomains);
  // g_mpiInformation.init(domainId, noDomains);

  // // Set MPI error handling (return error and handle in code)
  // MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  // Open cerr0 on MPI root
  if(domainId == 0) {
    cerr0.rdbuf(cerr.rdbuf());
  } else {
    cerr0.rdbuf(&nullBuffer);
  }

  // Check type sizes to make sure they have the expected size
  // If an error is produced but you want to proceed anyways, just comment the
  // offending line(s) below
  static_assert(CHAR_BIT == 8, "Unexpected type size: char has " XSTRINGIFY(CHAR_BIT) " bits (expected: 8)");
  static_assert(sizeof(MChar) == 1, "MChar has unexpected type size");
#ifndef _SX
  static_assert(sizeof(MBool) == 1, "MBool has unexpected type size");
#endif
  static_assert(sizeof(MInt) == 4, "MInt has unexpected type size");
  static_assert(sizeof(MInt) == 4, "MInt has unexpected type size");
  static_assert(sizeof(MLong) == 8, "MLong has unexpected type size");
  static_assert(sizeof(MFloat) == 8, "MFloat has unexpected type size");

  // Needed for m_log
  Environment::m_argc = m_argc;
  Environment::m_argv = m_argv;

  // Open log file
#if(MAIA_INFOOUT_FILE_TYPE == 1 && MAIA_INFOOUT_ROOT_ONLY == false)
  m_log.open("m_log_" + to_string(globalDomainId()), MAIA_INFOOUT_PROJECT_NAME, MAIA_INFOOUT_FILE_TYPE, globalMaiaCommWorld(),
             MAIA_INFOOUT_ROOT_ONLY);
#else
  m_log.open("m_log", MAIA_INFOOUT_PROJECT_NAME, MAIA_INFOOUT_FILE_TYPE, globalMaiaCommWorld(), MAIA_INFOOUT_ROOT_ONLY);
#endif

  // Set root-only writing property
  m_log.setRootOnly(MAIA_INFOOUT_ROOT_ONLY);

  // Set minimum flush size
  m_log.setMinFlushSize(MAIA_INFOOUT_MIN_FLUSH_SIZE);

  // Reset global debug level (may be overridden using the "-d <lvl>" command line parameter (cf. debug.h)
  SET_DEBUG_LEVEL(0);
  // SET_DEBUG_LEVEL(MAIA_DEBUG_TRACE_IN | MAIA_DEBUG_TRACE_OUT | MAIA_DEBUG_TRACE);
  // Note: do not trace this, since m_log is closed before this trace is finished the output to m_log results in a
  // segmentation fault!
  // TRACE();
  DEBUG("main:: domainId " << globalDomainId(), MAIA_DEBUG_LEVEL1);
  DEBUG("main:: noDomains " << globalNoDomains(), MAIA_DEBUG_LEVEL1);

  // Initialize likwid if likwid is enabled
#ifdef WITH_LIKWID
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  // Reset all timers
  DEBUG("main:: reset timer", MAIA_DEBUG_LEVEL1);
  RESET_TIMERS();

  NEW_TIMER_GROUP(tg, "MAIA");

  // Start timer for the total program execution
  NEW_TIMER(timertotal, "Total", tg);
  RECORD_TIMER_START(timertotal);

  // Create environment
  NEW_SUB_TIMER(timer1, "New Environment", timertotal);
  RECORD_TIMER_START(timer1);

  DEBUG("main:: new Environment", MAIA_DEBUG_LEVEL1);
  mEnvironment = new Environment(m_argc, m_argv);
  RECORD_TIMER_STOP(timer1);

  g_mpiInformation.printInfo();

  // Run MAIA... YEHAW!
  DEBUG("main:: mEnvironment->run", MAIA_DEBUG_LEVEL1);
  NEW_SUB_TIMER(timer2, "Run Environment", timertotal);
  RECORD_TIMER_START(timer2);
  mEnvironment->run();
  RECORD_TIMER_STOP(timer2);

  // Once mEnvironment has finished, call the cleanup function
  NEW_SUB_TIMER(timer3, "End Environment", timertotal);
  DEBUG("main:: mEnvironment->end", MAIA_DEBUG_LEVEL1);
  RECORD_TIMER_START(timer3);
  mEnvironment->end();
  RECORD_TIMER_STOP(timer3);

  // Finalize likwid if likwid is enabled
#ifdef WITH_LIKWID
  LIKWID_MARKER_CLOSE;
#endif

  // Delete mEnvironment
  DEBUG("main:: delete environment", MAIA_DEBUG_LEVEL1);
  if(mEnvironment != nullptr) {
    delete mEnvironment;
    mEnvironment = nullptr;
  }

  // Stop the timer for total program execution
  RECORD_TIMER_STOP(timertotal);

  // Show timer output
  DEBUG("main:: display all timers", MAIA_DEBUG_LEVEL1);
  STOP_ALL_RECORD_TIMERS();
  DISPLAY_ALL_TIMERS();

  // Save memory report to log file
  m_log << Scratch::printSelfReport();

  // Deallocate all memory
  DEBUG("main:: mDealloc", MAIA_DEBUG_LEVEL1);
  mDealloc();

  // Close log file streams
  DEBUG("main:: close streams", MAIA_DEBUG_LEVEL1);
  m_log.close();
  maia_res.close();

  // Stop MPI - after MPI_Finalize, no more calls to any MPI-related functions are allowed!
  // Note: do not use DEBUG() after m_log has been closed!
  // DEBUG("main:: MPI_Finalize", MAIA_DEBUG_LEVEL1);
#ifndef MAIA_WINDOWS
  fftw_mpi_cleanup(); // note(Fabian Orland, April 2nd, 2025) commented out for now because it leads to invalid free on exit
#endif

  #ifdef WITH_SCOREP
      SCOREP_USER_REGION_END(maiaRunRegion);
  #endif
  if(ownsMaiaCommWorld) {
    PMPI_Barrier(MPI_COMM_WORLD);
    PMPI_Comm_free(&maiaCommWorld);
  }
  MPI_Finalize();

  return EXIT_SUCCESS;
}
