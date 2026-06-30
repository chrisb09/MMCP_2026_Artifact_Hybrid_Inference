// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

#include "fvstructuredsolver.h"
#include <cmath>
#include <cstdlib>
#include "COMM/mpioverride.h"
#include "GRID/structuredpartition.h"
#include "IO/parallelio.h"
#include "IO/parallelio_hdf5.h"
#include "UTIL/maiamath.h"
#include "fvstructuredsolverwindowinfo.h"
#include "globals.h"
#if not defined(MAIA_MS_COMPILER)
#include <unistd.h>
#endif
#include <vector>

#include <chrono>

#include "logger.hpp"
#include "snapshot_writer.hpp"

using namespace std;

/**
 * \brief Constructor of the structured solver
 * \authors Pascal Meysonnat, Marian Albers
 * \param[in] solverID ID of this solver
 * \param[in] grid_ Pointer to the StructuredGrid object for which this solver is used
 * \param[in] propertiesGroups
 * \param[in] comm MPI communicator
 */
template <MInt nDim>
FvStructuredSolver<nDim>::FvStructuredSolver(MInt solverId, StructuredGrid<nDim>* grid_, MBool* propertiesGroups,
                                             const MPI_Comm comm)
  : Solver(solverId, comm),
    StructuredPostprocessing<nDim, FvStructuredSolver<nDim>>(),
    m_grid(grid_),
    m_eps(std::numeric_limits<MFloat>::epsilon()) {
  (void)propertiesGroups;
  const MLong oldAllocatedBytes = allocatedBytes();

  // initialize all timers
  initTimers();

  if(domainId() == 0) {
    cout << "Initializing Structured Solver..." << endl;
  }

  m_isActive = true;

  // for the zonal approach we will have to split the communicator
  m_StructuredComm = comm;

  // intialize the most important properties
  initializeFvStructuredSolver(propertiesGroups);

  // numericals method:
  setNumericalProperties();

  // set testcase parameters
  setTestcaseProperties();

  // set moving grid parameters
  setMovingGridProperties();

  setBodyForceProperties();

  // initialize cell
  m_cells = new StructuredCell;

  ///////////////////////////////////////////////////
  /////////////// GRID DECOMPOSITIONING /////////////
  ///////////////////////////////////////////////////
  RECORD_TIMER_START(m_timers[Timers::GridDecomposition]);
  m_grid->gridDecomposition(false);
  RECORD_TIMER_STOP(m_timers[Timers::GridDecomposition]);


  m_log << "Setting porous properties..." << endl;
  setPorousProperties();
  m_log << "Setting porous properties... SUCCESSFUL!" << endl;

  m_log << "Setting zonal properties..." << endl;
  setZonalProperties();
  m_log << "Setting zonal properties... SUCCESSFUL!" << endl;

  m_log << "Allocating variables..." << endl;
  allocateVariables();
  m_log << "Allocating variables... SUCCESSFUL!" << endl;

  m_log << "Reading input output properties..." << endl;
  setInputOutputProperties();
  m_log << "Reading input output properties... SUCCESSFUL!" << endl;

  //////////////////////////////////////////////////////////////
  ///////////////// ASSIGN WINDOW INFORMATION //////////////////
  //////////////////////////////////////////////////////////////
  // 4) assign window winformation (3D-2D possible because of dimension)
  //  but be careful, only 3D has been tested and implemented
  m_grid->setCells(m_cells);
  m_grid->prepareReadGrid(); // determines grid dimensions (point and cell wise) and allocates m_grid->m_coordinates

  m_noCells = m_grid->m_noCells;
  m_noActiveCells = m_grid->m_noActiveCells;
  m_noPoints = m_grid->m_noPoints;
  m_totalNoCells = m_grid->m_totalNoCells;

  m_nOffsetCells = m_grid->m_nOffsetCells;
  m_nOffsetPoints = m_grid->m_nOffsetPoints;
  m_nPoints = m_grid->m_nPoints;
  m_nActivePoints = m_grid->m_nActivePoints;
  m_nCells = m_grid->m_nCells;
  m_nActiveCells = m_grid->m_nActiveCells;
  m_noBlocks = m_grid->getNoBlocks();
  m_blockId = m_grid->getMyBlockId();

  m_windowInfo =
      make_unique<FvStructuredSolverWindowInfo<nDim>>(m_grid, m_StructuredComm, noDomains(), domainId(), m_solverId);

  m_windowInfo->readWindowInfo();
  m_windowInfo->initGlobals();

  m_windowInfo->readWindowCoordinates(m_grid->m_periodicDisplacements);

  createMPIGroups();

  readAndSetSpongeLayerProperties();

  m_setLocalWallDistance =
      Context::getSolverProperty<MBool>("setLocalWallDistance", m_solverId, AT_, &m_setLocalWallDistance);
  IF_CONSTEXPR(nDim == 3) {
    if(m_setLocalWallDistance) mTerm(-1, "Not implemented in 3D yet!");
  }
  if(m_zonal || m_rans) {
    if(!m_setLocalWallDistance) m_windowInfo->setWallInformation();
  }

  // Create the mapping for the boundary and domain windows
  // the windows are needed for boundary condition application
  // and data exchange between the domains
  m_windowInfo->createWindowMapping(&m_commChannelIn, &m_commChannelOut, &m_commChannelWorld, m_channelRoots,
                                    &m_commStg, &m_commStgRoot, &m_commStgRootGlobal, &m_commBC2600, &m_commBC2600Root,
                                    &m_commBC2600RootGlobal, &m_rescalingCommGrComm, &m_rescalingCommGrRoot,
                                    &m_rescalingCommGrRootGlobal, &m_commPerRotOne, &m_commPerRotTwo,
                                    &m_commPerRotWorld, m_commPerRotRoots, m_commPerRotGroup, m_grid->m_singularity,
                                    &m_grid->m_hasSingularity, &m_plenumComm, &m_plenumRoot);

  m_singularity = m_grid->m_singularity; // new SingularInformation[30]; //TODO_SS labels:FV,totest check this
  m_hasSingularity = m_grid->m_hasSingularity;

  //
  readAndSetAuxDataMap();

  // Creating the communication flags needed
  // for all inter-domain exchange operations


  m_periodicConnection = 0;
  // set flag for the periodic exchange
  for(MInt i = 0; i < (MInt)m_windowInfo->rcvMap.size(); i++) {
    if(m_windowInfo->rcvMap[i]->BC >= 4000 && m_windowInfo->rcvMap[i]->BC <= 4999) {
      m_periodicConnection = 1;
      break;
    }
  }

  // TODO_SS labels:FV,toremove the following is not used
  //  if(m_zonal || m_rans) {
  //    if (m_setLocalWallDistance)
  //      m_windowInfo->setLocalWallInformation();
  //  }

  ///////////////////////////////////////////////////
  ///////////////// READ GRID ///////////////////////
  ///////////////////////////////////////////////////
  // Information about singularities needs to be known for allocating metrices
  m_grid->allocateMetricsAndJacobians();

  allocateAndInitBlockMemory();

  m_windowInfo->createCommunicationExchangeFlags(m_sndComm, m_rcvComm, PV->noVariables, m_cells->pvariables);

  MInt averageCellsPerDomain = (MInt)(m_totalNoCells / noDomains());
  MFloat localDeviation = ((MFloat)averageCellsPerDomain - (MFloat)m_noActiveCells) / ((MFloat)averageCellsPerDomain);
  MFloat localDeviationSquare = POW2(localDeviation);
  MFloat globalMaxDeviation = F0;
  MFloat globalMinDeviation = F0;
  MFloat globalAvgDeviation = F0;
  MInt globalMaxNoCells = 0;
  MInt globalMinNoCells = 0;
  MPI_Allreduce(&localDeviation, &globalMaxDeviation, 1, MPI_DOUBLE, MPI_MAX, m_StructuredComm, AT_, "localDeviation",
                "globalMaxDeviation");
  MPI_Allreduce(&localDeviation, &globalMinDeviation, 1, MPI_DOUBLE, MPI_MIN, m_StructuredComm, AT_, "localDeviation",
                "globalMinDeviation");
  MPI_Allreduce(&localDeviationSquare, &globalAvgDeviation, 1, MPI_DOUBLE, MPI_SUM, m_StructuredComm, AT_,
                "localDeviation", "globalAvgDeviation");
  globalAvgDeviation = sqrt(globalAvgDeviation / noDomains());
  MPI_Allreduce(&m_noActiveCells, &globalMaxNoCells, 1, MPI_INT, MPI_MAX, m_StructuredComm, AT_, "m_noActiveCells",
                "globalMaxNoCells");
  MPI_Allreduce(&m_noActiveCells, &globalMinNoCells, 1, MPI_INT, MPI_MIN, m_StructuredComm, AT_, "m_noActiveCells",
                "globalMinNoCells");


  if(domainId() == 0) {
    cout << "///////////////////////////////////////////////////////////////////" << endl
         << "Total no. of grid cells: " << m_totalNoCells << endl
         << "Average cells per domain: " << averageCellsPerDomain << endl
         << "Max no of cells per domain: " << globalMaxNoCells << endl
         << "Min no of cells per domain: " << globalMinNoCells << endl
         << "Average deviation from average: " << globalAvgDeviation * 100.0 << " percent" << endl
         << "Maximum deviation from average: +" << globalMaxDeviation * 100.0 << " / " << globalMinDeviation * 100.0
         << " percent" << endl
         << "///////////////////////////////////////////////////////////////////" << endl;
  }

  // read the grid from partition and
  // and move coordinates to right position
  if(domainId() == 0) {
    cout << "Reading Grid..." << endl;
  }
  m_log << "->reading the grid file" << endl;
  RECORD_TIMER_START(m_timers[Timers::GridReading]);
  m_grid->readGrid();
  RECORD_TIMER_STOP(m_timers[Timers::GridReading]);
  m_log << "------------- Grid read successfully! -------------- " << endl;
  if(domainId() == 0) {
    cout << "Reading Grid SUCCESSFUL!" << endl;
  }

  // get the zonal BC information
  if(m_zonal) {
    m_windowInfo->setZonalBCInformation();
  }

  // set properties for synthetic turbulence generation method
  setSTGProperties();

  // set the properties for bc2600 (if existing)
  setProfileBCProperties();

  // initialize the postprocessing class
  initStructuredPostprocessing();

  // --- ML Coupling setup (new CPP-ML-Interface) ---
  RECORD_TIMER_START(m_timers[Timers::MLCoupling]);
  RECORD_TIMER_START(m_timers[Timers::Setup]);

  MInt defaultMLInterval = 100;
  MInt mlInterval = Context::getBasicProperty<MInt>("mlInterval", AT_, &defaultMLInterval);
  MInt mlStart = mlInterval;
  MInt defaultMLInputSeqLen = 5;
  MInt mlInputSeqLen = Context::getBasicProperty<MInt>("mlInputLength", AT_, &defaultMLInputSeqLen);
  MInt hdfOutputInterval = Context::getBasicProperty<MInt>("solutionInterval", AT_);
  MInt defaultMLStepCoeff = 24;
  MInt mlStepCoeff = Context::getBasicProperty<MInt>("mlStepCoefficient", AT_, &defaultMLStepCoeff);
  MInt defaultMLForecastWindow = 2;
  MInt mlForecastWindow = Context::getBasicProperty<MInt>("mlForecastWindow", AT_, &defaultMLForecastWindow);
  MFloat defaultMLScalingFactor = 1.0;
  MFloat mlScalingFactor = Context::getBasicProperty<MFloat>("mlScalingFactor", AT_, &defaultMLScalingFactor);
  MInt defaultMLInputStepDistance = 24;
  MInt mlInputStepDistance = Context::getBasicProperty<MInt>("mlInputStepDistance", AT_, &defaultMLInputStepDistance);
  MInt defaultMLCubeOverlap = 0;
  MInt mlCubeOverlap = Context::getBasicProperty<MInt>("mlCubeOverlap", AT_, &defaultMLCubeOverlap);
  MInt defaultMLCubeD = 8;
  MInt mlCubeD = Context::getBasicProperty<MInt>("mlCubeD", AT_, &defaultMLCubeD);

  // Allocate float buffers for U/V/W double->float copy
  MLong totalCells = static_cast<MLong>(m_nCells[0]) * static_cast<MLong>(m_nCells[1]) * static_cast<MLong>(m_nCells[2]);
  for (int f = 0; f < 3; ++f) {
    m_mlInputBuf[f].resize(totalCells);
    m_mlOutputBuf[f].resize(totalCells);
  }

  // Build MLCouplingData<float> wrappers around owned float buffers
  std::vector<MLCouplingTensor<float>> input_tensors;
  std::vector<MLCouplingTensor<float>> output_tensors;
  std::vector<std::vector<int>> dims_vec(3, {m_nCells[0], m_nCells[1], m_nCells[2]});
  for (int f = 0; f < 3; ++f) {
    input_tensors.push_back(MLCouplingTensor<float>::wrap_flat(
        m_mlInputBuf[f].data(), dims_vec[f]));
    output_tensors.push_back(MLCouplingTensor<float>::wrap_flat(
        m_mlOutputBuf[f].data(), dims_vec[f]));
  }
  MLCouplingData<float> input_data(std::move(input_tensors));
  MLCouplingData<float> output_data(std::move(output_tensors));

  // Build ConfigOverrides from Context properties (override config.toml defaults)
  ConfigOverrides overrides;
  MPI_Comm ml_comm = globalMaiaCommWorld();
  overrides.dotted["provider.app_comm"] = static_cast<void*>(&ml_comm);
  overrides.dotted["provider.model_file"] = Context::getBasicProperty<MString>("modelPath", AT_);
  overrides.dotted["behavior.global_step_offset"] = static_cast<int64_t>(m_restartTimeStep);
  overrides.dotted["behavior.inference_interval"] = static_cast<int64_t>(mlInterval);
  overrides.dotted["behavior.coupled_steps_before_inference"] = static_cast<int64_t>(mlInputSeqLen);
  overrides.dotted["behavior.step_increment_after_inference"] = static_cast<int64_t>(mlStepCoeff);
  overrides.dotted["behavior.hdf_output_interval"] = static_cast<int64_t>(hdfOutputInterval);
  MInt totalTimesteps = Context::getBasicProperty<MInt>("timeSteps", AT_);
  overrides.dotted["behavior.total_timesteps"] = static_cast<int64_t>(totalTimesteps);
  overrides.dotted["behavior.scaling_factor"] = static_cast<double>(mlScalingFactor);
  overrides.dotted["behavior.forecast_window"] = static_cast<int64_t>(mlForecastWindow);
  overrides.dotted["behavior.input_step_distance"] = static_cast<int64_t>(mlInputStepDistance);
  overrides.dotted["behavior.inference_start_step"] = static_cast<int64_t>(mlStart);
  overrides.dotted["application.cube_dimension"] = static_cast<int64_t>(mlCubeD);
  overrides.dotted["application.cube_overlap"] = static_cast<int64_t>(mlCubeOverlap);
  overrides.dotted["application.input_sequence_length"] = static_cast<int64_t>(mlInputSeqLen);
  overrides.dotted["application.forecast_window"] = static_cast<int64_t>(mlForecastWindow);
  overrides.dotted["application.n_ghost_layers"] = static_cast<int64_t>(m_noGhostLayers);

  // Create the MLCoupling instance via config file
  m_mlCoupler.reset(MLCoupling<float,float>::create_from_config(
      "./config.toml", std::move(input_data), std::move(output_data), overrides));

  log_init<9>("Setting up ML Coupler (new CMI)",
           std::array<std::string, 9>{"mlInterval", "mlInputLength", "solutionInterval",
            "mlStepCoefficient", "mlForecastWindow", "mlScalingFactor",
            "mlInputStepDistance", "mlCubeOverlap", "mlCubeD"},
           std::array<int, 9>{mlInterval, mlInputSeqLen, hdfOutputInterval,
            mlStepCoeff, mlForecastWindow, static_cast<MInt>(mlScalingFactor),
            mlInputStepDistance, mlCubeOverlap, mlCubeD});
  snapshot::init();

  RECORD_TIMER_STOP(m_timers[Timers::Setup]);
  RECORD_TIMER_STOP(m_timers[Timers::MLCoupling]);


  // print allocated scratch memory
  if(domainId() == 0) {
    MFloat scratchMemory = (Scratch::getTotalMemory() / 1024.0) * noDomains();
    MString memoryUnit = " KB";
    if(scratchMemory > 1024.0) {
      scratchMemory /= 1024.0;
      memoryUnit = " MB";
    }
    if(scratchMemory > 1024.0) {
      scratchMemory /= 1024.0;
      memoryUnit = " GB";
    }
    cout << "=== Total global scratch space memory: " << setprecision(2) << fixed << scratchMemory << memoryUnit
         << " ===" << endl;
  }

  printAllocatedMemory(oldAllocatedBytes, "FvStructuredSolver", m_StructuredComm);
}


template <MInt nDim>
FvStructuredSolver<nDim>::~FvStructuredSolver() {
  snapshot::finalize();
  log_deinit();
  RECORD_TIMER_STOP(m_timers[Timers::Structured]);
  delete m_cells;
}


/**
 * \brief Counts the number of necessary FQ fields, allocates them and corrects the indexes of the FQ variable pointers
 * \author Marian Albers
 * \date 17.09.2015
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::initializeFQField() {
  TRACE();
  // count the number of needed FQ fields and allocate
  MInt noFQFieldsNeeded = 0;
  for(MInt i = 0; i < FQ->maxNoFQVariables; i++) {
    noFQFieldsNeeded += FQ->neededFQVariables[i];
  }

  FQ->noFQVariables = noFQFieldsNeeded;

  m_log << "Allocating " << noFQFieldsNeeded << " FQ fields for " << m_noCells << "..." << endl;
  mAlloc(m_cells->fq, noFQFieldsNeeded, m_noCells, "m_cells->fq", F0, AT_);
  mAlloc(FQ->loadedFromRestartFile, noFQFieldsNeeded, "FQ->loadedFromRestartFile", false, AT_);
  m_log << "Allocating " << noFQFieldsNeeded << " FQ fields for " << m_noCells << "...SUCCESSFUL" << endl;

  MInt currentPos = 0;
  for(MInt i = 0; i < FQ->maxNoFQVariables; i++) {
    if(FQ->neededFQVariables[i] == 0) {
      continue;
    }

    FQ->activateFQField(i, currentPos, FQ->outputFQVariables[i], FQ->boxOutputFQVariables[i]);

    FQ->noFQBoxOutput += (MInt)FQ->boxOutputFQVariables[i];
    currentPos++;
  }
}

template <MInt nDim>
void FvStructuredSolver<nDim>::allocateAndInitBlockMemory() {
  if(m_movingGrid || m_bodyForce) {
    if(m_travelingWave) {
      mAlloc(m_tempWaveSample, (PV->noVariables + (2 * nDim - 3)), m_noCells, "m_tempWaveSample", F0, FUN_);
    }
  }

  if(m_localTimeStep) {
    mAlloc(m_cells->localTimeStep, m_noCells, "m_cells->localTimeStep", -1.01010101, AT_);
  }

  mAlloc(m_cells->variables, m_maxNoVariables, m_noCells, "m_cells->variables", -99999.0, AT_);
  mAlloc(m_cells->pvariables, m_maxNoVariables, m_noCells, "m_cells->pvariables", -99999.0, AT_);
  mAlloc(m_cells->temperature, m_noCells, "m_cells->temperature", -99999.0, AT_);
  mAlloc(m_cells->oldVariables, m_maxNoVariables, m_noCells, "m_cells->oldVariables", -99999.0, AT_);
  mAlloc(m_cells->dss, nDim, m_noCells, "m_cells->dss", F0, AT_);


  // always allocate moving grid volume fluxes
  // but leave them zero if not moving grid
  mAlloc(m_cells->dxt, nDim, m_noCells, "m_cells->dxt", F0, AT_);

  // allocate viscous flux computation variables
  MInt noVarsFluxes = (CV->noVariables - 1 + m_rans);

  m_log << "Allocating fFlux with " << (CV->noVariables - 1 + m_rans) << " variables for " << m_noCells << " cells"
        << endl;

  mAlloc(m_cells->rightHandSide, CV->noVariables, m_noCells, "m_cells->rhs", F0, AT_);
  mAlloc(m_cells->flux, CV->noVariables, m_noCells, "m_cells->flux", 10000.0, AT_);
  mAlloc(m_cells->eFlux, noVarsFluxes, m_noCells, "m_cells->eFlux", F0, AT_);
  mAlloc(m_cells->fFlux, noVarsFluxes, m_noCells, "m_cells->fFlux", F0, AT_);
  mAlloc(m_cells->gFlux, noVarsFluxes, m_noCells, "m_cells->gFlux", F0, AT_);
  // TODO_SS labels:FV,toenhance find a better way to save porous variables
  mAlloc(m_cells->viscousFlux, (nDim + m_porous), m_noCells, "m_cells->viscousFlux", 1000.0, AT_);

  mAlloc(m_QLeft, m_maxNoVariables, "m_QLeft", F0, AT_);
  mAlloc(m_QRight, m_maxNoVariables, "m_QRight", F0, AT_);


  IF_CONSTEXPR(nDim == 2) {
    if(m_viscCompact) mAlloc(m_cells->dT, 4, m_noCells, "m_cells->dT", F0, AT_);
  }
}


/**
 * \brief Reads properties and initializes variables associated with input/output.
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::setInputOutputProperties() {
  TRACE();

  m_outputIterationNumber = 0;
  m_outputFormat = ".hdf5";
  m_lastOutputTimeStep = -1;

  /*! \property
    \page propertiesFVSTRCTRD
    \section forceOutputInterval
    <code>MInt FvStructuredSolver::m_forceOutputInterval </code>\n
    default = <code>"./out"</code>\n \n
    Interval in which auxDataFiles (containing forces etc.).\n
    should be written.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_forceOutputInterval = 0;
  if(Context::propertyExists("forceOutputInterval", m_solverId)) {
    m_forceOutputInterval =
        Context::getSolverProperty<MInt>("forceOutputInterval", m_solverId, AT_, &m_forceOutputInterval);
  } else if(Context::propertyExists("dragOutputInterval", m_solverId)) {
    m_forceOutputInterval =
        Context::getSolverProperty<MInt>("dragOutputInterval", m_solverId, AT_, &m_forceOutputInterval);
  }

  if(m_forceOutputInterval > 0) {
    /*! \property
      \page propertiesFVSTRCTRD
      \section auxOutputDir
      <code>MString FvStructuredSolver::m_auxOutputDir </code>\n
      default = <code> solutionOutput</code>\n \n
      Folder for auxData files.\n
      Keywords: <i>FORCES, IO, STRUCTURED</i>
    */
    m_auxOutputDir = m_solutionOutput;
    if(Context::propertyExists("auxOutputDir", m_solverId)) {
      m_auxOutputDir = Context::getSolverProperty<MString>("auxOutputDir", m_solverId, AT_);

      MString comparator = "/";
      if(strcmp((m_auxOutputDir.substr(m_auxOutputDir.length() - 1, 1)).c_str(), comparator.c_str()) != 0) {
        m_auxOutputDir = m_auxOutputDir + "/";
      }
    }
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section forceAsciiOutputInterval
    <code>MInt FvStructuredSolver::m_forceAsciiOutputInterval </code>\n
    default = <code>"./out"</code>\n \n
    Interval in which the integrated forces .\n
    should be written to an ASCII file.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_forceAsciiOutputInterval = 0;
  if(Context::propertyExists("forceAsciiOutputInterval", m_solverId)) {
    m_forceAsciiOutputInterval =
        Context::getSolverProperty<MInt>("forceAsciiOutputInterval", m_solverId, AT_, &m_forceAsciiOutputInterval);
  } else if(Context::propertyExists("dragAsciiOutputInterval", m_solverId)) {
    m_forceAsciiOutputInterval =
        Context::getSolverProperty<MInt>("dragAsciiOutputInterval", m_solverId, AT_, &m_forceAsciiOutputInterval);
  }


  /*! \property
    \page propertiesFVSTRCTRD
    \section forceAsciiComputeInterval
    <code>MInt FvStructuredSolver::m_forceAsciiComputeInterval </code>\n
    default = <code>"./out"</code>\n \n
    Interval in which the integrated forces .\n
    should be computed.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>Forces, IO, STRUCTURED</i>
  */
  m_forceAsciiComputeInterval = 0;
  if(Context::propertyExists("forceAsciiComputeInterval", m_solverId)) {
    m_forceAsciiComputeInterval =
        Context::getSolverProperty<MInt>("forceAsciiComputeInterval", m_solverId, AT_, &m_forceAsciiComputeInterval);
  } else if(Context::propertyExists("dragAsciiComputeInterval", m_solverId)) {
    m_forceAsciiComputeInterval =
        Context::getSolverProperty<MInt>("dragAsciiComputeInterval", m_solverId, AT_, &m_forceAsciiComputeInterval);
  }

  if(m_forceAsciiComputeInterval > m_forceAsciiOutputInterval) {
    m_forceAsciiComputeInterval = m_forceAsciiOutputInterval;
  }

  if(m_forceAsciiOutputInterval > 0 && m_forceAsciiComputeInterval == 0) {
    m_forceAsciiComputeInterval = 1;
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section forceSecondOrder
    <code>MInt FvStructuredSolver::m_forceSecondOrder </code>\n
    default = <code>"./out"</code>\n \n
    Second-order computation of force .\n
    Possible values are:\n
    <ul>
    <li>Boolean True/False</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_forceSecondOrder = true;
  if(Context::propertyExists("forceSecondOrder", m_solverId)) {
    m_forceSecondOrder = Context::getSolverProperty<MBool>("forceSecondOrder", m_solverId, AT_, &m_forceSecondOrder);
  } else if(Context::propertyExists("dragSecondOrder", m_solverId)) {
    m_forceSecondOrder = Context::getSolverProperty<MBool>("dragSecondOrder", m_solverId, AT_, &m_forceSecondOrder);
  }

  if(m_forceSecondOrder) {
    m_log << "Second order force computation is activated" << endl;
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section outputOffset
    <code>MInt FvStructuredSolver::m_outputOffset </code>\n
    default = <code> 0 </code>\n \n
    Time step before which no output should be written.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>IO, STRUCTURED</i>
  */
  m_outputOffset = 0;
  m_outputOffset = Context::getSolverProperty<MInt>("outputOffset", m_solverId, AT_, &m_outputOffset);

  /*! \property
    \page propertiesFVSTRCTRD
    \section ignoreUID
    <code>MInt FvStructuredSolver::m_ignoreUID </code>\n
    default = <code> 0</code>\n \n
    Switch to override the UID check for
    restart files
    Possible values are:\n
    <ul>
    <li>0 = UID is checked</li>
    <li>1 = UID is not checked</li>
    </ul>
    Keywords: <i>RESTART, IO, STRUCTURED</i>
  */
  m_ignoreUID = 0;
  if(Context::propertyExists("ignoreUID", m_solverId)) {
    m_ignoreUID = Context::getSolverProperty<MBool>("ignoreUID", m_solverId, AT_, &m_ignoreUID);
    m_log << "WARNING!!!!!!!!!!!!!!: UID was not checked. Solution and grid might not fit together" << endl;
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section restartFile
    <code>MInt FvStructuredSolver::m_restartFile </code>\n
    default = <code> 0</code>\n \n
    Possible values are:\n
    <ul>
    <li>0 = initial start</li>
    <li>1 = start from restart file</li>
    </ul>
    Keywords: <i>RESTART, IO, STRUCTURED</i>
  */
  m_restart = false;
  m_restart = Context::getSolverProperty<MBool>("restartFile", m_solverId, AT_, &m_restart);

  m_restartTimeStep = 0;

  /*! \property
    \page propertiesFVSTRCTRD
    \section useNonSpecifiedRestartFile
    <code>MString FvStructuredSolver::m_useNonSpecifiedRestartFile </code>\n
    default = <code> 0</code>\n \n
    Keywords: <i>RESTART, IO, STRUCTURED</i>
  */
  m_useNonSpecifiedRestartFile = false;
  m_useNonSpecifiedRestartFile =
      Context::getSolverProperty<MBool>("useNonSpecifiedRestartFile", m_solverId, AT_, &m_useNonSpecifiedRestartFile);

  /*! \property
    \page propertiesFVSTRCTRD
    \section changeMa
    <code>MInt FvStructuredSolver::m_changeMa </code>\n
    default = <code> 0</code>\n \n
    Specify whether the variables should be transformed to
    a changed Ma number
    Possible values are:\n
    <ul>
    <li>0 = no conversion</li>
    <li>1 = convert all variables to new Ma number</li>
    </ul>
    Keywords: <i>RESTART, IO, STRUCTURED</i>
  */
  m_changeMa = false;
  if(m_restart) {
    m_changeMa = Context::getSolverProperty<MBool>("changeMa", m_solverId, AT_, &m_changeMa);
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section debugOutput
    <code>MInt FvStructuredSolver::m_debugOutput </code>\n
    default = <code> 0</code>\n \n
    Write out debug information as solverId and cellId to solution file.
    Possible values are:\n
    <ul>
    <li>0 = no debug output</li>
    <li>1 = write solverId, cellId</li>
    </ul>
    Keywords: <i>RESTART, IO, STRUCTURED</i>
  */
  m_debugOutput = false;
  m_debugOutput = Context::getSolverProperty<MBool>("debugOutput", m_solverId, AT_, &m_debugOutput);

  if(m_debugOutput) {
    FQ->neededFQVariables[FQ->BLOCKID] = 1;
    FQ->neededFQVariables[FQ->CELLID] = 1;
  }

  FQ->neededFQVariables[FQ->MU_L] = 1;
  FQ->outputFQVariables[FQ->MU_L] = false;
  FQ->neededFQVariables[FQ->MU_T] = 1;
  FQ->outputFQVariables[FQ->MU_T] = false;

  /*! \property
    \page propertiesFVSTRCTRD
    \section savePartitionOutput
    <code>MInt FvStructuredSolver::m_savePartitionOutput </code>\n
    default = <code> 0</code>\n \n
    Save also partitioned solution file with current number of domains.\n
    Possible values are:\n
    <ul>
    <li>0 = no output</li>
    <li>1 = write partitioned file</li>
    </ul>
    Keywords: <i>RESTART, IO, STRUCTURED</i>
  */
  m_savePartitionOutput = false;
  m_savePartitionOutput =
      Context::getSolverProperty<MBool>("savePartitionOutput", m_solverId, AT_, &m_savePartitionOutput);

  /*! \property
    \page propertiesFVSTRCTRD
    \section computeCf
    <code>MInt FvStructuredSolver::m_bForce </code>\n
    default = <code> 0</code>\n \n
    Compute and write skin friction and pressure coefficient.\n
    Possible values are:\n
    <ul>
    <li>0 = no output</li>
    <li>1 = write cf values</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_bForce = false;
  if(Context::propertyExists("computeForce", m_solverId)) {
    m_bForce = Context::getSolverProperty<MBool>("computeForce", m_solverId, AT_, &m_bForce);
  } else if(Context::propertyExists("computeCfCp", m_solverId)) {
    m_bForce = Context::getSolverProperty<MBool>("computeCfCp", m_solverId, AT_, &m_bForce);
  }

  if(m_bForce) {
    m_log << "<<<<< Skin-friction and Pressure Coefficient Computation: ENABLED" << endl;
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section computePower
    <code>MInt FvStructuredSolver::m_bPower </code>\n
    default = <code> 0</code>\n \n
    Compute and write out the power spent for actuation.\n
    Possible values are:\n
    <ul>
    <li>0 = no output</li>
    <li>1 = write power values</li>
    </ul>
    Keywords: <i>FORCES, IO, POWER, STRUCTURED</i>
  */
  m_bPower = false;
  m_bPower = Context::getSolverProperty<MBool>("computePower", m_solverId, AT_, &m_bPower);
  if(m_bPower) {
    m_log << "<<<<< Power Computation: ENABLED" << endl;
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section detailAuxData
    <code>MInt FvStructuredSolver::m_detailAuxData </code>\n
    default = <code> 0</code>\n \n
    Write additional information into auxData files (areas, coordiantes).\n
    Possible values are:\n
    <ul>
    <li>0 = no output</li>
    <li>1 = write additional info</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_detailAuxData = false;
  m_detailAuxData = Context::getSolverProperty<MBool>("detailAuxData", m_solverId, AT_, &m_detailAuxData);

  /*! \property
    \page propertiesFVSTRCTRD
    \section computeCpLineAverage
    <code>MInt FvStructuredSolver::m_bCpLineAveraging </code>\n
    default = <code> 0</code>\n \n
    Trigger the line averaging of the cp/cf value.\n
    Possible values are:\n
    <ul>
    <li>0 = no averaging</li>
    <li>1 = compute line average</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_bForceLineAverage = false;
  if(Context::propertyExists("computeForceLineAverage", m_solverId)) {
    m_bForceLineAverage =
        Context::getSolverProperty<MBool>("computeForceLineAverage", m_solverId, AT_, &m_bForceLineAverage);
  } else if(Context::propertyExists("computeCpLineAverage", m_solverId)) {
    m_bForceLineAverage =
        Context::getSolverProperty<MBool>("computeCpLineAverage", m_solverId, AT_, &m_bForceLineAverage);
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section forceAveragingDir
    <code>MInt FvStructuredSolver::m_forceAveragingDir </code>\n
    default = <code> 0</code>\n \n
    Direction in which to compute the force line average\n
    Possible values are:\n
    <ul>
    <li>0 = i-direction</li>
    <li>1 = j-direction</li>
    <li>2 = k-direction</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_forceAveragingDir = 0;
  if(Context::propertyExists("forceAveragingDir", m_solverId)) {
    m_forceAveragingDir = Context::getSolverProperty<MInt>("forceAveragingDir", m_solverId, AT_, &m_forceAveragingDir);
  } else if(Context::propertyExists("cpAveragingDir", m_solverId)) {
    m_forceAveragingDir = Context::getSolverProperty<MInt>("cpAveragingDir", m_solverId, AT_, &m_forceAveragingDir);
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section auxDataCoordinateLimits
    <code>MInt FvStructuredSolver::m_auxDataCoordinateLimits </code>\n
    default = <code> 0</code>\n \n
    Trigger the limitation of the cp/cf computation region.\n
    Possible values are:\n
    <ul>
    <li>0 = no limits</li>
    <li>1 = read in limits</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_auxDataCoordinateLimits = false;
  m_auxDataCoordinateLimits =
      Context::getSolverProperty<MBool>("auxDataCoordinateLimits", m_solverId, AT_, &m_auxDataCoordinateLimits);

  // verify the conditions for auxilary data computation, i.e., when force output is higher than 1 then cp and cf should
  // be enabled
  if(m_forceOutputInterval || m_forceAsciiOutputInterval) {
    if(!m_bForce) {
      m_bForce = true;
    }
  }

  if(m_auxDataCoordinateLimits) {
    m_auxDataLimits = nullptr;
    MInt noAuxDataLimits = 4;
    mAlloc(m_auxDataLimits, noAuxDataLimits, "m_auxDataLimits", F0, AT_);

    for(MInt i = 0; i < noAuxDataLimits; ++i) {
      /*! \property
  \page propertiesFVSTRCTRD
        \section auxDataLimits
        <code>MFloat FvStructuredSolver::m_auxDataLimits </code>\n
        default = <code> 0</code>\n \n
        Limiting coordinates for 2D rectangle\n
        for c_d computation. \n
        Possible values are:\n
        <ul>
        <li>0.0,2.0,0.0,2.0 = limits a rectangle of 2.0 x 2.0 </li>
        </ul>
        Keywords: <i>FORCE, IO, STRUCTURED</i>
      */
      m_auxDataLimits[i] = -99999.9;
      m_auxDataLimits[i] = Context::getSolverProperty<MFloat>("auxDataLimits", m_solverId, AT_, &m_auxDataLimits[i], i);
    }

    m_log << "AuxData limited area"
          << ", lower x-limit: " << m_auxDataLimits[0] << ", upper x-limit: " << m_auxDataLimits[1]
          << ", lower z-limit: " << m_auxDataLimits[2] << ", upper z-limit: " << m_auxDataLimits[3] << endl;
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section computeLambda2
    <code>MInt StructuredSolver::m_computeLamda2 </code>\n
    default = <code>0</code>\n \n
    this property triggers the lambda2 calculation. It will then be written out with
    the same frequency as used for saveOutput \n
    possible values are:
    <ul>
    <li>Non-negative integervalues: <0> := off; <1>:=on</li>
    </ul>
    Keywords: <i>TURBULENCE, VORTICES, STRUCTURED</i>
  */
  m_computeLambda2 = false;
  m_computeLambda2 = Context::getSolverProperty<MBool>("computeLambda2", m_solverId, AT_, &m_computeLambda2);
  if(m_computeLambda2) {
    FQ->neededFQVariables[FQ->LAMBDA2] = 1;
    FQ->boxOutputFQVariables[FQ->LAMBDA2] = 1;
  }


  /*! \property
    \page propertiesFVSTRCTRD
    \section vorticityOutput
    <code>MInt StructuredSolver::m_vorticityOutput </code>\n
    default = <code>0</code>\n \n
    Trigger vorticity computation and output\n
    possible values are:
    <ul>
    <li>Non-negative integervalues: <0> := off; <1>:=on</li>
    </ul>
    Keywords: <i>TURBULENCE, VORTICITY, STRUCTURED</i>
  */
  m_vorticityOutput = false;
  m_vorticityOutput = Context::getSolverProperty<MBool>("vorticityOutput", m_solverId, AT_, &m_vorticityOutput);


  /*! \property
    \page propertiesFVSTRCTRD
    \section averageVorticity
    <code>MInt StructuredSolver::m_averageVorticity </code>\n
    default = <code>0</code>\n \n
    Average the vorticity in the postprocessing\n
    possible values are:
    <ul>
    <li>Non-negative integervalues: <0> := off; <1>:=on</li>
    </ul>
    Keywords: <i>POSTPROCESSING, VORTICITY, STRUCTURED</i>
  */
  m_averageVorticity = false;
  m_averageVorticity = Context::getSolverProperty<MBool>("pp_averageVorticity", m_solverId, AT_, &m_averageVorticity);

  if(m_vorticityOutput || m_averageVorticity) {
    for(MInt v = 0; v < nDim; v++) {
      FQ->neededFQVariables[FQ->VORTX + v] = 1;
    }
  }
  // if no vorticity output is activated but averaged vorticity is on then output
  // is deactivated for solution output
  if(!m_vorticityOutput) {
    for(MInt v = 0; v < nDim; v++) {
      FQ->outputFQVariables[FQ->VORTX + v] = false;
    }
  }


  ///////////////////////////////////////////////////
  ///////////////// Variable Names //////////////////
  ///////////////////////////////////////////////////
  mAlloc(m_variableNames, m_maxNoVariables, "m_variableNames", AT_);
  switch(nDim) {
    case 1: {
      m_variableNames[CV->RHO_U] = "rhoU";
      m_variableNames[CV->RHO_E] = "rhoE";
      m_variableNames[CV->RHO] = "rho";
      break;
    }
    case 2: {
      m_variableNames[CV->RHO_U] = "rhoU";
      m_variableNames[CV->RHO_V] = "rhoV";
      m_variableNames[CV->RHO_E] = "rhoE";
      m_variableNames[CV->RHO] = "rho";
      break;
    }
    case 3: {
      m_variableNames[CV->RHO_U] = "rhoU";
      m_variableNames[CV->RHO_V] = "rhoV";
      m_variableNames[CV->RHO_W] = "rhoW";
      m_variableNames[CV->RHO_E] = "rhoE";
      m_variableNames[CV->RHO] = "rho";
      break;
    }
    default: {
      mTerm(1, AT_, "spatial dimension not implemented for m_variableNames");
      break;
    }
  }

  // fill up the rest of the variales for the species
  if(nDim + 2 < CV->noVariables) {
    m_variableNames[nDim + 2] = "rhoZ";
    for(MInt i = nDim + 2; i < CV->noVariables; ++i) {
      stringstream number;
      number << i - (nDim + 2);
      MString varName = "rho" + number.str();
      m_variableNames[i] = varName;
    }
  }

  mAlloc(m_pvariableNames, m_maxNoVariables, "m_pvariableNames", AT_);
  switch(nDim) {
    case 1: {
      m_pvariableNames[PV->U] = "u";
      m_pvariableNames[PV->P] = "p";
      m_pvariableNames[PV->RHO] = "rho";
      break;
    }
    case 2: {
      m_pvariableNames[PV->U] = "u";
      m_pvariableNames[PV->V] = "v";
      m_pvariableNames[PV->P] = "p";
      m_pvariableNames[PV->RHO] = "rho";
      break;
    }
    case 3: {
      m_pvariableNames[PV->U] = "u";
      m_pvariableNames[PV->V] = "v";
      m_pvariableNames[PV->W] = "w";
      m_pvariableNames[PV->P] = "p";
      m_pvariableNames[PV->RHO] = "rho";
      break;
    }
    default: {
      mTerm(1, AT_, "spatial dimension not implemented for m_variableNames");
      break;
    }
  }

  // fill up the rest of the variales for the species
  if(nDim + 2 < m_maxNoVariables) {
    for(MInt i = nDim + 2; i < m_maxNoVariables; ++i) {
      stringstream number;
      number << i - (nDim + 2);
      MString varName = "rans" + number.str();
      m_pvariableNames[i] = varName;
    }
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section residualOutputInterval
    <code>MInt FvStructuredSolver::m_residualOutputInterval </code>\n
    default = <code> 0</code>\n \n
    Interval for the Residual computation
    Possible values are:\n
    <ul>
    <li>Integer >= 1</li>
    </ul>
    Keywords: <i>RESIDUAL, IO, STRUCTURED</i>
  */
  m_residualOutputInterval = 1;
  m_residualOutputInterval =
      Context::getSolverProperty<MInt>("residualOutputInterval", m_solverId, AT_, &m_residualOutputInterval);

  m_residualFileExist = false;

  /////////////////////////////////////////////////
  ///////////////// Interpolated Points Output ////
  /////////////////////////////////////////////////

  m_intpPointsStart = nullptr;
  m_intpPointsDelta = nullptr;
  m_intpPointsDelta2D = nullptr;
  m_intpPointsNoPoints = nullptr;
  m_intpPointsNoPoints2D = nullptr;
  m_intpPointsCoordinates = nullptr;
  m_intpPointsHasPartnerGlobal = nullptr;
  m_intpPointsHasPartnerLocal = nullptr;
  m_intpPointsVarsGlobal = nullptr;
  m_intpPointsVarsLocal = nullptr;
  m_intpPointsNoPointsTotal = 0;
  m_intpPointsNoLines = 0;
  m_intpPointsNoLines2D = 0;

  /*! \property
    \page propertiesFVSTRCTRD
    \section lineOutputInterval
    <code>MInt FvStructuredSolver::m_intpPointsOutputInterval </code>\n
    default = <code> 0</code>\n \n
    Interval of the line interpolation output.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
  */
  m_intpPointsOutputInterval = 0;
  m_intpPointsOutputInterval =
      Context::getSolverProperty<MInt>("intpPointsOutputInterval", m_solverId, AT_, &m_intpPointsOutputInterval);

  if(m_intpPointsOutputInterval > 0) {
    /*! \property
      \page propertiesFVSTRCTRD
      \section lineOutputDir
      <code>MInt FvStructuredSolver::m_intpPointsOutputDir </code>\n
      default = <code> m_solutionOutput</code>\n \n
      Folder to write the line output files.\n
      Possible values are:\n
      <ul>
      <li>String with path</li>
      </ul>
      Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
    */
    m_intpPointsOutputDir = m_solutionOutput;
    if(Context::propertyExists("intpPointsOutputDir", m_solverId)) {
      m_intpPointsOutputDir = Context::getSolverProperty<MString>("intpPointsOutputDir", m_solverId, AT_);

      MString comparator = "/";
      if(strcmp((m_intpPointsOutputDir.substr(m_intpPointsOutputDir.length() - 1, 1)).c_str(), comparator.c_str())
         != 0) {
        m_intpPointsOutputDir = m_intpPointsOutputDir + "/";
      }
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section lineStartX
      <code>MInt FvStructuredSolver::lineStartX </code>\n
      default = <code> 0</code>\n \n
      Point in x-dir to start line distribution.\n
      Possible values are:\n
      <ul>
      <li>Float</li>
      </ul>
      Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
    */
    MInt noLineStartX = Context::propertyLength("intpPointsStartX", m_solverId);

    /*! \property
      \page propertiesFVSTRCTRD
      \section lineStartY
      <code>MInt FvStructuredSolver::lineStartY </code>\n
      default = <code> 0</code>\n \n
      Point in y-dir to start line distribution.\n
      Possible values are:\n
      <ul>
      <li>Float</li>
      </ul>
      Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
    */
    MInt noLineStartY = Context::propertyLength("intpPointsStartY", m_solverId);

    /*! \property
      \page propertiesFVSTRCTRD
      \section lineStartZ
      <code>MInt FvStructuredSolver::lineStartZ </code>\n
      default = <code> 0</code>\n \n
      Point in z-dir to start line distribution.\n
      Possible values are:\n
      <ul>
      <li>Float</li>
      </ul>
      Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
    */
    MInt noLineStartZ = Context::propertyLength("intpPointsStartZ", m_solverId);

    /*! \property
      \page propertiesFVSTRCTRD
      \section lineDeltaX
      <code>MInt FvStructuredSolver::lineDeltaX </code>\n
      default = <code> 0</code>\n \n
      The delta in x-dir between the line points.\n
      Possible values are:\n
      <ul>
      <li>Float</li>
      </ul>
      Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
    */
    MInt noLineDeltaX = Context::propertyLength("intpPointsDeltaX", m_solverId);

    /*! \property
      \page propertiesFVSTRCTRD
      \section lineDeltaY
      <code>MInt FvStructuredSolver::lineDeltaY </code>\n
      default = <code> 0</code>\n \n
      The delta in y-dir between the line points.\n
      Possible values are:\n
      <ul>
      <li>Float</li>
      </ul>
      Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
    */
    MInt noLineDeltaY = Context::propertyLength("intpPointsDeltaY", m_solverId);

    /*! \property
      \page propertiesFVSTRCTRD
      \section lineDeltaZ
      <code>MInt FvStructuredSolver::lineDeltaZ </code>\n
      default = <code> 0</code>\n \n
      The delta in z-dir between the line points.\n
      Possible values are:\n
      <ul>
      <li>Float</li>
      </ul>
      Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
    */
    MInt noLineDeltaZ = Context::propertyLength("intpPointsDeltaZ", m_solverId);

    /*! \property
      \page propertiesFVSTRCTRD
      \section lineNoPoints
      <code>MInt FvStructuredSolver::lineNoPoints </code>\n
      default = <code> 0</code>\n \n
      Number of the points to distribute.\n
      Possible values are:\n
      <ul>
      <li>Integer > 0</li>
      </ul>
      Keywords: <i>LINES, INTERPOLATION, IO, STRUCTURED</i>
    */
    MInt noLineNoPoints = Context::propertyLength("intpPointsNoPoints", m_solverId);

    m_intpPointsNoLines = noLineStartX;

    if(m_intpPointsNoLines != noLineStartX || m_intpPointsNoLines != noLineStartY || m_intpPointsNoLines != noLineStartZ
       || m_intpPointsNoLines != noLineDeltaX || m_intpPointsNoLines != noLineDeltaY
       || m_intpPointsNoLines != noLineDeltaZ || m_intpPointsNoLines != noLineNoPoints) {
      mTerm(1, AT_,
            "The number of Entries for 'intpPointsStartX', 'intpPointsStartY',"
            "'intpPointsStartZ', 'intpPointsDeltaX', 'intpPointsDeltaY' and "
            "'intpPointsDeltaZ' do not coincide!! Please check");
    }

    // set number of intpPointss in second direction
    if(Context::propertyExists("intpPointsDeltaX2D", m_solverId)
       && Context::propertyExists("intpPointsDeltaY2D", m_solverId)
       && Context::propertyExists("intpPointsDeltaZ2D", m_solverId)
       && Context::propertyExists("intpPointsNoPoints2D", m_solverId)) {
      /*! \property
  \page propertiesFVSTRCTRD
        \section intpPointsDeltaX2D
        <code>MInt FvStructuredSolver::intpPointsDeltaX2d </code>\n
        default = <code> 0</code>\n \n
        The delta in x-dir for the second\n
        dimension if 2d field interpolation is desired.\n
        Possible values are:\n
        <ul>
        <li>Float</li>
        </ul>
        Keywords: <i>FIELDS, INTERPOLATION, IO, STRUCTURED</i>
      */
      MInt noLineDeltaX2d = Context::propertyLength("intpPointsDeltaX2D", m_solverId);

      /*! \property
  \page propertiesFVSTRCTRD
        \section lineDeltaY2D
        <code>MInt FvStructuredSolver::lineDeltaY2d </code>\n
        default = <code> 0</code>\n \n
        The delta in y-dir for the second\n
        dimension if 2d field interpolation is desired.\n
        Possible values are:\n
        <ul>
        <li>Float</li>
        </ul>
        Keywords: <i>FIELDS, INTERPOLATION, IO, STRUCTURED</i>
      */
      MInt noLineDeltaY2d = Context::propertyLength("intpPointsDeltaY2D", m_solverId);

      /*! \property
  \page propertiesFVSTRCTRD
        \section lineDeltaZ2D
        <code>MInt FvStructuredSolver::lineDeltaZ2d </code>\n
        default = <code> 0</code>\n \n
        The delta in z-dir for the second\n
        dimension if 2d field interpolation is desired.\n
        Possible values are:\n
        <ul>
        <li>Float</li>
        </ul>
        Keywords: <i>FIELDS, INTERPOLATION, IO, STRUCTURED</i>
      */
      MInt noLineDeltaZ2d = Context::propertyLength("intpPointsDeltaZ2D", m_solverId);

      /*! \property
  \page propertiesFVSTRCTRD
        \section lineNoPoints2D
        <code>MInt FvStructuredSolver::lineNoPoints2D </code>\n
        default = <code> 0</code>\n \n
        The number of points for the second\n
        dimension if 2d field interpolation is desired.\n
        Possible values are:\n
        <ul>
        <li>Integer > 0</li>
        </ul>
        Keywords: <i>FIELDS, INTERPOLATION, IO, STRUCTURED</i>
      */
      MInt noLineNoPoints2D = Context::propertyLength("intpPointsNoPoints2D", m_solverId);

      m_intpPointsNoLines2D = noLineNoPoints2D;

      if(m_intpPointsNoLines2D != noLineDeltaX2d || m_intpPointsNoLines2D != noLineDeltaY2d
         || m_intpPointsNoLines2D != noLineDeltaZ2d || m_intpPointsNoLines2D != noLineNoPoints2D) {
        cout << "no2dLines: " << m_intpPointsNoLines2D << " lineDeltaY2D: " << noLineDeltaY2d
             << "  lineDeltaX2D: " << noLineDeltaX2d << "  lineDeltaZ2D: " << noLineDeltaZ2d << endl;
        mTerm(1, AT_,
              "The number of Entries for 'lineDeltaX2D', 'lineDeltaY2D' and 'lineDeltaZ2D' do not coincide!! "
              "Please check");
      }

    } else {
      m_intpPointsNoLines2D = m_intpPointsNoLines; // just to solve allocating problems
    }

    mAlloc(m_intpPointsStart, nDim, m_intpPointsNoLines, "m_intpPointsStart", F0, AT_);
    mAlloc(m_intpPointsDelta, nDim, m_intpPointsNoLines, "m_intpPointsDelta", F0, AT_);
    mAlloc(m_intpPointsNoPoints, m_intpPointsNoLines, "m_intpPointsNoPoints", 0, AT_);
    mAlloc(m_intpPointsNoPoints2D, m_intpPointsNoLines2D, "m_intpPointsNoPoints2D", 0, AT_);
    mAlloc(m_intpPointsDelta2D, nDim, m_intpPointsNoLines2D, "m_intpPointsDelta2D", F0, AT_);
    mAlloc(m_intpPointsOffsets, m_intpPointsNoLines, "m_intpPointsOffsets", 0, AT_);

    // for every "first direction" line there is only one field although there are more possibilities to combine the
    // "first and second direction" lines therefore the fieldOffset is equally sized to the number of "first direction"
    // lines (m_intpPointsNoLines)

    // read in the values for the startpoints and directionVectors

    // first direction
    for(MInt i = 0; i < m_intpPointsNoLines; ++i) {
      // initialize with unrealistic values
      for(MInt dim = 0; dim < nDim; dim++) {
        m_intpPointsStart[dim][i] = -99999.9;
        m_intpPointsDelta[dim][i] = -99999.9;
      }
      m_intpPointsNoPoints[i] = -1;

      m_intpPointsStart[0][i] =
          Context::getSolverProperty<MFloat>("intpPointsStartX", m_solverId, AT_, &m_intpPointsStart[0][i], i);
      m_intpPointsStart[1][i] =
          Context::getSolverProperty<MFloat>("intpPointsStartY", m_solverId, AT_, &m_intpPointsStart[1][i], i);
      m_intpPointsStart[2][i] =
          Context::getSolverProperty<MFloat>("intpPointsStartZ", m_solverId, AT_, &m_intpPointsStart[2][i], i);
      m_intpPointsDelta[0][i] =
          Context::getSolverProperty<MFloat>("intpPointsDeltaX", m_solverId, AT_, &m_intpPointsDelta[0][i], i);
      m_intpPointsDelta[1][i] =
          Context::getSolverProperty<MFloat>("intpPointsDeltaY", m_solverId, AT_, &m_intpPointsDelta[1][i], i);
      m_intpPointsDelta[2][i] =
          Context::getSolverProperty<MFloat>("intpPointsDeltaZ", m_solverId, AT_, &m_intpPointsDelta[2][i], i);
      m_intpPointsNoPoints[i] =
          Context::getSolverProperty<MInt>("intpPointsNoPoints", m_solverId, AT_, &m_intpPointsNoPoints[i], i);
    }

    m_intpPoints = false;
    // second direction
    if(Context::propertyExists("intpPointsDeltaX2D", m_solverId)
       && Context::propertyExists("intpPointsDeltaY2D", m_solverId)
       && Context::propertyExists("intpPointsDeltaZ2D", m_solverId)
       && Context::propertyExists("intpPointsNoPoints2D", m_solverId)) {
      m_intpPoints = true;

      for(MInt i = 0; i < m_intpPointsNoLines2D; ++i) {
        // initialize with unrealistic values
        for(MInt dim = 0; dim < nDim; dim++) {
          m_intpPointsDelta2D[dim][i] = -99999.9;
        }
        m_intpPointsNoPoints2D[i] = -1;

        m_intpPointsDelta2D[0][i] =
            Context::getSolverProperty<MFloat>("intpPointsDeltaX2D", m_solverId, AT_, &m_intpPointsDelta2D[0][i], i);
        m_intpPointsDelta2D[1][i] =
            Context::getSolverProperty<MFloat>("intpPointsDeltaY2D", m_solverId, AT_, &m_intpPointsDelta2D[1][i], i);
        m_intpPointsDelta2D[2][i] =
            Context::getSolverProperty<MFloat>("intpPointsDeltaZ2D", m_solverId, AT_, &m_intpPointsDelta2D[2][i], i);
        m_intpPointsNoPoints2D[i] =
            Context::getSolverProperty<MInt>("intpPointsNoPoints2D", m_solverId, AT_, &m_intpPointsNoPoints2D[i], i);
      }

    } else {
      for(MInt i = 0; i < m_intpPointsNoLines2D; i++) {
        m_intpPointsNoPoints2D[i] = 1;
      }
      for(MInt fieldId = 0; fieldId < m_intpPointsNoLines; fieldId++) {
        for(MInt dim = 0; dim < nDim; dim++) {
          m_intpPointsDelta2D[dim][fieldId] = 0;
        }
      }
    }

    for(MInt i = 0; i < m_intpPointsNoLines; i++) {
      m_intpPointsNoPointsTotal += m_intpPointsNoPoints[i] * m_intpPointsNoPoints2D[i];
    }

    mAlloc(m_intpPointsCoordinates, nDim, m_intpPointsNoPointsTotal, "m_intpPointsCoordinates", F0, AT_);
    mAlloc(m_intpPointsHasPartnerGlobal, m_intpPointsNoPointsTotal, "m_intpPointsHasPartnerGlobal", 0, AT_);
    mAlloc(m_intpPointsHasPartnerLocal, m_intpPointsNoPointsTotal, "m_intpPointsHasPartnerLocal", 0, AT_);
    mAlloc(m_intpPointsVarsLocal, CV->noVariables, m_intpPointsNoPointsTotal, "m_intpPointsVarsLocal", F0, AT_);
    mAlloc(m_intpPointsVarsGlobal, CV->noVariables, m_intpPointsNoPointsTotal, "m_intpPointsVarsGlobal", F0, AT_);

    // all fields are in ONE array (m_intpPointsCoordinates)
    MInt offset = 0;

    // putting startpoints at the right place and generating the field
    for(MInt fieldId = 0; fieldId < m_intpPointsNoLines; fieldId++) {
      m_intpPointsOffsets[fieldId] = offset;

      for(MInt pointId2d = 0; pointId2d < m_intpPointsNoPoints2D[fieldId]; pointId2d++) {
        for(MInt pointId = 0; pointId < m_intpPointsNoPoints[fieldId]; pointId++) {
          for(MInt dim = 0; dim < nDim; dim++) {
            m_intpPointsCoordinates[dim][offset] = m_intpPointsStart[dim][fieldId]
                                                   + pointId * m_intpPointsDelta[dim][fieldId]
                                                   + pointId2d * m_intpPointsDelta2D[dim][fieldId];
          }
          offset++;
        }
      }
    }

    // Finished reading in the line output properties and generating the lines
  }


  /////////////////////////////////////////////////
  ///////////////// Box Output ////////////////////
  /////////////////////////////////////////////////

  // this method writes out a defined sub-volume of the
  // computational domain. The subvolume is defined by
  // an starting point/offset coordinate (i,j,k) and
  // a size (also in computational coordinates)
  // The cell-center coordinates can also be written out
  // which is useful for visualization, especially for
  // moving grids

  /*! \property
    \page propertiesFVSTRCTRD
    \section boxOutputInterval
    <code>MInt FvStructuredSolver::boxOutputInterval </code>\n
    default = <code> 0</code>\n \n
    Interval to write out the box output files.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
  */
  m_boxOutputInterval = 0;
  m_boxOutputInterval = Context::getSolverProperty<MInt>("boxOutputInterval", m_solverId, AT_, &m_boxOutputInterval);

  if(m_boxOutputInterval > 0) {
    /*! \property
      \page propertiesFVSTRCTRD
      \section boxBlock
      <code>MInt FvStructuredSolver::boxBlocks </code>\n
      default = <code> 0</code>\n \n
      Solvers in which the box is contained.\n
      Possible values are:\n
      <ul>
      <li>Integer >= 0</li>
      </ul>
      Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
    */
    m_boxNoBoxes = Context::propertyLength("boxBlock", m_solverId); // number of Boxes to be written out

    /*! \property
      \page propertiesFVSTRCTRD
      \section boxWriteCoordinates
      <code>MInt FvStructuredSolver::m_boxWriteCoordinates </code>\n
      default = <code> 0</code>\n \n
      Write cell-center coordinates into the boxes.\n
      Possible values are:\n
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
    */
    m_boxWriteCoordinates = false;
    m_boxWriteCoordinates =
        Context::getSolverProperty<MBool>("boxWriteCoordinates", m_solverId, AT_, &m_boxWriteCoordinates);

    /*! \property
      \page propertiesFVSTRCTRD
      \section boxOutputDir
      <code>MInt FvStructuredSolver::m_boxOutputDir </code>\n
      default = <code> m_solutionOutput</code>\n \n
      Output folder to write the box output files.\n
      Possible values are:\n
      <ul>
      <li>String containing path</li>
      </ul>
      Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
    */
    m_boxOutputDir = m_solutionOutput;
    if(Context::propertyExists("boxOutputDir", m_solverId)) {
      m_boxOutputDir = Context::getSolverProperty<MString>("boxOutputDir", m_solverId, AT_);

      MString comparator = "/";
      if(strcmp((m_boxOutputDir.substr(m_boxOutputDir.length() - 1, 1)).c_str(), comparator.c_str()) != 0) {
        m_boxOutputDir = m_boxOutputDir + "/";
      }
    }

    mAlloc(m_boxBlock, m_boxNoBoxes, "m_boxBlock", 0, AT_);
    mAlloc(m_boxOffset, m_boxNoBoxes, nDim, "m_boxOffset", 0, AT_);
    mAlloc(m_boxSize, m_boxNoBoxes, nDim, "m_boxSize", 0, AT_);

    for(MInt i = 0; i < m_boxNoBoxes; ++i) {
      m_boxBlock[i] = -1;

      for(MInt dim = 0; dim < nDim; dim++) {
        m_boxOffset[i][dim] = -1;
        m_boxSize[i][dim] = -1;
      }

      m_boxBlock[i] = Context::getSolverProperty<MInt>("boxBlock", m_solverId, AT_, &m_boxBlock[i], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section boxOffsetK
        <code>MInt FvStructuredSolver::m_boxOffsetK </code>\n
        default = <code> 0</code>\n \n
        Offset of the box in K-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      IF_CONSTEXPR(nDim == 3) {
        m_boxOffset[i][0] = Context::getSolverProperty<MInt>("boxOffsetK", m_solverId, AT_, &m_boxOffset[i][0], i);
      }

      /*! \property
  \page propertiesFVSTRCTRD
        \section boxOffsetJ
        <code>MInt FvStructuredSolver::m_boxOffsetJ </code>\n
        default = <code> 0</code>\n \n
        Offset of the box in J-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
      */

      m_boxOffset[i][nDim - 2] = Context::getSolverProperty<MInt>("boxOffsetJ", m_solverId, AT_, &m_boxOffset[i][1], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section boxOffsetI
        <code>MInt FvStructuredSolver::m_boxOffsetI </code>\n
        default = <code> 0</code>\n \n
        Offset of the box in I-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_boxOffset[i][nDim - 1] = Context::getSolverProperty<MInt>("boxOffsetI", m_solverId, AT_, &m_boxOffset[i][2], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section boxSizeK
        <code>MInt FvStructuredSolver::m_boxSizeK </code>\n
        default = <code> 0</code>\n \n
        Size of the box in K-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer > 0</li>
        </ul>
        Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      IF_CONSTEXPR(nDim == 3) {
        m_boxSize[i][0] = Context::getSolverProperty<MInt>("boxSizeK", m_solverId, AT_, &m_boxSize[i][0], i);
      }

      /*! \property
  \page propertiesFVSTRCTRD
        \section boxSizeJ
        <code>MInt FvStructuredSolver::m_boxSizeJ </code>\n
        default = <code> 0</code>\n \n
        Size of the box in J-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer > 0</li>
        </ul>
        Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_boxSize[i][nDim - 2] = Context::getSolverProperty<MInt>("boxSizeJ", m_solverId, AT_, &m_boxSize[i][1], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section boxSizeI
        <code>MInt FvStructuredSolver::m_boxSizeI </code>\n
        default = <code> 0</code>\n \n
        Size of the box in I-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer > 0</li>
        </ul>
        Keywords: <i>BOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_boxSize[i][nDim - 1] = Context::getSolverProperty<MInt>("boxSizeI", m_solverId, AT_, &m_boxSize[i][2], i);
    }
  }


  /////////////////////////////////////////////////
  ///////////////// Nodal Box Output //////////////
  /////////////////////////////////////////////////


  // this method is similar to the standard box output
  // such that you choose a subvolume of your domain to
  // write out at a given sampling rate. Here, the variables
  // are interpolated to the grid-nodes which are then written out
  // instead of the cell-center values (as in the standard box method)

  /*! \property
    \page propertiesFVSTRCTRD
    \section nodalBoxOutputInterval
    <code>MInt FvStructuredSolver::nodalBoxOutputInterval </code>\n
    default = <code> 0</code>\n \n
    Interval to write out the nodalBox output files.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
  */
  m_nodalBoxOutputInterval = 0;
  m_nodalBoxOutputInterval =
      Context::getSolverProperty<MInt>("nodalBoxOutputInterval", m_solverId, AT_, &m_nodalBoxOutputInterval);

  if(m_nodalBoxOutputInterval > 0) {
    m_nodalBoxInitialized = false;
    m_nodalBoxTotalLocalSize = -1;

    /*! \property
      \page propertiesFVSTRCTRD
      \section nodalBoxBlock
      <code>MInt FvStructuredSolver::nodalBoxBlocks </code>\n
      default = <code> 0</code>\n \n
      Solvers in which the nodalBox is contained.\n
      Possible values are:\n
      <ul>
      <li>Integer >= 0</li>
      </ul>
      Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
    */
    m_nodalBoxNoBoxes = Context::propertyLength("nodalBoxBlock", m_solverId); // number of NodalBoxes to be written out

    /*! \property
      \page propertiesFVSTRCTRD
      \section nodalBoxWriteCoordinates
      <code>MInt FvStructuredSolver::m_nodalBoxWriteCoordinates </code>\n
      default = <code> 0</code>\n \n
      Write cell-center coordinates into the nodalBoxes.\n
      Possible values are:\n
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
    */
    m_nodalBoxWriteCoordinates = false;
    m_nodalBoxWriteCoordinates =
        Context::getSolverProperty<MBool>("nodalBoxWriteCoordinates", m_solverId, AT_, &m_nodalBoxWriteCoordinates);

    /*! \property
      \page propertiesFVSTRCTRD
      \section nodalBoxOutputDir
      <code>MInt FvStructuredSolver::m_nodalBoxOutputDir </code>\n
      default = <code> m_solutionOutput</code>\n \n
      Output folder to write the nodalBox output files.\n
      Possible values are:\n
      <ul>
      <li>String containing path</li>
      </ul>
      Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
    */
    m_nodalBoxOutputDir = m_solutionOutput;
    if(Context::propertyExists("nodalBoxOutputDir", m_solverId)) {
      m_nodalBoxOutputDir = Context::getSolverProperty<MString>("nodalBoxOutputDir", m_solverId, AT_);

      MString comparator = "/";
      if(strcmp((m_nodalBoxOutputDir.substr(m_nodalBoxOutputDir.length() - 1, 1)).c_str(), comparator.c_str()) != 0) {
        m_nodalBoxOutputDir = m_nodalBoxOutputDir + "/";
      }
    }

    mAlloc(m_nodalBoxBlock, m_nodalBoxNoBoxes, "m_nodalBoxBlock", 0, AT_);
    mAlloc(m_nodalBoxOffset, m_nodalBoxNoBoxes, nDim, "m_nodalBoxOffset", 0, AT_);
    mAlloc(m_nodalBoxPoints, m_nodalBoxNoBoxes, nDim, "m_nodalBoxPoints", 0, AT_);

    for(MInt i = 0; i < m_nodalBoxNoBoxes; ++i) {
      m_nodalBoxBlock[i] = -1;

      for(MInt dim = 0; dim < nDim; dim++) {
        m_nodalBoxOffset[i][dim] = -1;
        m_nodalBoxPoints[i][dim] = -1;
      }

      m_nodalBoxBlock[i] = Context::getSolverProperty<MInt>("nodalBoxBlock", m_solverId, AT_, &m_nodalBoxBlock[i], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section nodalBoxOffsetK
        <code>MInt FvStructuredSolver::m_nodalBoxOffsetK </code>\n
        default = <code> 0</code>\n \n
        Offset of the nodalBox in K-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_nodalBoxOffset[i][0] =
          Context::getSolverProperty<MInt>("nodalBoxOffsetK", m_solverId, AT_, &m_nodalBoxOffset[i][0], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section nodalBoxOffsetJ
        <code>MInt FvStructuredSolver::m_nodalBoxOffsetJ </code>\n
        default = <code> 0</code>\n \n
        Offset of the nodalBox in J-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_nodalBoxOffset[i][1] =
          Context::getSolverProperty<MInt>("nodalBoxOffsetJ", m_solverId, AT_, &m_nodalBoxOffset[i][1], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section nodalBoxOffsetI
        <code>MInt FvStructuredSolver::m_nodalBoxOffsetI </code>\n
        default = <code> 0</code>\n \n
        Offset of the nodalBox in I-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_nodalBoxOffset[i][2] =
          Context::getSolverProperty<MInt>("nodalBoxOffsetI", m_solverId, AT_, &m_nodalBoxOffset[i][2], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section nodalBoxPointsK
        <code>MInt FvStructuredSolver::m_nodalBoxPointsK </code>\n
        default = <code> 0</code>\n \n
        Size of the nodalBox in K-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer > 0</li>
        </ul>
        Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_nodalBoxPoints[i][0] =
          Context::getSolverProperty<MInt>("nodalBoxPointsK", m_solverId, AT_, &m_nodalBoxPoints[i][0], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section nodalBoxPointsJ
        <code>MInt FvStructuredSolver::m_nodalBoxPointsJ </code>\n
        default = <code> 0</code>\n \n
        Size of the nodalBox in J-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer > 0</li>
        </ul>
        Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_nodalBoxPoints[i][1] =
          Context::getSolverProperty<MInt>("nodalBoxPointsJ", m_solverId, AT_, &m_nodalBoxPoints[i][1], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section nodalBoxPointsI
        <code>MInt FvStructuredSolver::m_nodalBoxPointsI </code>\n
        default = <code> 0</code>\n \n
        Size of the nodalBox in I-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer > 0</li>
        </ul>
        Keywords: <i>NODALBOXES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_nodalBoxPoints[i][2] =
          Context::getSolverProperty<MInt>("nodalBoxPointsI", m_solverId, AT_, &m_nodalBoxPoints[i][2], i);
    }
  }


  /////////////////////////////////////////////////
  ///////// ASCII Point Interpolation Output //////
  /////////////////////////////////////////////////


  // this method is useful for high-frequency writing values of
  // given points to an ASCII file. The points can be given by
  // physical coordinates onto which one of the chosen variables
  // is then interpolated to

  /*! \property
    \page propertiesFVSTRCTRD
    \section asciiCellOutputInterval
    <code>MInt FvStructuredSolver::m_pointsToAsciiOutputInterval </code>\n
    default = <code>"./out"</code>\n \n
    Interval in which the integrated forces .\n
    should be written to an ASCII file.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_pointsToAsciiOutputInterval = 0;
  m_pointsToAsciiOutputInterval =
      Context::getSolverProperty<MInt>("pointsToAsciiOutputInterval", m_solverId, AT_, &m_pointsToAsciiOutputInterval);


  /*! \property
    \page propertiesFVSTRCTRD
    \section asciiCellComputeInterval
    <code>MInt FvStructuredSolver::m_pointsToAsciiComputeInterval </code>\n
    default = <code>"./out"</code>\n \n
    Interval in which the integrated  .\n
    should be computed.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>FORCES, IO, STRUCTURED</i>
  */
  m_pointsToAsciiComputeInterval = 0;
  m_pointsToAsciiComputeInterval = Context::getSolverProperty<MInt>("pointsToAsciiComputeInterval", m_solverId, AT_,
                                                                    &m_pointsToAsciiComputeInterval);

  if(m_pointsToAsciiComputeInterval > m_pointsToAsciiOutputInterval) {
    m_pointsToAsciiComputeInterval = m_pointsToAsciiOutputInterval;
  }

  if(m_pointsToAsciiOutputInterval > 0 && m_pointsToAsciiComputeInterval == 0) {
    m_pointsToAsciiComputeInterval = 1;
  }

  if(m_pointsToAsciiOutputInterval > 0) {
    m_pointsToAsciiVarId = 0;
    m_pointsToAsciiVarId =
        Context::getSolverProperty<MInt>("pointsToAsciiVarId", m_solverId, AT_, &m_pointsToAsciiVarId);

    m_pointsToAsciiLastOutputStep = 0;
    m_pointsToAsciiLastComputationStep = 0;

    m_pointsToAsciiNoPoints = Context::propertyLength("pointsToAsciiX", m_solverId);

    mAlloc(m_pointsToAsciiCoordinates, nDim, m_pointsToAsciiNoPoints, "m_pointsToAsciiCoordinates", 0.0, AT_);
    mAlloc(m_pointsToAsciiHasPartnerLocal, m_pointsToAsciiNoPoints, "m_pointsToAsciiHasPartnerLocal", 0, AT_);
    mAlloc(m_pointsToAsciiHasPartnerGlobal, m_pointsToAsciiNoPoints, "m_pointsToAsciiHasPartnerGlobal", 0, AT_);
    mAlloc(m_pointsToAsciiVars, m_pointsToAsciiOutputInterval, m_pointsToAsciiNoPoints + 3, "m_pointsToAsciiVars", 0.0,
           AT_);

    if(domainId() == 0) {
      cout << "noAsciiCells: " << m_pointsToAsciiNoPoints << endl;
    }

    for(MInt i = 0; i < m_pointsToAsciiNoPoints; ++i) {
      for(MInt dim = 0; dim < nDim; dim++) {
        m_pointsToAsciiCoordinates[dim][i] = -1.0;
      }

      /*! \property
  \page propertiesFVSTRCTRD
        \section pointsToAsciiCoordinatesX
        <code>MInt FvStructuredSolver::m_pointsToAsciiX </code>\n
        default = <code> 0</code>\n \n
        Offset of the asciiCell in K-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>ASCIICELLES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_pointsToAsciiCoordinates[0][i] =
          Context::getSolverProperty<MFloat>("pointsToAsciiX", m_solverId, AT_, &m_pointsToAsciiCoordinates[0][i], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section pointsToAsciiCoordinatesY
        <code>MInt FvStructuredSolver::m_pointsToAsciiY </code>\n
        default = <code> 0</code>\n \n
        Offset of the asciiCell in J-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>ASCIICELLES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_pointsToAsciiCoordinates[1][i] =
          Context::getSolverProperty<MFloat>("pointsToAsciiY", m_solverId, AT_, &m_pointsToAsciiCoordinates[1][i], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section pointsToAsciiCoordinatesZ
        <code>MInt FvStructuredSolver::m_pointsToAsciiZ </code>\n
        default = <code> 0</code>\n \n
        Offset of the asciiCell in I-dir.\n
        Possible values are:\n
        <ul>
        <li>Integer >= 0</li>
        </ul>
        Keywords: <i>ASCIICELLES, INTERPOLATION, IO, STRUCTURED</i>
      */
      m_pointsToAsciiCoordinates[2][i] =
          Context::getSolverProperty<MFloat>("pointsToAsciiZ", m_solverId, AT_, &m_pointsToAsciiCoordinates[2][i], i);
    }
  }


  /////////////////////////////////////////////////
  ////////// Convective Unit Output ///////////////
  /////////////////////////////////////////////////

  /*! \property
    \page propertiesFVSTRCTRD
    \section useConvectiveUnitWrite
    <code>MInt FvStructuredSolver::m_useConvectiveUnitWrite </code>\n
    default = <code> 0</code>\n \n
    Solution interval controlled by fraction or multiple of convective unit\n
    instead of globalTimeStep.\n
    Possible values are:\n
    <ul>
    <li>0: off</li>
    <li>1: on</li>
    </ul>
    Keywords: <i>IO, STRUCTURED</i>
  */
  m_useConvectiveUnitWrite = false;
  m_useConvectiveUnitWrite =
      Context::getSolverProperty<MBool>("useConvectiveUnitWrite", m_solverId, AT_, &m_useConvectiveUnitWrite);

  if(m_useConvectiveUnitWrite) {
    /*! \property
      \page propertiesFVSTRCTRD
      \section convectiveUnitInterval
      <code>MInt FvStructuredSolver::convectiveUnitInterval </code>\n
      default = <code> 0</code>\n \n
      Solution interval controlled by fraction or multiple of convective unit\n
      instead of globalTimeStep.\n
      Possible values are:\n
      <ul>
      <li>floating point number > 0.0 </li>
      </ul>
      Keywords: <i>IO, STRUCTURED</i>
    */
    m_convectiveUnitInterval = 1.0;
    m_convectiveUnitInterval =
        Context::getSolverProperty<MFloat>("convectiveUnitInterval", m_solverId, AT_, &m_convectiveUnitInterval);

    m_noConvectiveOutputs = 0;

    /*! \property
      \page propertiesFVSTRCTRD
      \section sampleSolutionFiles
      <code>MInt FvStructuredSolver::sampleSolutionFiles </code>\n
      default = <code> 0</code>\n \n
      Trigger the output of solution files controlled\n
      by the convectiveUnitInterval.\n
      Possible values are:\n
      <ul>
      <li>0: off</li>
      <li>1: on</li>
      </ul>
      Keywords: <i>IO, STRUCTURED</i>
    */
    m_sampleSolutionFiles = false;
    m_sampleSolutionFiles =
        Context::getSolverProperty<MBool>("sampleSolutionFiles", m_solverId, AT_, &m_sampleSolutionFiles);
  }

  /////////////////////////////////////////////////
  ////////// Interpolation restart  ///////////////
  /////////////////////////////////////////////////
  /*! \property
    \page propertiesFVSTRCTRD
    \section restartInterpolation
    <code>MInt FvStructuredSolver::m_restartInterpolation </code>\n
    default = <code> 0</code>\n \n
    Restart the computation with an interpolated field\n
    from a given donorVars/Grid.\n
    Possible values are:\n
    <ul>
    <li>0: off</li>
    <li>1: on</li>
    </ul>
    Keywords: <i>INTERPOLATION, IO, STRUCTURED</i>
  */
  m_restartInterpolation = false;
  m_restartInterpolation =
      Context::getSolverProperty<MBool>("restartInterpolation", m_solverId, AT_, &m_restartInterpolation);
}

/**
 * \brief Initializes the general properties of the FV Structured solver
 * \author Pascal Meysonnat
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::initializeFvStructuredSolver(MBool* propertiesGroups) {
  TRACE();

  (void)propertiesGroups;

  /*! \property
    \page propertiesFVSTRCTRD
    \section noSpecies
    <code>MInt FvStructuredSolver::m_noSpecies </code>\n
    default = <code> 0</code>\n
    Number of species for future species computation.\n
    Possible values are:\n
    <ul>
    <li>0: off</li>
    <li>1: on</li>
    </ul>
    Keywords: <i>SPECIES, STRUCTURED</i>
  */
  m_noSpecies = 0;
  m_noSpecies = Context::getSolverProperty<MInt>("noSpecies", m_solverId, AT_, &m_noSpecies);

  FQ = make_unique<StructuredFQVariables>();

  // allocate the array for counting the needed fq fields
  mAlloc(FQ->neededFQVariables, FQ->maxNoFQVariables, "FQ->neededFQVariables", 0, AT_);
  mAlloc(FQ->outputFQVariables, FQ->maxNoFQVariables, "FQ->outputFQVariables", true, AT_);
  mAlloc(FQ->boxOutputFQVariables, FQ->maxNoFQVariables, "FQ->boxOutputFQVariables", false, AT_);

  m_timeStep = 0;
  m_periodicConnection = 0;
}

/**
 * \brief Reads and initializes properties associated with the Testcase.
 * \author Pascal Meysonnat
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::setTestcaseProperties() {
  /*! \property
    \page propertiesFVSTRCTRD
    \section referenceLength
    <code>MFloat FvStructuredSolver::m_Pr </code>\n
    default = <code>1.0</code>\n \n
    WARNING: Do NOT use any value different than 1.0 - The correct implementation of this is not checked,
    so it probably will not do what you think it does/should do. Don't use it unless you REALLY know what you are doing.
    Reference Length L - The length = 1.0 of the grid is scaled with L.
    Possible values are:
    <ul>
    <li>1.0 +- eps</li>
    </ul>
    Keywords: <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_referenceLength = F1;
  m_referenceLength = Context::getSolverProperty<MFloat>("referenceLength", m_solverId, AT_, &m_referenceLength);

  if(fabs(m_referenceLength - F1) > m_eps) {
    m_log << "WARNING: referenceLength != 1.0. The correct implementation of this is not checked. Don't use it "
             "unless you REALLY know what you are doing."
          << endl;
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section physicalReferenceLength
    <code>MFloat FvStructuredSolver::m_Pr </code>\n
    default = <code>1.0</code>\n \n
    Physical Reference Length L - TODO labels:FV !
    Possible values are:
    <ul>
    <li>Non-negative floating point values</li>
    </ul>
    Keywords: <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_physicalReferenceLength = F1;
  m_physicalReferenceLength =
      Context::getSolverProperty<MFloat>("physicalReferenceLength", m_solverId, AT_, &m_physicalReferenceLength);

  /*! \property
    \page propertiesFVSTRCTRD
    \section Re
    <code>MFloat FvStructuredSolver::m_Re </code>\n
    default = <code>no default value</code>\n \n
    Reynolds number is defined with your infinity variables. \n
    In the code the Reynolds number is nondimensionalized to a Reynolds number based on the stagnation variables a_0,
    mu_0, rho_0 \n \f$ Re_{0} = Re_{\infty}  \frac{\mu_{\infty}}{\rho_{\infty} Ma \sqrt{T_{\infty}} } = \frac{\rho_0 a_0
    l}{\mu_{0}}\f$: <ul> <li> \f$ mu_{\infty} \f$,  \f$ mu_{0} \f$ - viscosity  by the infinity, stagnation temperature
    </li> <li> \f$ Ma \f$ is the mach number </li> <li> \f$ T_{\infty} \f$ is the infinity temperature (free stream
    temperature)</li>
    </ul>
    possible values are:
    <ul>
    <li>Non-negative floating point values of the order of 0.1</li>
    </ul>
    Keywords: <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_Re = Context::getSolverProperty<MFloat>("Re", m_solverId, AT_) / m_referenceLength;

  /*! \property
    \page propertiesFVSTRCTRD
    \section Pr
    <code>MFloat FvStructuredSolver::m_Pr </code>\n
    default = <code>0.72</code>\n \n
    Prandtl number  - non-dimensionalized with stagnant flow conditions
    \f$ \mu_{0},  \lambda_{0}, c_{p} \f$:
    possible values are:
    <ul>
    <li>Non-negative floating point values</li>
    </ul>
    Keywords: <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_Pr = 0.72;
  m_Pr = Context::getSolverProperty<MFloat>("Pr", m_solverId, AT_, &m_Pr);
  m_rPr = 1. / m_Pr;

  /*! \property
    \page propertiesFVSTRCTRD
    \section ReTau
    <code>MFloat FvStructuredSolver::m_ReTau </code>\n
    default = <code>no default value</code>\n \n
    ReTau value for certain flows as channel or pipe flow\n
    to control the pressure gradient.\n
    possible values are:
    <ul>
    <li>Non-negative floating point values</li>
    </ul>
    Keywords <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_ReTau = Context::getSolverProperty<MFloat>("ReTau", m_solverId, AT_) / m_referenceLength;

  /*! \property
    \page propertiesFVSTRCTRD
    \section Ma
    <code>MFloat FvStructuredSolver::m_Ma </code>\n
    default = <code>no default value</code>\n \n
    Mach's number - \f$ M_{\infty} = \frac{u_\infty}{a_\infty} \f$:
    possible values are:
    <ul>
    <li>Non-negative floating point values</li>
    </ul>
    Keywords <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_Ma = Context::getSolverProperty<MFloat>("Ma", m_solverId, AT_);

  /*! \property
    \page propertiesFVSTRCTRD
    \section angle
    <code>MFloat* FvStructuredSolver::m_angle </code>\n
    default = <code>no default value</code>\n \n
    m_angle[nDim] - Angles of rotation around the z, and y axes.
    possible values are:
    <ul>
    <li>Non-negative floating point values</li>
    </ul>
    Keywords: <i>FINITE_VOLUME,BOUNDARY CONDITION</i>
  */
  mAlloc(m_angle, nDim, "m_angle", F0, AT_);
  if(Context::propertyExists("angle", m_solverId)) {
    for(MInt i = 0; i < (nDim - 1); i++) {
      m_angle[i] = Context::getSolverProperty<MFloat>("angle", m_solverId, AT_, i);
      m_angle[i] *= PI / 180.0;
    }
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section considerVolumeForces
    <code>MInt FvStructuredSolver::m_considerVolumeForces </code>\n
    default = <code> 0 </code>\n \n
    Trigger to use volume forces\n
    Possible values are:\n
    <ul>
    <li>0: off</li>
    <li>1: on</li>
    </ul>
    Keywords: <i>FORCING, STRUCTURED</i>
  */
  mAlloc(m_volumeForce, nDim, "m_volumeAcceleration", F0, AT_);
  m_considerVolumeForces = false;
  m_considerVolumeForces =
      Context::getSolverProperty<MBool>("considerVolumeForces", m_solverId, AT_, &m_considerVolumeForces);

  if(m_considerVolumeForces) {
    for(MInt i = 0; i < nDim; i++) {
      /*! \property
  \page propertiesFVSTRCTRD
        \section volumeForce
        <code>MInt FvStructuredSolver::volumeForce </code>\n
        default = <code> 0 </code>\n \n
        Numerical value of the volume force\n
        in each space direction.\n
        Possible values are:\n
        <ul>
        <li>floating point number</li>
        </ul>
        Keywords: <i>FORCING, STRUCTURED</i>
      */
      m_volumeForce[i] = Context::getSolverProperty<MFloat>("volumeForce", m_solverId, AT_, i);
    }
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section gamma
    <code>MFloat FvStructuredSolver::m_gamma </code>\n
    default = <code>1.4</code>\n \n
    Ratio of specific heats - \f$ \gamma = c_p / c_v \f$
    possible values are:
    <ul>
    <li>Non-negative floating point values</li>
    </ul>
    Keywords: <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_gamma = 1.4;
  m_gamma = Context::getSolverProperty<MFloat>("gamma", m_solverId, AT_, &m_gamma);

  m_gammaMinusOne = m_gamma - F1;
  m_fgammaMinusOne = F1 / (m_gammaMinusOne);

  /*! \property
    \page propertiesFVSTRCTRD
    \section initialCondition
    <code>MInt FvStructuredSolver::m_initialCondition </code>\n
    default = <code>no default value</code>\n \n
    Selects the initial condition.
    possible values are:
    <ul>
    <li>See the initialCondition() function of the corresponding solver</li>
    </ul>
    Keywords: <i>INITIAL_CONDITION, FINITE_VOLUME</i>
  */
  m_initialCondition = Context::getSolverProperty<MInt>("initialCondition", m_solverId, AT_);

  m_channelFullyPeriodic = false;
  m_channelFullyPeriodic =
      Context::getSolverProperty<MBool>("channelFullyPeriodic", m_solverId, AT_, &m_channelFullyPeriodic);
  if(m_channelFullyPeriodic && (m_initialCondition == 1233 || m_initialCondition == 1234))
    mTerm(1, "Fully periodic channel computation works with volume forces and not a pressure gradient!");
  m_channelHeight = -F1;
  m_channelWidth = -F1;
  m_channelLength = -F1;
  m_channelInflowPlaneCoordinate = -111111.1111111;
  m_channelC1 = 5.0;
  m_channelC2 = -3.05;
  m_channelC3 = 2.5;
  m_channelC4 = 5.5;
  if(m_channelFullyPeriodic) {
    m_considerVolumeForces = true;
    m_volumeForceMethod = Context::getSolverProperty<MInt>("volumeForceMethod", m_solverId, AT_);
    m_volumeForceUpdateInterval = Context::getSolverProperty<MInt>("volumeForceUpdateInterval", m_solverId, AT_);
  }
  switch(m_initialCondition) {
    case 1233:
    case 1234: { // we are dealing with channel flows

      /*! \property
  \page propertiesFVSTRCTRD
        \section channelHeight
        <code>MInt FvStructuredSolver::m_channelHeigth </code>\n
        default = <code> 1.0 </code>\n \n
        Height of the half of the channel, necessary\n
        to compute correct pressure gradient\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelHeight = Context::getSolverProperty<MFloat>("channelHeight", m_solverId, AT_);

      /*! \property
  \page propertiesFVSTRCTRD
        \section channelWidth
        <code>MInt FvStructuredSolver::m_channelWidth </code>\n
        default = <code> 1.0 </code>\n \n
        Width of the channel, necessary\n
        to compute correct pressure gradient\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelWidth = Context::getSolverProperty<MFloat>("channelWidth", m_solverId, AT_);

      /*! \property
  \page propertiesFVSTRCTRD
        \section channelLength
        <code>MInt FvStructuredSolver::m_channelLength </code>\n
        default = <code> 1.0 </code>\n \n
        Length of the channel, necessary\n
        to compute correct pressure gradient\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelLength = Context::getSolverProperty<MFloat>("channelLength", m_solverId, AT_);

      /*! \property
  \page propertiesFVSTRCTRD
        \section channelInflowCoordinate
        <code>MInt FvStructuredSolver::m_channelInflowCoordinate </code>\n
        default = <code> 1.0 </code>\n \n
        Coordinate of the channel inflow plane.\n
        Possible values are:\n
        <ul>
        <li>Floating point</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelInflowPlaneCoordinate = Context::getSolverProperty<MFloat>("channelInflowCoordinate", m_solverId, AT_);

      /*! \property
  \page propertiesFVSTRCTRD
        \section loglawC1
        <code>MInt FvStructuredSolver::m_channelC1 </code>\n
        default = <code> 5.0 </code>\n \n
        First parameter for log-law for channel initialization\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelC1 = Context::getSolverProperty<MFloat>("loglawC1", m_solverId, AT_, &m_channelC1);

      /*! \property
  \page propertiesFVSTRCTRD
        \section loglawC2
        <code>MInt FvStructuredSolver::m_channelC2 </code>\n
        default = <code> -3.05 </code>\n \n
        Second parameter for log-law for channel initialization\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelC2 = Context::getSolverProperty<MFloat>("loglawC2", m_solverId, AT_, &m_channelC1);

      /*! \property
  \page propertiesFVSTRCTRD
        \section loglawC3
        <code>MInt FvStructuredSolver::m_channelC3 </code>\n
        default = <code> 2.5 </code>\n \n
        Third parameter for log-law for channel initialization\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelC3 = Context::getSolverProperty<MFloat>("loglawC3", m_solverId, AT_, &m_channelC1);

      /*! \property
  \page propertiesFVSTRCTRD
        \section loglawC4
        <code>MInt FvStructuredSolver::m_channelC4 </code>\n
        default = <code> 5.5 </code>\n \n
        Fourth parameter for log-law for channel initialization\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelC4 = Context::getSolverProperty<MFloat>("loglawC4", m_solverId, AT_, &m_channelC1);
      m_log << "============= Channel flow activated =============" << endl;
      m_log << "-> channelHeight: " << m_channelHeight << endl;
      m_log << "-> channelWidth:  " << m_channelWidth << endl;
      m_log << "-> channelLength: " << m_channelLength << endl;
      m_log << "-> channelInflowPlaneCoordinate: " << m_channelInflowPlaneCoordinate << endl;
      m_log << "-> Log law properties: " << endl;
      m_log << "--> C1: " << m_channelC1 << endl;
      m_log << "--> C2: " << m_channelC2 << endl;
      m_log << "--> C3: " << m_channelC3 << endl;
      m_log << "--> C4: " << m_channelC4 << endl;
      m_log << "============= Channel flow summary finished =============" << endl;
      break;
    }
    case 1236: { // we are dealing with pipe flows
      // comment:
      // since channel and pipe have the same boundary conditions, the variable names  are "recycled".
      //  They stay distinguishable in the input property file

      /*! \property
  \page propertiesFVSTRCTRD
        \section pipeDiameter
        <code>MInt FvStructuredSolver::m_channelHeight </code>\n
        default = <code> 1.0 </code>\n \n
        Diameter of the pipe.\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelHeight = Context::getSolverProperty<MFloat>("pipeDiameter", m_solverId, AT_);
      m_channelWidth = m_channelHeight;

      /*! \property
  \page propertiesFVSTRCTRD
        \section pipeLength
        <code>MInt FvStructuredSolver::m_channelLength </code>\n
        default = <code> 1.0 </code>\n \n
        Length of the pipe.\n
        Possible values are:\n
        <ul>
        <li>Floating point > 0.0</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelLength = Context::getSolverProperty<MFloat>("pipeLength", m_solverId, AT_);

      /*! \property
  \page propertiesFVSTRCTRD
        \section pipeInflowCoordinate
        <code>MInt FvStructuredSolver::m_channelInflowPlaneCoordinate </code>\n
        default = <code> 1.0 </code>\n \n
        Coordinate of the channel inflow plane.\n
        Possible values are:\n
        <ul>
        <li>Floating point</li>
        </ul>
        Keywords: <i>CHANNEL, IO, STRUCTURED</i>
      */
      m_channelInflowPlaneCoordinate = Context::getSolverProperty<MFloat>("pipeInflowCoordinate", m_solverId, AT_);
      m_channelC1 = Context::getSolverProperty<MFloat>("loglawC1", m_solverId, AT_, &m_channelC1);
      m_channelC2 = Context::getSolverProperty<MFloat>("loglawC2", m_solverId, AT_, &m_channelC1);
      m_channelC3 = Context::getSolverProperty<MFloat>("loglawC3", m_solverId, AT_, &m_channelC1);
      m_channelC4 = Context::getSolverProperty<MFloat>("loglawC4", m_solverId, AT_, &m_channelC1);
      m_log << "============= Pipe flow activated =============" << endl;
      m_log << "-> pipeRadius: " << m_channelHeight << endl;
      m_log << "-> pipeLength: " << m_channelLength << endl;
      m_log << "-> pipeInflowPlaneCoordinate: " << m_channelInflowPlaneCoordinate << endl;
      m_log << "-> Log law properties: " << endl;
      m_log << "--> C1: " << m_channelC1 << endl;
      m_log << "--> C2: " << m_channelC2 << endl;
      m_log << "--> C3: " << m_channelC3 << endl;
      m_log << "--> C4: " << m_channelC4 << endl;
      m_log << "============= Pipe flow summary finished =============" << endl;
      break;
    }
    default:
      break;
  }


  /*! \property
    \page propertiesFVSTRCTRD
    \section referenceTemperature
    <code>MFloat FvStructuredSolver::m_referenceTemperature </code>\n
    default = <code>273.15</code>\n \n
    Reference temperature \f$ T_{\mathrm{ref}}\f$
    Used to scale the Sutherland's constant as follows: \f$ S/T_{\mathrm{ref}} \f$
    Also used for the computation of the reference sound speed and combustion (TF) related quantities
    possible values are:
    <ul>
    <li>Non-negative floating point values</li>
    </ul>
    Keywords: <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_referenceTemperature = 273.15;
  m_referenceTemperature =
      Context::getSolverProperty<MFloat>("referenceTemperature", m_solverId, AT_, &m_referenceTemperature);

  /*! \property
    \page propertiesFVSTRCTRD
    \section sutherlandConstant
    <code>MFloat FvStructuredSolver::m_sutherlandConstant </code>\n
    default = <code>110.4 K</code>\n \n
    Sutherland's constant. Used by Sutherland's law.
    possible values are:
    <ul>
    <li>Non-negative floating point values</li>
    </ul>
    Keywords: <i>FINITE_VOLUME, VARIABLES</i>
  */
  m_sutherlandConstant = 110.4;
  m_sutherlandConstant =
      Context::getSolverProperty<MFloat>("sutherlandConstant", m_solverId, AT_, &m_sutherlandConstant);
  m_sutherlandConstant /= m_referenceTemperature;
  m_sutherlandPlusOne = m_sutherlandConstant + F1;

  /*! \property
    \page propertiesFVSTRCTRD
    \section tripSandpaper
    <code>MInt FvStructuredSolver::m_useSandpaperTrip </code>\n
    default = <code> false </code>\n \n
    Activate sandpaper trip forcing.\n
    Possible values are:\n
    <ul>
    <li>Bool: True/False</li>
    </ul>
    Keywords: <i>TRIP, BOUNDARYLAYER, STRUCTURED</i>
  */
  m_useSandpaperTrip = false;
  if(Context::propertyExists("tripSandpaper", m_solverId)) {
    m_useSandpaperTrip = Context::getSolverProperty<MBool>("tripSandpaper", m_solverId, AT_, &m_useSandpaperTrip);
  }

  MString govEqs = "NAVIER_STOKES";
  govEqs = Context::getSolverProperty<MString>("govEqs", m_solverId, AT_, &govEqs);

  m_euler = false;
  if(string2enum(govEqs) == EULER) {
    m_euler = true;
  }


  /*! \property
    \page propertiesFVSTRCTRD
    \section fsc
    <code>MInt FvStructuredSolver::m_fsc </code>\n
    default = <code> false </code>\n \n
    Activate the Falkner-Skan-Cooke inflow boundary conditions.\n
    Possible values are:\n
    <ul>
    <li>Bool: True/False</li>
    </ul>
    Keywords: <i>TRIP, BOUNDARYLAYER, STRUCTURED</i>
  */
  m_fsc = 0;
  m_fsc = Context::getSolverProperty<MInt>("fsc", m_solverId, AT_, &m_fsc);
  m_fsc = m_fsc ? 1 : 0;
  if(m_fsc) {
    if(!approx(m_angle[0], F0, MFloatEps))
      mTerm(1, "angle[0] is not zero. Refer to the description of the fsc property");
    m_Re /= cos(m_angle[1]);
    initFsc();
  }

  m_useBlasius = false;
  m_useBlasius = Context::getSolverProperty<MBool>("useBlasius", m_solverId, AT_, &m_useBlasius);
  if(m_useBlasius) {
    initBlasius();
  }
}

/**
 * \brief Reads and initializes properties associated with the Moving Grid Methods
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::setMovingGridProperties() {
  TRACE();

  m_movingGridTimeOffset = F0;
  m_movingGridStepOffset = 0;
  m_movingGridInitialStart = true;
  m_gridMovingMethod = 0;
  m_movingGrid = false;
  m_wallVel = F0;

  m_travelingWave = false;
  m_streamwiseTravelingWave = false;
  m_waveTimeStepComputed = false;
  m_waveSpeed = 0.0;
  m_waveBeginTransition = 0.0;
  m_waveEndTransition = 0.0;
  m_waveRestartFadeIn = false;
  m_waveLength = 0.0;
  m_waveAmplitude = 0.0;
  m_waveTime = 0.0;
  m_waveCellsPerWaveLength = -1;
  m_waveNoStepsPerCell = -1;


  /*! \property
    \page propertiesFVSTRCTRD
    \section movingGrid
    <code>MInt FvStructuredSolver::m_movingGrid </code>\n
    default = <code> 0 </code>\n \n
    Trigger to use moving grid methods\n
    Possible values are:\n
    <ul>
    <li>0: off</li>
    <li>1: on</li>
    </ul>
    Keywords: <i>MOVING, STRUCTURED</i>
  */
  if(Context::propertyExists("movingGrid", m_solverId)) {
    m_movingGrid = Context::getSolverProperty<MBool>("movingGrid", m_solverId, AT_, &m_movingGrid);
  }


  if(m_movingGrid) {
    m_mgExchangeCoordinates = true;
    if(Context::propertyExists("mgExchangeCoordinates", m_solverId)) {
      m_mgExchangeCoordinates =
          Context::getSolverProperty<MBool>("mgExchangeCoordinates", m_solverId, AT_, &m_mgExchangeCoordinates);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section gridMovingMethod
      <code>MInt FvStructuredSolver::m_gridMovingMethod </code>\n
      default = <code> 0 </code>\n \n
      Number of the moving grid function\n
      specified in the switch in fvstructuredsolver3d\n
      Possible values are:\n
      <ul>
      <li>Integer >= 0</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED</i>
    */
    m_gridMovingMethod = Context::getSolverProperty<MInt>("gridMovingMethod", m_solverId, AT_);

    /*! \property
      \page propertiesFVSTRCTRD
      \section wallVel
      <code>MInt FvStructuredSolver::m_wallVel </code>\n
      default = <code> 0 </code>\n \n
      Value of the wall velocity for\n
      certain moving grid methods.\n
      Possible values are:\n
      <ul>
      <li>Floating point</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED</i>
    */
    if(Context::propertyExists("wallVel", m_solverId)) {
      m_wallVel = Context::getSolverProperty<MFloat>("wallVel", m_solverId, AT_);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section movingGridTimeOffset
      <code>MInt FvStructuredSolver::m_movingGridTimeOffset </code>\n
      default = <code> 0 </code>\n \n
      Numerical value of the time offset\n
      for moving grid functions.\n
      Possible values are:\n
      <ul>
      <li>Floating point</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED</i>
    */
    if(Context::propertyExists("movingGridTimeOffset", m_solverId)) {
      m_movingGridTimeOffset =
          Context::getSolverProperty<MFloat>("movingGridTimeOffset", m_solverId, AT_, &m_movingGridTimeOffset);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section movingGridSaveGrid
      <code>MInt FvStructuredSolver::m_movingGridSaveGrid </code>\n
      default = <code> 0 </code>\n \n
      Trigger to write out the moving grid\n
      at each solution time step.\n
      Possible values are:\n
      <ul>
      <li>0: off</li>
      <li>1: on</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED</i>
    */
    m_movingGridSaveGrid = 0;
    if(Context::propertyExists("movingGridSaveGrid", m_solverId)) {
      m_movingGridSaveGrid =
          Context::getSolverProperty<MBool>("movingGridSaveGrid", m_solverId, AT_, &m_movingGridSaveGrid);
    }


    /*! \property
      \page propertiesFVSTRCTRD
      \section synchronizedMGOutput
      <code>MInt FvStructuredSolver::m_synchronizedMGOutput </code>\n
      default = <code> 0 </code>\n \n
      Trigger to write out the moving grid synchronized with the moving grid, i.e.,\n
      the solution is written out in an integer of timesteps to move one cell further.\n
      Possible values are:\n
      <ul>
      <li>0: off</li>
      <li>1: on</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED, IO</i>
    */
    m_synchronizedMGOutput = false;
    m_synchronizedMGOutput =
        Context::getSolverProperty<MBool>("synchronizedMGOutput", m_solverId, AT_, &m_synchronizedMGOutput);

    m_log << "synchronizedMGOutput is activated? " << m_synchronizedMGOutput << endl;

    if(m_gridMovingMethod == 9 || m_gridMovingMethod == 10 || m_gridMovingMethod == 11 || m_gridMovingMethod == 13) {
      m_travelingWave = true;
      m_constantTimeStep = true;
    }

    if(m_gridMovingMethod == 12 || m_gridMovingMethod == 15) {
      m_streamwiseTravelingWave = true;
      m_constantTimeStep = true;
    }
  }
}

/**
 * \brief Reads and initializes properties associated with the Moving Grid Methods
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::setBodyForceProperties() {
  TRACE();

  /*! \property
    \page propertiesFVSTRCTRD
    \section bodyForce
    <code>MInt FvStructuredSolver::m_bodyForce </code>\n
    default = <code> 0 </code>\n \n
    Trigger the use of a specific time-dependent body forcing\n
    Possible values are:\n
    <ul>
    <li>0: off</li>
    <li>1: on</li>
    </ul>
    Keywords: <i>MOVING, STRUCTURED</i>
  */
  m_bodyForce = false;
  if(Context::propertyExists("bodyForce", m_solverId)) {
    m_bodyForce = Context::getSolverProperty<MBool>("bodyForce", m_solverId, AT_, &m_bodyForce);
  }

  if(m_bodyForce) {
    m_movingGridTimeOffset = F0;
    m_movingGridStepOffset = 0;
    m_movingGridInitialStart = true;
    m_bodyForceMethod = 0;
    m_movingGridTimeOffset = F0;

    m_travelingWave = false;
    m_waveTimeStepComputed = false;
    m_waveSpeed = 0.0;
    m_waveBeginTransition = 0.0;
    m_waveEndTransition = 0.0;
    m_waveRestartFadeIn = false;
    m_waveLength = 0.0;
    m_waveAmplitude = 0.0;
    m_waveTime = 0.0;
    m_waveCellsPerWaveLength = -1;
    m_waveNoStepsPerCell = -1;
    m_synchronizedMGOutput = 0;

    m_mgExchangeCoordinates = true;
    if(Context::propertyExists("mgExchangeCoordinates", m_solverId)) {
      m_mgExchangeCoordinates =
          Context::getSolverProperty<MBool>("mgExchangeCoordinates", m_solverId, AT_, &m_mgExchangeCoordinates);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section bodyForceMethod
      <code>MInt FvStructuredSolver::m_bodyForceMethod </code>\n
      default = <code> 0 </code>\n \n
      Number of the moving grid function\n
      specified in the switch in fvstructuredsolver3d\n
      Possible values are:\n
      <ul>
      <li>Integer >= 0</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED</i>
    */
    m_bodyForceMethod = Context::getSolverProperty<MInt>("bodyForceMethod", m_solverId, AT_);

    /*! \property
      \page propertiesFVSTRCTRD
      \section wavePenetrationHeight
      <code>MInt FvStructuredSolver::m_wavePenetrationHeight </code>\n
      default = <code> 0 </code>\n \n
      Value of the wall velocity for\n
      certain moving grid methods.\n
      Possible values are:\n
      <ul>
      <li>Floating point</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED</i>
    */
    if(Context::propertyExists("wavePenetrationHeight", m_solverId)) {
      m_wavePenetrationHeight = Context::getSolverProperty<MFloat>("wavePenetrationHeight", m_solverId, AT_);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section movingGridTimeOffset
      <code>MInt FvStructuredSolver::m_movingGridTimeOffset </code>\n
      default = <code> 0 </code>\n \n
      Numerical value of the time offset\n
      for moving grid functions.\n
      Possible values are:\n
      <ul>
      <li>Floating point</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED</i>
    */
    if(Context::propertyExists("movingGridTimeOffset", m_solverId)) {
      m_movingGridTimeOffset =
          Context::getSolverProperty<MFloat>("movingGridTimeOffset", m_solverId, AT_, &m_movingGridTimeOffset);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section synchronizedMGOutput
      <code>MInt FvStructuredSolver::m_synchronizedMGOutput </code>\n
      default = <code> 0 </code>\n \n
      Trigger to write out the moving grid synchronized with the moving grid, i.e.,\n
      the solution is written out in an integer of timesteps to move one cell further.\n
      Possible values are:\n
      <ul>
      <li>0: off</li>
      <li>1: on</li>
      </ul>
      Keywords: <i>MOVING, STRUCTURED, IO</i>
    */
    m_synchronizedMGOutput = false;
    m_synchronizedMGOutput =
        Context::getSolverProperty<MBool>("synchronizedMGOutput", m_solverId, AT_, &m_synchronizedMGOutput);

    m_log << "synchronizedMGOutput is activated? " << m_synchronizedMGOutput << endl;


    if(m_bodyForceMethod == 10 || m_bodyForceMethod == 11) {
      m_travelingWave = true;
      m_constantTimeStep = true;
    }
  }
}

/**
 * \brief Reads and initializes properties associated with the numerical method.
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::setNumericalProperties() {
  TRACE();


  /*! \property
    \page propertiesFVSTRCTRD
    \section constantTimeStep
    <code>MInt FvStructuredSolver::m_constantTimeStep </code>\n
    default = <code> 1 </code>\n \n
    Trigger the use of a constant time step\n
    (only computed once at startup)\n
    Possible values are:\n
    <ul>
    <li>0: off</li>
    <li>1: on</li>
    </ul>
    Keywords: <i>TIMESTEP, STRUCTURED</i>
  */
  m_constantTimeStep = true; // default is set to true
  m_constantTimeStep = Context::getSolverProperty<MBool>("constantTimeStep", m_solverId, AT_, &m_constantTimeStep);

  /*! \property
    \page propertiesFVSTRCTRD
    \section localTimeStep
    <code>MInt FvStructuredSolver::m_localTimeStep </code>\n
    default = <code> 0 </code>\n \n
    Trigger the use of local time-stepping.
    Possible values are:\n
    <ul>
    <li>0: off</li>
    <li>1: on</li>
    </ul>
    Keywords: <i>TIMESTEP, STRUCTURED</i>
  */
  m_localTimeStep = false;
  m_localTimeStep = Context::getSolverProperty<MBool>("localTimeStep", m_solverId, AT_, &m_localTimeStep);

  /*! \property
    \page propertiesFVSTRCTRD
    \section timeStepComputationInterval
    <code>MInt FvStructuredSolver::m_timeStepComputationInterval </code>\n
    default = <code> 1 </code>\n \n
    Set the interval for the recomputation of\n
    the time step.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 1</li>
    </ul>
    Keywords: <i>TIMESTEP, STRUCTURED</i>
  */
  m_timeStepComputationInterval = 1; // time step computation interval
  if(!m_constantTimeStep) {
    m_timeStepComputationInterval = Context::getSolverProperty<MInt>("timeStepComputationInterval", m_solverId, AT_,
                                                                     &m_timeStepComputationInterval);
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section noGhostLayers
    <code>MInt FvStructuredSolver::m_noGhostLayers </code>\n
    default = <code> 2 </code>\n \n
    Number of ghost-layers around the active mesh.\n
    the time step.\n
    Possible values are:\n
    <ul>
    <li>Integer >= 1</li>
    </ul>
    Keywords: <i>GRID, STRUCTURED</i>
  */
  m_noGhostLayers = Context::getSolverProperty<MInt>("noGhostLayers", m_solverId, AT_);


  /*! \property
    \page propertiesFVSTRCTRD
    \section cfl
    <code>MInt FvStructuredSolver::m_cfl </code>\n
    default = <code>no default</code>\n \n
    Courant number C - Factor of the CFL condition \n \n
    possible values are:
    <ul>
    <li> positive floating point values < stability limit of the time-stepping method </li>
    <li> For default RK5 scheme the theoretical stability limit is C<4. Due to cut and small cells C<1.5 or even C
    <= 1.0 is recommended. </li>
    </ul>
    Keywords: <i> FINITE_VOLUME, STABILITY, TIME_INTEGRATION </i>
  */
  m_cfl = Context::getSolverProperty<MFloat>("cfl", m_solverId, AT_);

  /*! \property
    \page propertiesFVSTRCTRD
    \section limiter
    <code>MInt FvStructuredSolver::m_limiter </code>\n
    default = <code>0</code>\n \n
    Trigger the use of the limiter.\n
    possible values are:
    <ul>
    <li> 0 </li>
    </ul>
    Keywords: <i> FINITE_VOLUME, STABILITY, LIMITER, STRUCTURED </i>
  */
  m_limiter = false;
  m_limiter = Context::getSolverProperty<MBool>("limiter", m_solverId, AT_, &m_limiter);

  if(m_limiter) {
    /*! \property
      \page propertiesFVSTRCTRD
      \section limiterMethod
      <code>MInt FvStructuredSolver::m_limiterMethod </code>\n
      default = <code>ALBADA</code>\n \n
      Name of the limiter to use.\n
      possible values are:
      <ul>
      <li> ALBADA, VENKATAKRISHNAN, MINMOD</li>
      </ul>
      Keywords: <i> FINITE_VOLUME, STABILITY, LIMITER, STRUCTURED </i>
    */
    m_limiterMethod = "ALBADA";
    m_limiterMethod = Context::getSolverProperty<MString>("limiterMethod", m_solverId, AT_);
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section limiterVisc
    <code>MInt FvStructuredSolver::m_limiterVisc </code>\n
    default = <code>0</code>\n \n
    Trigger the use of the limiter.\n
    possible values are:
    <ul>
    <li> 0 </li>
    </ul>
    Keywords: <i> FINITE_VOLUME, STABILITY, LIMITER, STRUCTURED </i>
  */
  // NOTE: with enabled viscous limiter the simulation is no longer conservative
  m_limiterVisc = false;
  m_limiterVisc = Context::getSolverProperty<MBool>("limiterVisc", m_solverId, AT_, &m_limiterVisc);

  if(m_limiterVisc) {
    ASSERT(nDim == 2, "Only available in 2D by now!");
    m_CFLVISC = Context::getSolverProperty<MFloat>("cfl_visc", m_solverId, AT_); // Maybe it is a stupid name
    FQ->neededFQVariables[FQ->LIMITERVISC] = 1;
    FQ->outputFQVariables[FQ->LIMITERVISC] = false;
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section musclScheme
    <code>MInt FvStructuredSolver::m_musclScheme </code>\n
    default = <code>Standard</code>\n \n
    Sets the MUSCL scheme for the structured solver.
    possible values are:
    <ul>
    <li> Other valid MUSCL schemes </li>
    </ul>
    Keywords: <i> STRUCTURED, FV, MUSCL</i>
  */
  /*! \property
    \page propertiesFVSTRCTRD
    \section musclScheme
    <code>MInt FvStructuredSolver::m_musclScheme </code>\n
    default = <code>Standard</code>\n \n
    Name of the MUSCL scheme to be used\n
    possible values are:
    <ul>
    <li> Standard, Stretched</li>
    </ul>
    Keywords: <i> FINITE_VOLUME, STABILITY, LIMITER, STRUCTURED </i>
  */
  m_musclScheme = "Standard";
  m_musclScheme = Context::getSolverProperty<MString>("musclScheme", m_solverId, AT_, &m_musclScheme);

  /*! \property
    \page propertiesFVSTRCTRD
    \section ausmScheme
    <code>MInt FvStructuredSolver::m_ausmScheme </code>\n
    default = <code>Standard</code>\n \n
    Sets the AUSM scheme for the structured solver.
    possible values are:
    <ul>
    <li> Other valid AUSM schemes </li>
    </ul>
    Keywords: <i> STRUCTURED, FV, MUSCL</i>
  */
  /*! \property
    \page propertiesFVSTRCTRD
    \section ausmScheme
    <code>MInt FvStructuredSolver::m_ausmScheme </code>\n
    default = <code>Standard</code>\n \n
    Name of the AUSM scheme to be used.\n
    possible values are:
    <ul>
    <li>Standard, PTHRC, AUSMDV</li>
    </ul>
    Keywords: <i> FINITE_VOLUME, STABILITY, LIMITER, STRUCTURED </i>
  */
  m_ausmScheme = "Standard";
  m_ausmScheme = Context::getSolverProperty<MString>("ausmScheme", m_solverId, AT_, &m_ausmScheme);

  /*! \property
    \page propertiesFVSTRCTRD
    \section convergenceCriterion
    <code>MInt FvStructuredSolver::m_convergenceCriterion </code>\n
    default = <code>Standard</code>\n \n
    Set the convergence criterion to stop computation.\n
    possible values are:
    <ul>
    <li>Float < 1.0</li>
    </ul>
    Keywords: <i> FINITE_VOLUME, CONVERGENCE, STRUCTURED </i>
  */
  m_convergenceCriterion = 1e-12;
  m_convergenceCriterion =
      Context::getSolverProperty<MFloat>("convergenceCriterion", m_solverId, AT_, &m_convergenceCriterion);

  /*! \property
    \page propertiesFVSTRCTRD
    \section upwindCoefficient
    <code>MInt FvStructuredSolver::m_chi</code>\n
    default = <code>0.0</code>\n \n
    Chi for AUSM pressure splitting.\n
    possible values are:
    <ul>
    <li>Float >= 0.0</li>
    </ul>
    Keywords: <i> FINITE_VOLUME, CONVERGENCE, STRUCTURED </i>
  */
  m_chi = 0.0;
  m_chi = Context::getSolverProperty<MFloat>("upwindCoefficient", m_solverId, AT_);

  /*! \property
    \page propertiesFVSTRCTRD
    \section viscousFlux
    <code>MInt FvStructuredSolver::m_viscCompact</code>\n
    default = <code>false</code>\n \n
    Makes your life easy.\n
    possible values are:
    <ul>
    <li>Bool true/false</li>
    </ul>
    Keywords: <i> FINITE_VOLUME, CONVERGENCE, STRUCTURED </i>
  */
  m_viscCompact = false;
  m_viscCompact = Context::getSolverProperty<MBool>("viscCompact", m_solverId, AT_, &m_viscCompact);
}


/**
 * \brief Set properties for porous blocks
 * \date 2020-03-19
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::setPorousProperties() {
  TRACE();

  /*! \property
    \page propertiesFVSTRCTRD
    \section porous
    <code>MFloat FvStructuredSolver::m_porous </code>\n
    default = <code>0</code>\n \n
    Trigger a porous computation.\n
    possible values are:
    <ul>
    <li>true/false</li>
    </ul>
    Keywords: <i>POROUS, STRUCTURED</i>
  */
  m_porous = false;
  if(Context::propertyExists("porous", m_solverId)) {
    m_porous = Context::getSolverProperty<MBool>("porous", m_solverId, AT_);
  }

  if(!m_porous) return;

  m_blockType = "fluid";
  /*! \property
    \page propertiesFVSTRCTRD
    \section porousSolvers
    default = <code>0</code>\n \n
    IDs of the porous solvers.\n
    possible values are:
    <ul>
    <li>Integer >= 0</li>
    </ul>
    Keywords: <i>RANS, ZONAL, STRUCTURED</i>
  */
  const MInt noPorousSolvers = Context::propertyLength("porousSolvers", m_solverId);
  m_porousBlockIds.resize(/*m_noBlocks*/ m_grid->getNoBlocks(), -1);
  //  m_porousID = -1; //in property file different porous solvers can be assigned different props
  const MInt inputBoxID = m_grid->getMyBlockId();
  for(MInt i = 0; i < noPorousSolvers; ++i) {
    const MInt porousSolver = Context::getSolverProperty<MInt>("porousSolvers", m_solverId, AT_, i);
    m_porousBlockIds[porousSolver] = i;
    if(inputBoxID == porousSolver) m_blockType = "porous";

    //    m_porousSolvers[i] = Context::getSolverProperty<MInt>("porousSolvers", m_solverId, AT_,
    //                                                           &m_porousSolvers[i], i);
    //    if (inputBoxID == m_porousSolvers[i]) {
    //      m_blockType = "porous";
    //      m_porousID = i;
    //    }
  }

  // TODO_SS labels:FV later only allocate FQ->POROSITY if m_blockType=="porous"
  FQ->neededFQVariables[FQ->POROSITY] = 1;
  FQ->outputFQVariables[FQ->POROSITY] = false;

  if(m_blockType == "porous") {
    m_log << "Domain " << domainId()
          << " belongs to a " + m_blockType + " solver. PorousId=" << m_porousBlockIds[inputBoxID]
          << endl; // m_porousID << endl;
    //    FQ->neededFQVariables[FQ->DARCY] = 1;
    //    FQ->outputFQVariables[FQ->DARCY] = false;
    //    FQ->neededFQVariables[FQ->FORCH] = 1;
    //    FQ->outputFQVariables[FQ->FORCH] = false;
  }
  // Actually the following variables are only required if m_blockType=="porous"
  FQ->neededFQVariables[FQ->DARCY] = 1;
  FQ->outputFQVariables[FQ->DARCY] = false;
  FQ->neededFQVariables[FQ->FORCH] = 1;
  FQ->outputFQVariables[FQ->FORCH] = false;
  for(MInt d = 0; d < nDim; ++d) {
    FQ->neededFQVariables[FQ->NORMAL[d]] = 1;
    FQ->outputFQVariables[FQ->NORMAL[d]] = false;
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::initPorous() {
  TRACE();

  // Assign defaults
  m_c_Dp = 0.2;
  m_c_Dp_eps = 0.2;
  m_c_wd = 5.0;
  m_c_t = 0.0;
  m_c_eps = 0.0;

  // Read in from property file
  m_c_Dp = Context::getSolverProperty<MFloat>("c_Dp", m_solverId, AT_, &m_c_Dp);
  m_c_Dp_eps = Context::getSolverProperty<MFloat>("c_Dp_eps", m_solverId, AT_, &m_c_Dp_eps);
  m_c_wd = Context::getSolverProperty<MFloat>("c_wd", m_solverId, AT_, &m_c_wd);
  m_c_t = Context::getSolverProperty<MFloat>("c_t", m_solverId, AT_, &m_c_t);
  m_c_eps = Context::getSolverProperty<MFloat>("c_eps", m_solverId, AT_, &m_c_eps);

  // Fill porosity, Da, Forch (By now each porous solver is homogeneous)
  if(m_blockType == "porous") {
    const MInt inputBoxID = m_grid->getMyBlockId();
    const MFloat por = Context::getSolverProperty<MFloat>("porosity", m_solverId, AT_,
                                                          m_porousBlockIds[inputBoxID]); // m_porousID);
    const MFloat Da =
        Context::getSolverProperty<MFloat>("Da", m_solverId, AT_, m_porousBlockIds[inputBoxID]); // m_porousID);
    const MFloat cf =
        Context::getSolverProperty<MFloat>("cf", m_solverId, AT_, m_porousBlockIds[inputBoxID]); // m_porousID);
    for(MInt cellId = 0; cellId < m_noCells; ++cellId) {
      m_cells->fq[FQ->POROSITY][cellId] = por;
      m_cells->fq[FQ->DARCY][cellId] = Da;
      m_cells->fq[FQ->FORCH][cellId] = cf;
    }
  } else {
    // Fluid solver has only porosity
    for(MInt cellId = 0; cellId < m_noCells; ++cellId) {
      m_cells->fq[FQ->POROSITY][cellId] = 1.0;
      m_cells->fq[FQ->DARCY][cellId] = std::numeric_limits<MFloat>::max();
    }
  }

  // TODO_SS labels:FV,toenhance Move stuff from initBc6002 into here
}


template <MInt nDim>
MInt FvStructuredSolver<nDim>::timer(const MInt timerId) const {
  TRACE();

  // Abort if timers not initialized
  if(!m_isInitTimers) {
    TERMM(1, "Timers were not initialized.");
  }

  return m_timers[timerId];
}

template <MInt nDim>
void FvStructuredSolver<nDim>::initTimers() {
  TRACE();
  m_timers.fill(-1);

  NEW_TIMER_GROUP_NOCREATE(m_timerGroup, "Structured Solver");
  NEW_TIMER_NOCREATE(m_timers[Timers::Structured], "total object lifetime", m_timerGroup);
  RECORD_TIMER_START(m_timers[Timers::Structured]);

  // Create & start constructor timer
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Constructor], "Constructor", m_timers[Timers::Structured]);
  RECORD_TIMER_START(m_timers[Timers::Constructor]);

  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::GridDecomposition], "Grid Decomposition", m_timers[Timers::Constructor]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::GridReading], "Grid reading", m_timers[Timers::Constructor]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::BuildUpSponge], "Build up sponge", m_timers[Timers::Constructor]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::ComputeMetrics], "Compute Metrics", m_timers[Timers::Constructor]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::ComputeJacobian], "Compute Jacobian", m_timers[Timers::Constructor]);

  // Create regular solver-wide timers
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::RunInit], "Init", m_timers[Timers::Structured]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::LoadRestart], "Load restart", m_timers[Timers::RunInit]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::LoadVariables], "load restart variables", m_timers[Timers::LoadRestart]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::LoadSponge], "load sponge", m_timers[Timers::LoadRestart]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::LoadSTG], "load STG", m_timers[Timers::LoadRestart]);


  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Run], "Run function", m_timers[Timers::Structured]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MainLoop], "Main loop", m_timers[Timers::Run]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::ConvectiveFlux], "Convective Flux", m_timers[Timers::MainLoop]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::ViscousFlux], "Viscous Flux", m_timers[Timers::MainLoop]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGVolumeFlux], "Volume Flux", m_timers[Timers::MainLoop]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SandpaperTrip], "Sandpaper tripping", m_timers[Timers::MainLoop]);

  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MovingGrid], "Moving Grid", m_timers[Timers::MainLoop]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGVolumeFlux], "MG Volume Flux", m_timers[Timers::MovingGrid]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGMoveGrid], "MG Move Grid", m_timers[Timers::MovingGrid]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGExchange], "MG Exchange", m_timers[Timers::MGMoveGrid]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGCellCenterCoordinates], "MG Cell Center Coordinates",
                         m_timers[Timers::MGMoveGrid]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGMetrics], "MG Metrics", m_timers[Timers::MGMoveGrid]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGSurfaceMetrics], "MG Surface Metrics", m_timers[Timers::MGMetrics]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGCellMetrics], "MG Cell Metrics", m_timers[Timers::MGMetrics]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGCornerMetrics], "MG Corner Metrics", m_timers[Timers::MGMetrics]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGJacobian], "MG Jacobian", m_timers[Timers::MGMoveGrid]);

  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MGSaveGrid], "MG Save Grid", m_timers[Timers::MovingGrid]);

  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Exchange], "Exchange", m_timers[Timers::MainLoop]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Gather], "Gather", m_timers[Timers::Exchange]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Send], "Send", m_timers[Timers::Exchange]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SendWait], "SendWait", m_timers[Timers::Exchange]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Receive], "Receive", m_timers[Timers::Exchange]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::ReceiveWait], "ReceiveWait", m_timers[Timers::Exchange]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Scatter], "Scatter", m_timers[Timers::Exchange]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Receive], "Receive", m_timers[Timers::Exchange]);

  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::BoundaryCondition], "Boundary Conditions", m_timers[Timers::MainLoop]);

  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::MLCoupling], "MLCoupling", m_timers[Timers::MainLoop]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Setup], "Setup Coupling", m_timers[Timers::MLCoupling]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Coupling], "Coupling without Inference", m_timers[Timers::MLCoupling]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::Inference], "Inference Coupling", m_timers[Timers::MLCoupling]);

  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::RungeKutta], "RungeKutta", m_timers[Timers::MainLoop]);

  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SaveOutput], "Save output", m_timers[Timers::Run]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SaveSolution], "Save solution", m_timers[Timers::SaveOutput]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SaveForces], "Save forces", m_timers[Timers::SaveOutput]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SaveAuxdata], "Save auxdata", m_timers[Timers::SaveOutput]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SaveBoxes], "Save boxes", m_timers[Timers::SaveOutput]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SaveIntpPoints], "Save lines", m_timers[Timers::SaveOutput]);


  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::SetTimeStep], "Set Time Step", m_timers[Timers::MainLoop]);
  NEW_SUB_TIMER_NOCREATE(m_timers[Timers::UpdateSponge], "Update sponge", m_timers[Timers::MainLoop]);

  // Set status to initialized
  m_isInitTimers = true;
}

template <MInt nDim>
void FvStructuredSolver<nDim>::createMPIGroups() {
  // Channel Communication
  mAlloc(m_channelRoots, 4, "m_channelRoots", -1, AT_);

  // Periodic rotation boundary condtions communicators
  mAlloc(m_commPerRotRoots, 4, "m_commPerRotRoots", -1, AT_);

  ////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////MPI Zonal////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  if(m_zonal) {
    mAlloc(m_commZonal, m_noBlocks, "m_commZonal", AT_);
    mAlloc(m_commZonalRoot, m_noBlocks, "m_commZonalRoot", -1, AT_);
    mAlloc(m_commZonalRootGlobal, m_noBlocks, "m_commZonalRoot", -1, AT_);

    m_commZonalMyRank = -1;
    m_zonalRootRank = false;


    // first collect the ranks of each input solver in one array and count the number
    for(MInt i = 0; i < m_noBlocks; i++) {
      MPI_Group groupZonal, newGroupZonal;
      vector<MInt> tmpPartitions;
      MInt blockDomainId = 0;
      MInt hasBlockDomain = 0;
      MInt hasBlockDomainGlobal = 0;
      MIntScratchSpace nblockDomainArray(noDomains(), AT_, "nblockDomainArray");
      MIntScratchSpace nblockDomainOffset(noDomains(), AT_, "nblockDomainOffset");
      if(m_blockId == i) {
        blockDomainId = domainId();
        hasBlockDomain = 1;
      }
      MPI_Allreduce(&hasBlockDomain, &hasBlockDomainGlobal, 1, MPI_INT, MPI_SUM, m_StructuredComm, AT_,
                    "hasBlockDomain", "hasBlockDomainGlobal");
      MPI_Allgather(&hasBlockDomain, 1, MPI_INT, &nblockDomainArray[0], 1, MPI_INT, m_StructuredComm, AT_,
                    "hasBlockDomain", "nblockDomainArray[0]");
      nblockDomainOffset[0] = 0;
      for(MInt j = 1; j < noDomains(); j++) {
        nblockDomainOffset[j] = nblockDomainOffset[j - 1] + nblockDomainArray[j - 1];
      }
      MIntScratchSpace zonalRanks(hasBlockDomainGlobal, AT_, "zonalRanks");
      MPI_Allgatherv(&blockDomainId, nblockDomainArray[domainId()], MPI_INT, &zonalRanks[0], &nblockDomainArray[0],
                     &nblockDomainOffset[0], MPI_INT, m_StructuredComm, AT_, "blockDomainId", "zonalRanks[0]");


      MInt zonalcommsize = hasBlockDomainGlobal;

      MPI_Comm_group(m_StructuredComm, &groupZonal, AT_, "groupZonal");
      MPI_Group_incl(groupZonal, zonalcommsize, &zonalRanks[0], &newGroupZonal, AT_);
      MPI_Comm_create(m_StructuredComm, newGroupZonal, &m_commZonal[i], AT_, "m_commZonal[i]");

      if(domainId() == zonalRanks[0]) {
        MPI_Comm_rank(m_commZonal[i], &m_commZonalRoot[i]);
        MPI_Comm_rank(m_StructuredComm, &m_commZonalRootGlobal[i]);
        m_zonalRootRank = true;
      }

      MPI_Bcast(&m_commZonalRoot[0], m_noBlocks, MPI_INT, zonalRanks[0], m_StructuredComm, AT_, "m_commZonalRoot[0]");
      MPI_Bcast(&m_commZonalRootGlobal[0], m_noBlocks, MPI_INT, zonalRanks[0], m_StructuredComm, AT_,
                "m_commZonalRootGlobal[0]");
    }
  }
}

/**
 * \brief
 *
 * \date 2020-03-19
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::readAndSetAuxDataMap() {
  TRACE();

  /*! \property
    \page propertiesFVSTRCTRD
    \section auxDataType
    <code>MInt auxDataType </code>\n
    default = <code> 0</code>\n \n
    Specify the definition of "walls", i.e.,\n
    if only solid walls, fluid-porous interfaces,\n
    both or specific windows, should be considered\n
    for cp, cf etc. computations.\n
    Possible values are:\n
    <ul>
    <li>{0,1,2}</li>
    </ul>
    Keywords: <i>AUXDATA, IO, STRUCTURED</i>
  */
  // 0->only solids (default)
  // 1->fluid-porous interfaces and solids
  // 2->only fluid-porous interfaces
  // 3->specific windows only
  MInt auxDataType = 0;
  auxDataType = Context::getSolverProperty<MInt>("auxDataType", m_solverId, AT_, &auxDataType);

  std::vector<MInt> auxDataWindowIds;
  if(auxDataType == 3) {
    if(!Context::propertyExists("auxDataWindowIds", m_solverId))
      mTerm(1, "auxDataType==3 requires the property 'auxDataWindowIds'!");
    const MInt noAuxDataWindowIds = Context::propertyLength("auxDataWindowIds", m_solverId);
    auxDataWindowIds.resize(noAuxDataWindowIds);
    for(MInt i = 0; i < noAuxDataWindowIds; ++i) {
      auxDataWindowIds[i] =
          Context::getSolverProperty<MInt>("auxDataWindowIds", m_solverId, AT_, &auxDataWindowIds[i], i);
    }
  }

  // If force==true: solids and fluid-porous interfaces will be added to m_physicalAuxDataMap regardless of auxDataType;
  //                 but m_auxDataWindowIds will contain only those as specified by auxDataType; later all kinds of
  //                 outputs of the auxData will only include those maps which are in m_auxDataWindowIds, so those as
  //                 specified by auxDataType
  MBool force = false;
  if(m_rans && m_ransMethod == RANS_KEPSILON) force = true;

  m_windowInfo->createAuxDataMap(auxDataType, m_blockType, m_porousBlockIds, auxDataWindowIds, force);
}


template <MInt nDim>
void FvStructuredSolver<nDim>::readAndSetSpongeLayerProperties() {
  TRACE();

  // initialize the values for the Sponge!
  m_spongeLayerThickness = nullptr;
  m_betaSponge = nullptr;
  m_sigmaSponge = nullptr;
  m_noSpongeDomainInfos = 0; // number of bc/windows for bc
  m_spongeLayerType = 1;
  m_targetDensityFactor = F0;
  m_computeSpongeFactor = true;
  MBool readSpongeFromBc = true;

  /*! \property
    \page propertiesFVSTRCTRD
    \section useSponge
    <code>MInt FvStructuredSolver::m_useSponge </code>\n
    default = <code> 0</code>\n \n
    Trigger to use the sponge.\n
    Possible values are:\n
    <ul>
    <li>true/false</li>
    </ul>
    Keywords: <i>SPONGE, IO, STRUCTURED</i>
  */
  m_useSponge = false;
  m_useSponge = Context::getSolverProperty<MBool>("useSponge", m_solverId, AT_, &m_useSponge);

  if(m_useSponge) { // yes
    FQ->neededFQVariables[FQ->SPONGE_FACTOR] = 1;

    /*! \property
      \page propertiesFVSTRCTRD
      \section readSpongeFromBC
      <code>MBool readSpongeFromBC </code>\n
      default = <code> 0</code>\n \n
      Use the given BC numbers to apply sponge\n
      to each window connected with this BC number,\n
      otherwise use given windows IDs.\n
      Possible values are:\n
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>SPONGE, IO, STRUCTURED</i>
    */
    readSpongeFromBc = Context::getSolverProperty<MBool>("readSpongeFromBC", m_solverId, AT_, &readSpongeFromBc);

    MInt noSpongeIds;
    if(readSpongeFromBc) {
      /*! \property
  \page propertiesFVSTRCTRD
        \section spongeBndryCndIds
        <code>MInt noSpongeIds </code>\n
        default = <code> 0</code>\n \n
        Use the given BC numbers to apply sponge\n
        to each window connected with this BC number.\n
        Possible values are:\n
        <ul>
        <li>BC Id</li>
        </ul>
        Keywords: <i>SPONGE, IO, STRUCTURED</i>
      */
      noSpongeIds = Context::propertyLength("spongeBndryCndIds", m_solverId);
    } else {
      /*! \property
  \page propertiesFVSTRCTRD
        \section spongeWindowIds
        <code>MInt noSpongeIds </code>\n
        default = <code> 0</code>\n \n
        Use the given window IDs to apply\n
        a sponge to them.\n
        <ul>
        <li>Window ID</li>
        </ul>
        Keywords: <i>SPONGE, IO, STRUCTURED</i>
      */
      noSpongeIds = Context::propertyLength("spongeWindowIds", m_solverId);
    }
    m_noSpongeDomainInfos = noSpongeIds;

    /*! \property
      \page propertiesFVSTRCTRD
      \section spongeLayerType
      <code>MInt FvStructuredSolver::m_spongeLayerType </code>\n
      default = <code> 0</code>\n \n
      Type of the sponge layer, i.e., \n
      sponge to pressure, density, both,\n
      infinity values or predefined field etc.\n
      <ul>
      <li>Integer of Sponge type</li>
      </ul>
      Keywords: <i>SPONGE, IO, STRUCTURED</i>
    */
    m_spongeLayerType = Context::getSolverProperty<MInt>("spongeLayerType", m_solverId, AT_, &m_spongeLayerType);


    MInt noSpongeBeta = Context::propertyLength("betaSponge", m_solverId);
    MInt noSpongeSigma = Context::propertyLength("sigmaSponge", m_solverId);

    // The programm aborts if number of sponge properties does not fit together
    if(m_noSpongeDomainInfos != noSpongeBeta || m_noSpongeDomainInfos != noSpongeSigma) {
      mTerm(1, AT_, "The number of sponge properties does not match");
    }

    // else we can allocate the memory necessary for the sponge etc'
    mAlloc(m_spongeLayerThickness, m_noSpongeDomainInfos, "m_spongeLayerThicknesses", F0, AT_);
    mAlloc(m_sigmaSponge, m_noSpongeDomainInfos, "m_sigmaSponge", AT_);
    mAlloc(m_betaSponge, m_noSpongeDomainInfos, "m_betaSponge", AT_);
    mAlloc(m_spongeBcWindowInfo, m_noSpongeDomainInfos, "m_spongeBcWindowInfo", AT_);

    // read all the parameters
    for(MInt i = 0; i < m_noSpongeDomainInfos; ++i) {
      /*! \property
  \page propertiesFVSTRCTRD
        \section spongeLayerThickness
        <code>MFloat FvStructuredSolver::m_spongeLayerThickness </code>\n
        default = <code>-1.0</code>\n \n
        The property controls the thickness of the sponge layer in which the sponge layer forcing is applied. The sponge
        forcing term added to the rhs of a cell inside the sponge layer is given by \n \f$ \Delta L(\phi) = V \sigma
        \frac{\Delta x_{sp}^2}{L_s^2} \Delta \phi  \f$,  \n where \f$ V \f$ is the cell volume, \f$  \sigma  \f$ is the
        forcing amplitude, \f$  x_{sp} \f$ is the inner sponge layer boundary, \f$  L_{sp} \f$ is the sponge layer
        thickness and \f$  \Delta \phi = \phi - \phi_{target}  \f$ is the difference between the local and the freesteam
        values of \f$  \phi  \f$.\n possible values are: <ul> <li> Any non-negative floating point value. </li>
        </ul>
        Keywords: <i> STRUCTURED, SPONGE </i>
      */
      m_spongeLayerThickness[i] = -1.0;
      m_spongeLayerThickness[i] =
          Context::getSolverProperty<MFloat>("spongeLayerThickness", m_solverId, AT_, &m_spongeLayerThickness[i], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section betaSponge
        <code>MFloat StructuredSolver::m_betaSponge  </code>\n
        default = <code>0</code>\n \n
        The property controls the sponge function. Linear sponge spongeBeta = 1, quadratic spongeBeta = 2, etc. \n
        possible values are:
        <ul>
        <li>floating point values.</li>
        </ul>
        Keywords: <i>STRUCTURED, SPONGE, BETA, PROFIL</i>
      */
      m_betaSponge[i] = F0;
      m_betaSponge[i] =
          Context::getSolverProperty<MFloat>("betaSponge", m_solverId, AT_, &m_spongeLayerThickness[i], i);

      /*! \property
  \page propertiesFVSTRCTRD
        \section sigmaSponge
        <code>MFloat StructuredSolver::m_sigmaSponge  </code>\n
        default = <code>0</code>\n \n
        Controls the sigma of the sponge, i.e., the strength of the sponge.\n
        possible values are:
        <ul>
        <li>floating point values.</li>
        </ul>
        Keywords: <i>STRUCTURED, SPONGE, BETA, PROFIL</i>
      */
      m_sigmaSponge[i] = F0;
      m_sigmaSponge[i] =
          Context::getSolverProperty<MFloat>("sigmaSponge", m_solverId, AT_, &m_spongeLayerThickness[i], i);
    }


    if(readSpongeFromBc) {
      for(MInt i = 0; i < m_noSpongeDomainInfos; ++i) {
        m_spongeBcWindowInfo[i] = -1;
        m_spongeBcWindowInfo[i] =
            Context::getSolverProperty<MInt>("spongeBndryCndIds", m_solverId, AT_, &m_spongeBcWindowInfo[i], i);
      }
    } else {
      for(MInt i = 0; i < m_noSpongeDomainInfos; ++i) {
        m_spongeBcWindowInfo[i] = -1;
        m_spongeBcWindowInfo[i] =
            Context::getSolverProperty<MInt>("spongeWindowIds", m_solverId, AT_, &m_spongeBcWindowInfo[i], i);
      }
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section targetDensityFactor
      <code>MFloat FvStructuredSolver::m_targetDensityFactor </code>\n
      default = <code>1.0</code>\n \n
      The property controls the Intensity of the sponge layer correction regarding the density forcing term for some
      values of the spongeLayerType. The sponge forcing term added to the rhs of a cell inside the sponge layer is given
      by \n \f$ \Delta L(\phi) = V \sigma \frac{\Delta x_{sp}^2}{L_s^2} \Delta \phi  \f$,  \n where \f$ V \f$ is the
      cell volume, \f$  \sigma  \f$ is the forcing amplitude, \f$  x_{sp} \f$ is the inner sponge layer boundary, \f$
      L_{sp} \f$ is the sponge layer thickness and \f$  \Delta \phi = \phi - \phi_{target}  \f$ is the difference
      between the local and the freesteam values of \f$  \phi  \f$.\n The density target value is in these cases given
      as:    \n <code> deltaRho =a_pvariable( cellId ,  PV->RHO ) - m_rhoInfinity * m_targetDensityFactor;</code>\n See
      also spongeLayerType. Only meaningful and required with certain values for \ref spongeLayerType and if both \ref
      spongeLayerThickness and \ref sigmaSponge are specified and nonzero! ! \n \n possible values are: <ul>
      <li>Non-negative floating point values.</li>
      </ul>
      Keywords: <i>STRUCTURED, SPONGE</i>
    */
    m_targetDensityFactor = F1;
    m_targetDensityFactor =
        Context::getSolverProperty<MFloat>("targetDensityFactor", m_solverId, AT_, &m_targetDensityFactor);

    m_windowInfo->setSpongeInformation(m_noSpongeDomainInfos, m_betaSponge, m_sigmaSponge, m_spongeLayerThickness,
                                       m_spongeBcWindowInfo, readSpongeFromBc);

    /*! \property
      \page propertiesFVSTRCTRD
      \section computeSpongeFactor
      <code>MBool FvStructuredSolver::m_computeSpongeFactor  </code>\n
      default = <code>0</code>\n \n
      Trigger the sponge computation, if set to false.\n
      the sponge values will be read from the restart file\n
      which may be much faster due to the slow sponge computation.\n
      possible values are:
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>STRUCTURED, SPONGE</i>
    */
    m_computeSpongeFactor = true;
    if(Context::propertyExists("computeSpongeFactor", m_solverId)) {
      m_computeSpongeFactor =
          Context::getSolverProperty<MBool>("computeSpongeFactor", m_solverId, AT_, &m_computeSpongeFactor);
    }

    // we now have the spongeProperties save in m_windowInfo->m_spongeInfoMap
    // allocating and other stuff can be handled in the boundaryCondition constructor

    if(m_spongeLayerType == 2) {
      // active the two sponge layer FQ fields for rho and rhoE
      FQ->neededFQVariables[FQ->SPONGE_RHO] = 1;
      FQ->neededFQVariables[FQ->SPONGE_RHO_E] = 1;
    }

    if(m_spongeLayerType == 4) {
      // active the two sponge layer FQ fields for rho and rhoE
      FQ->neededFQVariables[FQ->SPONGE_RHO] = 1;
    }
  }
}


/**
 * \brief This function reads the properties required for Runge Kutta time stepping.
 * \author Pascal S. Meysonnat (see FV-> Gonzalo G-B)
 * \date July, 2013
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::setRungeKuttaProperties() {
  TRACE();

  //! Allocate and initialize Runge-Kutta coefficients:
  /*! \property
    \page propertiesFVSTRCTRD
    \section noRKSteps
    <code>MInt FvStructuredSolver::m_noRKSteps </code>\n
    default = <code>5</code>\n \n
    Number of steps in the Runge-Kutta time-stepping method.
    possible values are:
    <ul>
    <li>Positive integers.</li>
    </ul>
    Keywords: <i>FV, RUNGE KUTTA, TIME STEPPING</i>
  */
  m_noRKSteps = 5;
  m_noRKSteps = Context::getSolverProperty<MInt>("noRKSteps", m_solverId, AT_, &m_noRKSteps);

  mDeallocate(m_RKalpha);
  mAlloc(m_RKalpha, m_noRKSteps, "m_RKalpha", -F1, AT_);

  /*! \property
    \page propertiesFVSTRCTRD
    \section rkalpha-step
    <code>MFloat FvStructuredSolver::m_RKalpha[m_noRKSteps] </code>\n
    default = <code>0.25, 0.16666666666, 0.375, 0.5, 1</code> IF noRKSteps is 5.\n \n
    Coeffients of the Runge-Kutta time-stepping method.
    possible values are:
    <ul>
    <li>Floating point numbers (as many as Runge-Kutta steps).</li>
    </ul>
    Keywords: <i>FV, RUNGE KUTTA, TIME STEPPING</i>
  */
  if(m_noRKSteps == 5) { // default only valid m_noRKSteps == 5
    MFloat RK5DefaultCoeffs[5] = {0.25, 0.16666666666, 0.375, 0.5, 1.};
    for(MInt i = 0; i < m_noRKSteps; i++) {
      m_RKalpha[i] =
          Context::getSolverProperty<MFloat>("rkalpha-step", m_solverId, AT_, (MFloat*)&RK5DefaultCoeffs[i], i);
    }
  }

  if(m_noRKSteps == 1) m_RKalpha[0] = 1.0;

  // in case of zonal computations
  // different rk coefficients can be
  // set for the RANS zones
  if(m_rans) {
    if(Context::propertyExists("rkalpha-step-rans", m_solverId)) {
      MFloat RK5DefaultCoeffs[5] = {0.059, 0.14, 0.273, 0.5, 1.0};
      for(MInt i = 0; i < m_noRKSteps; i++) {
        m_RKalpha[i] =
            Context::getSolverProperty<MFloat>("rkalpha-step-rans", m_solverId, AT_, &RK5DefaultCoeffs[i], i);
      }
    }
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section rungeKuttaOrder
    <code>MFloat FvStructuredSolver::m_rungeKuttaOrder </code>\n
    default = <code>2</code>\n \n
    Defines the runge kutta method (order).
    possible values are:
    <ul>
    <li>2 - second order</li>
    <li>3 - third order</li>
    </ul>
    Keywords: <i>TIME_INTEGRATION, RUNGE_KUTTA</i>
  */
  m_rungeKuttaOrder = 2;
  m_rungeKuttaOrder = Context::getSolverProperty<MInt>("rungeKuttaOrder", m_solverId, AT_, &m_rungeKuttaOrder);
}


template <MInt nDim>
void FvStructuredSolver<nDim>::initializeRungeKutta() {
  TRACE();
  setRungeKuttaProperties();

  if(!m_restart) {
    m_time = F0;
    m_physicalTime = F0;
    globalTimeStep = 0;
  } else {
    m_time = -1.0;
    m_physicalTime = -1.0;
  }

  m_RKStep = 0;

  // this is used to initialize the counter the very first time this is called
  // and to initialize the very first residual
  if(approx(m_time, F0, m_eps)) {
    m_workload = 0;
    m_workloadIncrement = 1;
    m_firstMaxResidual = F0;
    m_firstAvrgResidual = F0;
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::setSTGProperties() {
  TRACE();
  m_stgIsActive = false;
  m_stgInitialStartup = false;
  m_stgFace = 0;

  IF_CONSTEXPR(nDim < 3) { return; }

  /*! \property
    \page propertiesFVSTRCTRD
    \section useSTG
    <code>MFloat FvStructuredSolver::m_stgIsActive </code>\n
    default = <code>0</code>\n \n
    Trigger the use of the STG BC.\n
    possible values are:
    <ul>
    <li>true/false</li>
    </ul>
    Keywords: <i>STG, STRUCTURED</i>
  */
  m_stgIsActive = false;
  if(Context::propertyExists("useSTG", m_solverId)) {
    m_stgIsActive = Context::getSolverProperty<MBool>("useSTG", m_solverId, AT_, &m_stgIsActive);
  }

  // switch this on for zonal computation
  if(m_zonal) {
    m_stgIsActive = true;
  }

  if(m_stgIsActive) {
    m_stgLocal = false;
    m_stgRootRank = false;

    for(MInt i = 0; i < abs((MInt)m_windowInfo->physicalBCMap.size()); ++i) {
      if(m_windowInfo->physicalBCMap[i]->BC == 7909) {
        m_stgLocal = true;
        m_stgFace = m_windowInfo->physicalBCMap[i]->face;
        break;
      }
    }

    if(m_stgLocal) {
      MPI_Comm_rank(m_commStg, &m_commStgMyRank);
    } else {
      m_commStgMyRank = -1;
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgSubSup
      <code>MFloat FvStructuredSolver::m_stgSubSup </code>\n
      default = <code>0</code>\n \n
      Use mixed subsonics/subsonic formulation\n
      of the STG boundary.\n
      possible values are:
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgSubSup = false;
    if(Context::propertyExists("stgSubSup", m_solverId)) {
      m_stgSubSup = Context::getSolverProperty<MBool>("stgSubSup", m_solverId, AT_, &m_stgSubSup);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgSupersonic
      <code>MFloat FvStructuredSolver::m_stgSupersonic </code>\n
      default = <code>0</code>\n \n
      Use supersonic STG boundary formulation.\n
      possible values are:
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgSupersonic = false;
    if(Context::propertyExists("stgSupersonic", m_solverId)) {
      m_stgSupersonic = Context::getSolverProperty<MBool>("stgSupersonic", m_solverId, AT_, &m_stgSupersonic);

      if(m_stgSupersonic && m_stgSubSup) {
        m_stgSubSup = false;
        if(domainId() == 0) {
          cout << "WARNING: You activated conflicting properties stgSubSup "
               << "and stgSupersonic, thus only the pure supersonic formulation will be used. "
               << "Switch off stgSupersonic to get the mixed formulation" << endl;
        }
      }
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section BLT1
      <code>MFloat FvStructuredSolver::m_stgBLT1 </code>\n
      default = <code>1.0</code>\n \n
      Defines the size of the STG virtual box\n
      in the x-direction as a fraction of the\n
      delta0 specified.\n
      possible values are:
      <ul>
      <li>Floating point > 0.0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgBLT1 = 1.0;
    m_stgBLT1 = Context::getSolverProperty<MFloat>("BLT1", m_solverId, AT_, &m_stgBLT1);

    /*! \property
      \page propertiesFVSTRCTRD
      \section BLT2
      <code>MFloat FvStructuredSolver::m_stgBLT2 </code>\n
      default = <code>2</code>\n \n
      Defines the size of the STG virtual box\n
      in the y-direction as a fraction of the\n
      delta0 specified.\n
      possible values are:
      <ul>
      <li>Floating point > 0.0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgBLT2 = 1.1;
    m_stgBLT2 = Context::getSolverProperty<MFloat>("BLT2", m_solverId, AT_, &m_stgBLT2);

    /*! \property
      \page propertiesFVSTRCTRD
      \section BLT3
      <code>MFloat FvStructuredSolver::m_stgBLT3 </code>\n
      default = <code>1.1</code>\n \n
      Defines the size of the STG virtual box\n
      in the z-direction as a fraction of the\n
      delta0 specified.\n
      possible values are:
      <ul>
      <li>Floating point > 0.0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgBLT3 = 1.0;
    m_stgBLT3 = Context::getSolverProperty<MFloat>("BLT3", m_solverId, AT_, &m_stgBLT3);

    /*! \property
      \page propertiesFVSTRCTRD
      \section deltaIn
      <code>MFloat FvStructuredSolver::m_stgDelta99Inflow</code>\n
      default = <code>-1.0</code>\n \n
      Defines the delta0 thickness at the inflow for the STG.\n
      possible values are:
      <ul>
      <li>Floating point > 0.0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgDelta99Inflow = -1.0;
    m_stgDelta99Inflow = Context::getSolverProperty<MFloat>("deltaIn", m_solverId, AT_, &m_stgDelta99Inflow);

    m_stgBLT1 = m_stgBLT1 * m_stgDelta99Inflow;
    m_stgBLT2 = m_stgBLT2 * m_stgDelta99Inflow;

    mAlloc(m_stgLengthFactors, 3, "m_solver->m_stgLengthFactors", F0, AT_);
    m_stgLengthFactors[0] = 1.0;
    m_stgLengthFactors[1] = 0.6;
    m_stgLengthFactors[2] = 1.5;

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgLengthFactors
      <code>MFloat FvStructuredSolver::m_stgLengthFactors </code>\n
      default = <code>1.0, 0.6, 1.5</code>\n \n
      The factor to scale the length scales\n
      in each coordinate direction with. For higher\n
      Reynolds number the values [1.0, 0.5, 1.4] \n
      produce better results.\n
      possible values are:
      <ul>
      <li>Float<3> > 0.0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    if(Context::propertyExists("stgLengthFactors", m_solverId)) {
      for(MInt i = 0; i < 3; i++) {
        m_stgLengthFactors[i] =
            Context::getSolverProperty<MFloat>("stgLengthFactors", m_solverId, AT_, &m_stgLengthFactors[i], i);
      }
    }

    mAlloc(m_stgRSTFactors, 3, "m_solver->m_stgRSTFactors", F0, AT_);
    m_stgRSTFactors[0] = 0.7;
    m_stgRSTFactors[1] = 0.4;
    m_stgRSTFactors[2] = 0.5;

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgRSTFactors
      <code>MFloat FvStructuredSolver::m_stgRSTFactors </code>\n
      default = <code>1.0, 0.6, 1.5</code>\n \n
      The factor to scale the length scales\n
      in each coordinate direction with. For higher\n
      Reynolds number the values [1.0, 0.5, 1.4] \n
      produce better results.\n
      possible values are:
      <ul>
      <li>Float<3> > 0.0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    if(Context::propertyExists("stgRSTFactors", m_solverId)) {
      for(MInt i = 0; i < 3; i++) {
        m_stgRSTFactors[i] =
            Context::getSolverProperty<MFloat>("stgRSTFactors", m_solverId, AT_, &m_stgRSTFactors[i], i);
      }
    }


    /*! \property
      \page propertiesFVSTRCTRD
      \section stgMaxNoEddies
      <code>MFloat FvStructuredSolver::m_stgMaxNoEddies </code>\n
      default = <code>200</code>\n \n
      Number of Eddies in the STG virtual box.\n
      possible values are:
      <ul>
      <li>Integer > 0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgMaxNoEddies = 200;
    m_stgMaxNoEddies = Context::getSolverProperty<MInt>("stgMaxNoEddies", m_solverId, AT_, &m_stgMaxNoEddies);

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgExple
      <code>MFloat FvStructuredSolver::m_stgExple </code>\n
      default = <code>0.5</code>\n \n
      Exponent of the STG LengthScale law.\n
      possible values are:
      <ul>
      <li>Floating point > 0.0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgExple = 0.5;
    m_stgExple = Context::getSolverProperty<MFloat>("stgExple", m_solverId, AT_, &m_stgExple);

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgEddieDistribution
      <code>MFloat FvStructuredSolver::m_stgEddieDistribution </code>\n
      default = <code>1.0</code>\n \n
      Shift die eddie distribution more to the wall\n
      or boundary layer edge.\n
      possible values are:
      <ul>
      <li>Floating point > 0.0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgEddieDistribution = 1.0;
    if(Context::propertyExists("stgEddieDistribution", m_solverId)) {
      m_stgEddieDistribution =
          Context::getSolverProperty<MFloat>("stgEddieDistribution", m_solverId, AT_, &m_stgEddieDistribution);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgCreateNewEddies
      <code>MFloat FvStructuredSolver::m_stgCreateNewEddies </code>\n
      default = <code>0</code>\n \n
      Enforces the creation of all new eddies in STG virtual box\n
      or boundary layer edge.\n
      possible values are:
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgCreateNewEddies = false;
    if(Context::propertyExists("stgCreateNewEddies", m_solverId)) {
      m_stgCreateNewEddies =
          Context::getSolverProperty<MBool>("stgCreateNewEddies", m_solverId, AT_, &m_stgCreateNewEddies);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgInitialStartup
      <code>MFloat FvStructuredSolver::m_stgInitialStartup </code>\n
      default = <code>0</code>\n \n
      Initialize STG Method at Startup\n
      possible values are:
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgInitialStartup = false;
    m_stgInitialStartup = Context::getSolverProperty<MBool>("stgInitialStartup", m_solverId, AT_, &m_stgInitialStartup);

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgEddieLengthScales
      <code>MFloat FvStructuredSolver::m_stgEddieLengthScales </code>\n
      default = <code>0</code>\n \n
      Connect length scales to eddies, not cells.\n
      possible values are:
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgEddieLengthScales = false;
    if(Context::propertyExists("stgEddieLengthScales", m_solverId)) {
      m_stgEddieLengthScales =
          Context::getSolverProperty<MBool>("stgEddieLengthScales", m_solverId, AT_, &m_stgEddieLengthScales);
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section stgShapeFunction
      <code>MFloat FvStructuredSolver::m_stgFunction </code>\n
      default = <code>4</code>\n \n
      Shape function to be used in STG method.\n
      possible values are:
      <ul>
      <li>Integer >= 0</li>
      </ul>
      Keywords: <i>STG, STRUCTURED</i>
    */
    m_stgShapeFunction = 4;
    if(Context::propertyExists("stgShapeFunction", m_solverId)) {
      m_stgShapeFunction = Context::getSolverProperty<MInt>("stgShapeFunction", m_solverId, AT_, &m_stgShapeFunction);
    }

    if(m_stgInitialStartup) {
      // activate the nut FQ field
      FQ->neededFQVariables[FQ->NU_T] = 1;
    }

    m_stgNoEddieProperties = 6;
    if(m_stgEddieLengthScales) {
      m_stgNoEddieProperties = 9;
    }
    mAlloc(m_stgEddies, m_stgMaxNoEddies, m_stgNoEddieProperties, "m_solver->m_stgEddies", -F1, AT_);

    m_stgNoVariables = 20;
    MInt noSTGCells = m_nCells[0] * m_nCells[1] * 3;
    mAlloc(m_cells->stg_fq, m_stgNoVariables, noSTGCells, "m_cells->stg_fq", F0, AT_);

    m_stgBoxSize[0] = 0;
    m_stgBoxSize[1] = 0;
    m_stgBoxSize[2] = 0;

    m_log << "===========================================================" << endl
          << "                    STG PROPERTIES " << endl
          << "===========================================================" << endl
          << "Initial Start: " << m_stgInitialStartup << endl
          << "SubSup (Mixed subsonic/supersonic bc): " << m_stgSubSup << endl
          << "Supersonic BC: " << m_stgSupersonic << endl
          << "BLT 1,2,3: " << m_stgBLT1 << ", " << m_stgBLT2 << ", " << m_stgBLT3 << endl
          << "Delta0 inflow: " << m_stgDelta99Inflow << endl
          << "Length factors: " << m_stgLengthFactors[0] << ", " << m_stgLengthFactors[1] << ", "
          << m_stgLengthFactors[2] << endl
          << "Number of eddies: " << m_stgMaxNoEddies << endl
          << "Length scale exponent: " << m_stgExple << endl
          << "Eddie distribution: " << m_stgEddieDistribution << endl
          << "Create new eddies: " << m_stgCreateNewEddies << endl
          << "Eddie lengthscales: " << m_stgEddieLengthScales << endl
          << "Shape function: " << m_stgShapeFunction << endl
          << "Number of eddie properties: " << m_stgNoEddieProperties << endl
          << "Number of stg variables: " << m_stgNoVariables << endl
          << "===========================================================" << endl;

    switch(m_stgFace) {
      case 0:
      case 1:
        m_stgBoxSize[0] = m_nCells[0];
        m_stgBoxSize[1] = m_nCells[1];
        m_stgBoxSize[2] = 3;
        break;
      default:
        mTerm(1, AT_, "STG Method is not prepared for faces different than 0 or 1!");
    }
  }
}

template <MInt nDim>
void FvStructuredSolver<nDim>::setProfileBCProperties() {
  TRACE();

  ////////////////////////////////////////////////////////
  //////////////// BC 2600 ///////////////////////////////
  ////////////////////////////////////////////////////////

  m_bc2600IsActive = false;
  m_bc2600 = false;
  m_bc2600RootRank = -1;
  for(MInt i = 0; i < abs((MInt)m_windowInfo->globalStructuredBndryCndMaps.size()); ++i) {
    if(m_windowInfo->globalStructuredBndryCndMaps[i]->BC == 2600) {
      m_bc2600IsActive = true;
      break;
    }
  }

  if(m_bc2600IsActive) {
    // now look if this domain contains parts of this bc
    MInt localRank = 9999999;
    for(MInt i = 0; i < abs((MInt)m_windowInfo->physicalBCMap.size()); ++i) {
      if(m_windowInfo->physicalBCMap[i]->BC == 2600) {
        m_bc2600 = true;
        m_bc2600Face = m_windowInfo->physicalBCMap[i]->face;
        localRank = domainId();
        break;
      }
    }

    MPI_Allreduce(&localRank, &m_bc2600RootRank, 1, MPI_INT, MPI_MIN, m_StructuredComm, AT_, "localRank",
                  "m_bc2600RootRank");

    if(m_bc2600) {
      MPI_Comm_rank(m_commBC2600, &m_commBC2600MyRank);
    } else {
      m_commBC2600MyRank = -1;
    }

    /*! \property
      \page propertiesFVSTRCTRD
      \section initialStartup2600
      <code>MFloat FvStructuredSolver::m_bc2600InitialStartup </code>\n
      default = <code>0</code>\n \n
      Trigger to indicate the initial start of the BC 2600\n
      load the value from the field into the BC field.\n
      possible values are:
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>BC2600, STRUCTURED</i>
    */
    m_bc2600InitialStartup = false;
    m_bc2600InitialStartup =
        Context::getSolverProperty<MBool>("initialStartup2600", m_solverId, AT_, &m_bc2600InitialStartup);

    mAlloc(m_bc2600noCells, nDim, "m_bc2600noCells", 0, AT_);
    mAlloc(m_bc2600noActiveCells, nDim, "m_bc2600noCells", 0, AT_);
    mAlloc(m_bc2600noOffsetCells, nDim, "m_bc2600noCells", 0, AT_);
    if(m_bc2600) {
      for(MInt dim = 0; dim < nDim; dim++) {
        m_bc2600noOffsetCells[dim] = m_nOffsetCells[dim];
        m_bc2600noCells[dim] = m_nCells[dim];
        m_bc2600noActiveCells[dim] = m_bc2600noCells[dim] - 2 * m_noGhostLayers;
      }
      m_bc2600noOffsetCells[nDim - 1] = 0;
      m_bc2600noCells[nDim - 1] = m_noGhostLayers;
      m_bc2600noActiveCells[nDim - 1] = m_noGhostLayers;
      MInt noCellsBC = 1;
      for(MInt dim = 0; dim < nDim; dim++) {
        noCellsBC *= m_bc2600noCells[dim];
      }
      mAlloc(m_bc2600Variables, m_maxNoVariables, noCellsBC, "m_bc2600Variables", -123.456, AT_);
    }
  }

  ////////////////////////////////////////////////////////
  //////////////// BC 2601 ///////////////////////////////
  ////////////////////////////////////////////////////////
  m_bc2601IsActive = false;
  m_bc2601 = false;
  for(MInt i = 0; i < abs((MInt)m_windowInfo->globalStructuredBndryCndMaps.size()); ++i) {
    if(m_windowInfo->globalStructuredBndryCndMaps[i]->BC == 2601) {
      m_bc2601IsActive = true;
    }
  }

  if(m_bc2601IsActive) {
    /*! \property
      \page propertiesFVSTRCTRD
      \section initialStartup2601
      <code>MFloat FvStructuredSolver::m_bc2601InitialStartup </code>\n
      default = <code>0</code>\n \n
      Trigger to indicate the initial start of the BC 2601\n
      load the value from the field into the BC field.\n
      possible values are:
      <ul>
      <li>true/false</li>
      </ul>
      Keywords: <i>BC2601, STRUCTURED</i>
    */
    m_bc2601InitialStartup = false;
    m_bc2601InitialStartup =
        Context::getSolverProperty<MBool>("initialStartup2601", m_solverId, AT_, &m_bc2601InitialStartup);

    /*! \property
      \page propertiesFVSTRCTRD
      \section gammaEpsilon2601
      <code>MFloat FvStructuredSolver::m_bc2601GammaEpsilon </code>\n
      default = <code>0</code>\n \n
      Gamma Epsilon value for the BC2601\n
      Keywords: <i>BC2601, STRUCTURED</i>
    */
    m_bc2601GammaEpsilon = 0.12;
    m_bc2601GammaEpsilon =
        Context::getSolverProperty<MFloat>("gammaEpsilon2601", m_solverId, AT_, &m_bc2601GammaEpsilon);

    mAlloc(m_bc2601noCells, nDim, "m_bc2601noCells", 0, AT_);
    mAlloc(m_bc2601noActiveCells, nDim, "m_bc2601noCells", 0, AT_);
    mAlloc(m_bc2601noOffsetCells, nDim, "m_bc2601noCells", 0, AT_);
    if(m_bc2601) {
      for(MInt dim = 0; dim < nDim; dim++) {
        m_bc2601noOffsetCells[dim] = m_nOffsetCells[dim];
        m_bc2601noCells[dim] = m_nCells[dim];
        m_bc2601noActiveCells[dim] = m_bc2601noCells[dim] - 2 * m_noGhostLayers;
      }
      m_bc2601noOffsetCells[nDim - 2] = 0;
      m_bc2601noCells[nDim - 2] = m_noGhostLayers;
      m_bc2601noActiveCells[nDim - 2] = m_noGhostLayers;
      MInt noCellsBC = 1;
      for(MInt dim = 0; dim < nDim; dim++) {
        noCellsBC *= m_bc2601noCells[dim];
      }
      mAlloc(m_bc2601Variables, CV->noVariables, noCellsBC, "m_bc2601Variables", -123.456, AT_);
    }
  }
}


/** \brief Set which zones are RANS and which are LES or if full LES or full RANS
 *  \date Jan, 2015
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::setZonalProperties() {
  TRACE();

  MBool fullRANS = false;
  m_zoneType = "LES";
  m_rans = false;

  /*! \property
    \page propertiesFVSTRCTRD
    \section zonal
    <code>MFloat FvStructuredSolver::m_zonal </code>\n
    default = <code>0</code>\n \n
    Trigger a zonal computation.\n
    possible values are:
    <ul>
    <li>true/false</li>
    </ul>
    Keywords: <i>ZONAL, STRUCTURED</i>
  */
  m_zonal = false;
  if(Context::propertyExists("zonal", m_solverId)) {
    m_zonal = Context::getSolverProperty<MBool>("zonal", m_solverId, AT_);
  }

  /*! \property
    \page propertiesFVSTRCTRD
    \section fullRANS
    <code>MBool fullRANS </code>\n
    default = <code>0</code>\n \n
    Trigger a zonal computation.\n
    possible values are:
    <ul>
    <li>true/false</li>
    </ul>
    Keywords: <i>RANS, ZONAL, STRUCTURED</i>
  */
  if(Context::propertyExists("fullRANS", m_solverId)) {
    fullRANS = Context::getSolverProperty<MBool>("fullRANS", m_solverId, AT_, &fullRANS);
  }

  if(m_zonal) {
    // activate the nut FQ field
    FQ->neededFQVariables[FQ->AVG_RHO] = 1;
    FQ->neededFQVariables[FQ->AVG_U] = 1;
    FQ->neededFQVariables[FQ->AVG_V] = 1;
    FQ->neededFQVariables[FQ->AVG_W] = 1;
    FQ->neededFQVariables[FQ->AVG_P] = 1;

    m_zonalExchangeInterval = 5;
    m_zonalExchangeInterval =
        Context::getSolverProperty<MInt>("zonalExchangeInterval", m_solverId, AT_, &m_zonalExchangeInterval);

    m_zonalAveragingFactor = 128.0;
    m_zonalAveragingFactor =
        Context::getSolverProperty<MFloat>("zonalAvgFactor", m_solverId, AT_, &m_zonalAveragingFactor);


    /*! \property
      \page propertiesFVSTRCTRD
      \section noRansZones
      <code>MInt noRansZones </code>\n
      default = <code>0</code>\n \n
      Number of zones (solvers) with RANS.\n
      possible values are:
      <ul>
      <li>Integer >= 0</li>
      </ul>
      Keywords: <i>RANS, ZONAL, STRUCTURED</i>
    */
    MInt noRansZones = 0;
    noRansZones = Context::getSolverProperty<MInt>("noRansZones", m_solverId, AT_, &noRansZones);
    MIntScratchSpace ransZones(noRansZones, AT_, "ransZones");
    ransZones.fill(0);

    /*! \property
      \page propertiesFVSTRCTRD
      \section ransZone
      <code>MInt[noRansZones] ransZones </code>\n
      default = <code>0</code>\n \n
      IDs of the RANS solvers.\n
      possible values are:
      <ul>
      <li>Integer >= 0</li>
      </ul>
      Keywords: <i>RANS, ZONAL, STRUCTURED</i>
    */
    for(MInt RANS = 0; RANS < noRansZones; RANS++) {
      ransZones[RANS] = Context::getSolverProperty<MInt>("ransZone", m_solverId, AT_, &ransZones[RANS], RANS);
    }

    // Find out if own partition is RANS
    for(MInt RANS = 0; RANS < noRansZones; RANS++) {
      MInt blockID = m_grid->getMyBlockId();
      if(blockID == ransZones[RANS]) {
        m_zoneType = "RANS";
        m_rans = true;
      }
    }

    // Count the zones
    MInt NOZONES[2] = {0, 0};
    MInt NOZONESH[2] = {0, 0};

    if(m_zoneType == "RANS") {
      NOZONES[0] = 1;
      NOZONES[1] = 0;
    } else {
      NOZONES[0] = 0;
      NOZONES[1] = 1;
    }

    MPI_Allreduce(NOZONES, NOZONESH, 2, MPI_INT, MPI_SUM, m_StructuredComm, AT_, "NOZONES", "NOZONESH");

    // Give overview over RANS/LES zones
    if(domainId() == 0) {
      cout << "////////////////////////////////////////////////////////////////////////" << endl;
      cout << "No of RANS partitions: " << NOZONESH[0] << " , No of LES partitions: " << NOZONESH[1] << endl;
      cout << "////////////////////////////////////////////////////////////////////////" << endl;
    }
    m_log << "No of RANS partitions: " << NOZONESH[0] << " , No of LES partitions: " << NOZONESH[1] << endl;
  }

  if(fullRANS) {
    m_log << "Starting a full RANS computation" << endl;
    m_zoneType = "RANS";
    m_rans = true;
  }

  if(m_zonal || m_rans) {
    m_ransMethod =
        static_cast<RansMethod>(string2enum(Context::getSolverProperty<MString>("ransMethod", m_solverId, AT_)));

    if(noRansEquations(m_ransMethod) == 2) {
      if(!Context::propertyExists("rans2eq_mode", m_solverId))
        mTerm(1, "Usage of 2-eq. RANS model requires specification of rans2eq_mode: {init|production}");
      m_rans2eq_mode = Context::getSolverProperty<MString>("rans2eq_mode", m_solverId, AT_, &m_rans2eq_mode);
      if(m_rans2eq_mode != "init" && m_rans2eq_mode != "production") mTerm(1, "OMG! OMG!");
    }

    if(m_porous && m_ransMethod != RANS_KEPSILON)
      mTerm(1, "Porous RANS computation is only supported by k-epsilon model!");

    // Read properties required to compute inflow k & epsilon
    if(m_ransMethod == RANS_KEPSILON) {
      m_keps_nonDimType = true; // true=>non-dim with a_0^2
      m_keps_nonDimType = Context::getSolverProperty<MBool>("keps_nonDimType", m_solverId, AT_, &m_keps_nonDimType);

      // TODO_SS labels:FV don't force specification of m_I and m_epsScale -> only for a few BCs and ICs mandatory
      if(!Context::propertyExists("turbulenceIntensity", m_solverId))
        mTerm(1, "Usage of k-epsilon model requires specification of inflow turbulenceIntensity");
      m_I = Context::getSolverProperty<MFloat>("turbulenceIntensity", m_solverId, AT_, &m_I);


      const MBool turbLengthScaleExists = Context::propertyExists("turbulentLengthScale", m_solverId);
      const MBool turbViscRatioExists = Context::propertyExists("turbulentViscosityRatio", m_solverId);
      const MBool turbEpsInfty = Context::propertyExists("turbEpsInfty", m_solverId);
      if(turbLengthScaleExists + turbViscRatioExists + turbEpsInfty != 1)
        mTerm(1, "Usage of k-epsilon model requires the specification of either a turbulentLengthScale or a "
                 "turbulentViscosityRatio");

      if(turbLengthScaleExists) {
        m_kepsICMethod = 1;
        m_epsScale = Context::getSolverProperty<MFloat>("turbulentLengthScale", m_solverId, AT_);
      } else if(turbViscRatioExists) {
        m_kepsICMethod = 2;
        // m_epsScale = muTurb / muLam
        m_epsScale = Context::getSolverProperty<MFloat>("turbulentViscosityRatio", m_solverId, AT_);
      } else {
        m_kepsICMethod = 3;
        m_epsScale = Context::getSolverProperty<MFloat>("turbEpsInfty", m_solverId, AT_);
      }

      if(m_porous) {
        FQ->neededFQVariables[FQ->POROUS_INDICATOR] = 1;
        FQ->outputFQVariables[FQ->POROUS_INDICATOR] = false;
      }
    }

    FQ->neededFQVariables[FQ->NU_T] = 1;
    FQ->outputFQVariables[FQ->NU_T] = true;

    if(m_ransMethod == RANS_KEPSILON) {
      FQ->neededFQVariables[FQ->UTAU] = 1;
      FQ->outputFQVariables[FQ->UTAU] = false;
      if(m_porous) {
        FQ->neededFQVariables[FQ->UTAU2] = 1;
        FQ->outputFQVariables[FQ->UTAU2] = false;
      }
    }

    if(m_ransMethod == RANS_SA_DV || m_ransMethod == RANS_KEPSILON) {
      FQ->neededFQVariables[FQ->WALLDISTANCE] = 1;
      FQ->outputFQVariables[FQ->WALLDISTANCE] = false;
    }
  }

  if(m_rans) {
    m_ransTransPos = -1000000.0;
    if(Context::propertyExists("ransTransPos", m_solverId)) {
      m_ransTransPos = Context::getSolverProperty<MFloat>("ransTransPos", m_solverId, AT_, &m_ransTransPos);
    }
  }
}

template <MInt nDim>
void FvStructuredSolver<nDim>::allocateVariables() {
  // We will decide whether we will use RANS Variables or not depending on the solver

  if(m_zoneType == "RANS") {
    // number of RANS equations
    m_noRansEquations = noRansEquations(m_ransMethod);

    CV = make_unique<MConservativeVariables<nDim>>(m_noSpecies, m_noRansEquations);
    PV = make_unique<MPrimitiveVariables<nDim>>(m_noSpecies, m_noRansEquations);
  } else {
    // we only have LES solvers
    m_noRansEquations = 0;
    CV = make_unique<MConservativeVariables<nDim>>(m_noSpecies);
    PV = make_unique<MPrimitiveVariables<nDim>>(m_noSpecies);
  }

  // LES zones have 5 but RANS zone have 6 variables,
  // we have to use maxNoVariables for allocations and
  // IO related procedures which use MPI
  m_maxNoVariables = -1;
  MPI_Allreduce(&PV->noVariables, &m_maxNoVariables, 1, MPI_INT, MPI_MAX, m_StructuredComm, AT_, "PV->noVariables",
                "m_maxNoVariables");
  m_log << "Max number of variables: " << m_maxNoVariables << endl;
}


/**
 * \brief Reset the right hand side to zero
 * \author Pascal Meysonnat
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::resetRHS() {
  // as only 1D array is used the ghostcells will also need to have a right hand side variables
  // if storage need to be saved go over the inner loops and remove storge of ghost rhs
  const MInt noVars = CV->noVariables;
  for(MInt cellId = 0; cellId < m_noCells; cellId++) {
    for(MInt varId = 0; varId < noVars; varId++) {
      m_cells->rightHandSide[varId][cellId] = F0;
    }
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::computePV() {
  // don't do this here, because it
  // will be executed before exchange() and
  // then wrong primitve values are in the GC
  // computePrimitiveVariables();
}


template <MInt nDim>
template <MFloat (FvStructuredSolver<nDim>::*totalEnergy_func)(MInt) const>
void FvStructuredSolver<nDim>::computeConservativeVariables_() {
  m_log << "we got into the general formulation but should be in the other one" << endl;
  const MFloat FgammaMinusOne = F1 / (m_gamma - 1.0);
  MFloat** const RESTRICT cvars = m_cells->variables;
  MFloat** const RESTRICT pvars = m_cells->pvariables;
  MFloat rhoVelocity[3] = {0.0, 0.0, 0.0};

  for(MInt cellId = 0; cellId < m_noCells; cellId++) {
    MFloat velPOW2 = F0;
    // copy the density
    const MFloat rho = pvars[PV->RHO][cellId];
    // compute the rho * velocity
    for(MInt i = 0; i < nDim; i++) {
      rhoVelocity[i] = pvars[PV->VV[i]][cellId] * rho;
      velPOW2 += POW2(pvars[PV->VV[i]][cellId]);
    }

    // regular conservative variables
    cvars[CV->RHO][cellId] = rho;
    for(MInt i = 0; i < nDim; i++) {
      cvars[CV->RHO_VV[i]][cellId] = rhoVelocity[i];
    }

    cvars[CV->RHO_E][cellId] =
        pvars[PV->P][cellId] * FgammaMinusOne + F1B2 * rho * velPOW2 + (this->*totalEnergy_func)(cellId);

    // rans
    for(MInt ransVar = 0; ransVar < m_noRansEquations; ransVar++) {
      // pvars[PV->RANS_VAR[ransVar]][cellId] = mMax(pvars[PV->RANS_VAR[ransVar]][cellId], F0);
      //      cvars[CV->RANS_VAR[ransVar]][cellId] = mMax(pvars[PV->RANS_VAR[ransVar]][cellId], F0)*rho;
      cvars[CV->RANS_VAR[ransVar]][cellId] = pvars[PV->RANS_VAR[ransVar]][cellId] * rho;
    }

    // species
    for(MInt s = 0; s < m_noSpecies; s++) {
      cvars[CV->RHO_Y[s]][cellId] = pvars[PV->Y[s]][cellId] * rho;
    }
  }
}

template <MInt nDim>
void FvStructuredSolver<nDim>::computeConservativeVariables() {
  if(noRansEquations(m_ransMethod) == 2) {
    if(m_rans2eq_mode == "production")
      computeConservativeVariables_<&FvStructuredSolver::totalEnergy_twoEqRans>();
    else
      computeConservativeVariables_();
  } else
    computeConservativeVariables_();
}


template <MInt nDim>
void FvStructuredSolver<nDim>::saveVarToPrimitive(MInt cellId, MInt varId, MFloat var) {
  m_cells->pvariables[varId][cellId] = var;
}


template <MInt nDim>
void FvStructuredSolver<nDim>::setVolumeForce() {
  TRACE();

  switch(m_volumeForceMethod) {
    case 0:
      // default
      break;
    case 1:
      // To be used in combination with bc2402
      if(globalTimeStep % m_volumeForceUpdateInterval == 0 && m_RKStep == 0) {
        const MFloat relaxationFactor = 0.02;
        MPI_Allreduce(MPI_IN_PLACE, &m_inflowVelAvg, 1, MPI_DOUBLE, MPI_MAX, m_StructuredComm, AT_, "MPI_IN_PLACE",
                      "inflowVelAvg");
        m_log << "GlobalTimeStep=" << globalTimeStep << setprecision(6)
              << ": inflowVelAvg(targetVelAvg)=" << m_inflowVelAvg << "(" << PV->UInfinity
              << "), volumeForce=" << m_volumeForce[0] << " -> ";
        const MFloat deltaVelAvg = PV->UInfinity - m_inflowVelAvg;
        m_volumeForce[0] += deltaVelAvg / (m_volumeForceUpdateInterval * m_timeStep) * relaxationFactor;
        m_volumeForce[0] = std::max(m_volumeForce[0], 0.0);
        m_log << setprecision(6) << m_volumeForce[0] << endl;
        m_inflowVelAvg = -1.0;
      }
      break;
    default:
      mTerm(1, "Unknown volume force method!");
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::computeVolumeForces() {
  TRACE();

  for(MInt dim = 0; dim < nDim; dim++) {
    for(MInt cellId = 0; cellId < m_noCells; cellId++) {
      m_cells->rightHandSide[CV->RHO_VV[dim]][cellId] +=
          m_cells->variables[CV->RHO][cellId] * m_volumeForce[dim] * m_cells->cellJac[cellId];
      m_cells->rightHandSide[CV->RHO_E][cellId] +=
          m_cells->variables[CV->RHO_VV[dim]][cellId] * m_volumeForce[dim] * m_cells->cellJac[cellId];
    }
  }
}


/**
 * \brief Saves variables of a given box instead
 *    of whole domain. Box start and end indices
 *    can be given in the property file
 *
 * \author Marian Albers, Pascal Meysonnat (modifications)
 * \date:  Nov 1, 2015,
 */
template <>
void FvStructuredSolver<3>::saveBoxes() {
  constexpr MInt nDim = 3;
  // create a file
  // a) all boxes for each time in one file
  // b) all boxes in one file and one file per output
  stringstream filename;
  filename << m_boxOutputDir << "boxOutput" << m_outputIterationNumber << m_outputFormat;

  ParallelIoHdf5 pio(filename.str(), maia::parallel_io::PIO_REPLACE, m_StructuredComm);

  writeHeaderAttributes(&pio, "boxes");
  writePropertiesAsAttributes(&pio, "");
  pio.setAttribute(m_boxNoBoxes, "noBoxes", "");

  // create datasets
  ParallelIo::size_type localBoxSize[nDim]{0};
  ParallelIo::size_type localBoxOffset[nDim]{0};
  ParallelIo::size_type globalBoxSize[nDim]{0};
  ParallelIo::size_type localDomainBoxOffset[nDim]{0};

  for(MInt i = 0; i < m_noBlocks; ++i) {
    for(MInt b = 0; b < m_boxNoBoxes; ++b) {
      if(m_boxBlock[b] == i) {
        // create a dataset for the solver

        for(MInt dim = 0; dim < nDim; ++dim) {
          globalBoxSize[dim] = m_boxSize[b][dim];
        }

        stringstream pathName;
        pathName << "/box" << b;

        pio.setAttribute(m_boxOffset[b][2], "offseti", pathName.str());
        pio.setAttribute(m_boxOffset[b][1], "offsetj", pathName.str());
        pio.setAttribute(m_boxOffset[b][0], "offsetk", pathName.str());

        pio.setAttribute(m_boxSize[b][2], "sizei", pathName.str());
        pio.setAttribute(m_boxSize[b][1], "sizej", pathName.str());
        pio.setAttribute(m_boxSize[b][0], "sizek", pathName.str());

        pio.setAttribute(i, "blockId", pathName.str());

        MInt hasCoordinates = 0;

        for(MInt v = 0; v < m_maxNoVariables; v++) {
          pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), m_pvariableNames[v], nDim, globalBoxSize);
        }
        // create datasets for fq-field
        for(MInt v = 0; v < FQ->noFQVariables; v++) {
          if(FQ->fqWriteOutputBoxes[v]) {
            pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), FQ->fqNames[v], nDim, globalBoxSize);
          }
        }
        // create datasets for the variables
        if(m_boxWriteCoordinates) {
          hasCoordinates = 1;
          pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), "x", nDim, globalBoxSize);
          pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), "y", nDim, globalBoxSize);
          pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), "z", nDim, globalBoxSize);
        }
        // write output to check if coordinates are contained within the variable list
        pio.setAttribute(hasCoordinates, "hasCoordinates", pathName.str());
      }
    }
  }

  for(MInt b = 0; b < m_boxNoBoxes; ++b) {
    // check if the box is contained your inputsolverId
    if(m_boxBlock[b] == m_blockId
       && ((m_nOffsetCells[2] <= m_boxOffset[b][2] && m_boxOffset[b][2] < m_nOffsetCells[2] + m_nActiveCells[2])
           || (m_boxOffset[b][2] <= m_nOffsetCells[2] && m_nOffsetCells[2] < m_boxOffset[b][2] + m_boxSize[b][2]))
       && ((m_nOffsetCells[1] <= m_boxOffset[b][1] && m_boxOffset[b][1] < m_nOffsetCells[1] + m_nActiveCells[1])
           || (m_boxOffset[b][1] <= m_nOffsetCells[1] && m_nOffsetCells[1] < m_boxOffset[b][1] + m_boxSize[b][1]))
       && ((m_nOffsetCells[0] <= m_boxOffset[b][0] && m_boxOffset[b][0] < m_nOffsetCells[0] + m_nActiveCells[0])
           || (m_boxOffset[b][0] <= m_nOffsetCells[0]
               && m_nOffsetCells[0] < m_boxOffset[b][0] + m_boxSize[b][0]))) { // the box is contained
      // get the size of the box!!!

      for(MInt dim = 0; dim < nDim; ++dim) {
        if(m_nOffsetCells[dim] <= m_boxOffset[b][dim]
           && m_boxOffset[b][dim] + m_boxSize[b][dim] < m_nOffsetCells[dim] + m_nActiveCells[dim]) {
          localBoxSize[dim] = m_boxSize[b][dim];
          localBoxOffset[dim] = 0;
          localDomainBoxOffset[dim] = m_boxOffset[b][dim] - m_nOffsetCells[dim];
        } else if(m_nOffsetCells[dim] <= m_boxOffset[b][dim]) {
          localBoxSize[dim] = (m_nOffsetCells[dim] + m_nActiveCells[dim]) - m_boxOffset[b][dim];
          localBoxOffset[dim] = 0;
          localDomainBoxOffset[dim] = m_boxOffset[b][dim] - m_nOffsetCells[dim];
        } else if(m_boxOffset[b][dim] <= m_nOffsetCells[dim]
                  && m_nOffsetCells[dim] + m_nActiveCells[dim] < m_boxOffset[b][dim] + m_boxSize[b][dim]) {
          localBoxSize[dim] = m_nActiveCells[dim];
          localBoxOffset[dim] = m_nOffsetCells[dim] - m_boxOffset[b][dim];
          localDomainBoxOffset[dim] = 0;
        } else {
          localBoxSize[dim] = (m_boxOffset[b][dim] + m_boxSize[b][dim]) - m_nOffsetCells[dim];
          localBoxOffset[dim] = m_nOffsetCells[dim] - m_boxOffset[b][dim];
          localDomainBoxOffset[dim] = 0;
        }
      }

      stringstream pathName;
      pathName << "/box" << b;
      MInt totalLocalSize = 1;
      for(MInt dim = 0; dim < nDim; ++dim) {
        totalLocalSize *= localBoxSize[dim];
      }
      MInt noFields = m_maxNoVariables;
      if(FQ->noFQBoxOutput > 0) noFields += FQ->noFQBoxOutput;
      if(m_boxWriteCoordinates) noFields += nDim;


      MFloatScratchSpace localBoxVar(noFields * totalLocalSize, AT_, "local Box Variables");

      MInt cellId = 0;
      MInt localId = 0;
      MInt offset = 0;

      MFloat noVars = m_maxNoVariables;
      for(MInt var = 0; var < noVars; ++var) {
        for(MInt k = m_noGhostLayers + localDomainBoxOffset[0];
            k < m_noGhostLayers + localDomainBoxOffset[0] + localBoxSize[0];
            ++k) {
          for(MInt j = m_noGhostLayers + localDomainBoxOffset[1];
              j < m_noGhostLayers + localDomainBoxOffset[1] + localBoxSize[1];
              ++j) {
            for(MInt i = m_noGhostLayers + localDomainBoxOffset[2];
                i < m_noGhostLayers + localDomainBoxOffset[2] + localBoxSize[2];
                ++i) {
              cellId = i + (j + k * m_nCells[1]) * m_nCells[2];
              MInt boxI = i - m_noGhostLayers - localDomainBoxOffset[2];
              MInt boxJ = j - m_noGhostLayers - localDomainBoxOffset[1];
              MInt boxK = k - m_noGhostLayers - localDomainBoxOffset[0];
              localId = var * totalLocalSize + (boxI + (boxJ + boxK * localBoxSize[1]) * localBoxSize[2]);
              localBoxVar[localId] = m_cells->pvariables[var][cellId];
            }
          }
        }
      }
      offset += m_maxNoVariables;

      if(FQ->noFQBoxOutput > 0) {
        for(MInt v = 0; v < FQ->noFQVariables; ++v) {
          if(FQ->fqWriteOutputBoxes[v]) {
            for(MInt k = m_noGhostLayers + localDomainBoxOffset[0];
                k < m_noGhostLayers + localDomainBoxOffset[0] + localBoxSize[0];
                ++k) {
              for(MInt j = m_noGhostLayers + localDomainBoxOffset[1];
                  j < m_noGhostLayers + localDomainBoxOffset[1] + localBoxSize[1];
                  ++j) {
                for(MInt i = m_noGhostLayers + localDomainBoxOffset[2];
                    i < m_noGhostLayers + localDomainBoxOffset[2] + localBoxSize[2];
                    ++i) {
                  cellId = i + (j + k * m_nCells[1]) * m_nCells[2];
                  MInt boxI = i - m_noGhostLayers - localDomainBoxOffset[2];
                  MInt boxJ = j - m_noGhostLayers - localDomainBoxOffset[1];
                  MInt boxK = k - m_noGhostLayers - localDomainBoxOffset[0];
                  localId = (offset)*totalLocalSize + (boxI + (boxJ + boxK * localBoxSize[1]) * localBoxSize[2]);
                  localBoxVar[localId] = m_cells->fq[v][cellId];
                }
              }
            }
            ++offset;
          }
        }
      }

      if(m_boxWriteCoordinates) {
        for(MInt dim = 0; dim < nDim; dim++) {
          for(MInt k = m_noGhostLayers + localDomainBoxOffset[0];
              k < m_noGhostLayers + localDomainBoxOffset[0] + localBoxSize[0];
              ++k) {
            for(MInt j = m_noGhostLayers + localDomainBoxOffset[1];
                j < m_noGhostLayers + localDomainBoxOffset[1] + localBoxSize[1];
                ++j) {
              for(MInt i = m_noGhostLayers + localDomainBoxOffset[2];
                  i < m_noGhostLayers + localDomainBoxOffset[2] + localBoxSize[2];
                  ++i) {
                cellId = i + (j + k * m_nCells[1]) * m_nCells[2];
                MInt boxI = i - m_noGhostLayers - localDomainBoxOffset[2];
                MInt boxJ = j - m_noGhostLayers - localDomainBoxOffset[1];
                MInt boxK = k - m_noGhostLayers - localDomainBoxOffset[0];
                localId = (offset + dim) * totalLocalSize + (boxI + (boxJ + boxK * localBoxSize[1]) * localBoxSize[2]);
                localBoxVar[localId] = m_cells->coordinates[dim][cellId];
              }
            }
          }
        }
        offset += nDim;
      }
      ///////////////////////////////////
      ///////write out the data!/////////
      ///////////////////////////////////
      //--> /primitive/conservative variables
      for(MInt v = 0; v < m_maxNoVariables; v++) {
        pio.writeArray(&localBoxVar[v * totalLocalSize], pathName.str(), m_pvariableNames[v], nDim, localBoxOffset,
                       localBoxSize);
      }
      offset = m_maxNoVariables;
      //--> fq field
      for(MInt v = 0; v < FQ->noFQVariables; ++v) {
        if(FQ->fqWriteOutputBoxes[v]) {
          pio.writeArray(&localBoxVar[(offset)*totalLocalSize], pathName.str(), FQ->fqNames[v], nDim, localBoxOffset,
                         localBoxSize);
          offset++;
        }
      }

      if(m_boxWriteCoordinates) {
        pio.writeArray(&localBoxVar[(offset + 0) * totalLocalSize], pathName.str(), "x", nDim, localBoxOffset,
                       localBoxSize);
        pio.writeArray(&localBoxVar[(offset + 1) * totalLocalSize], pathName.str(), "y", nDim, localBoxOffset,
                       localBoxSize);
        pio.writeArray(&localBoxVar[(offset + 2) * totalLocalSize], pathName.str(), "z", nDim, localBoxOffset,
                       localBoxSize);
        offset += nDim;
      }

    } else { // write out nothing as box is not contained
      stringstream pathName;
      pathName << "/box" << b;
      for(MInt dim = 0; dim < nDim; ++dim) {
        localBoxSize[dim] = 0;
        localBoxOffset[dim] = 0;
      }
      MFloat empty = 0;
      for(MInt v = 0; v < m_maxNoVariables; ++v) {
        pio.writeArray(&empty, pathName.str(), m_pvariableNames[v], nDim, localBoxOffset, localBoxSize);
      }
      for(MInt v = 0; v < FQ->noFQVariables; ++v) {
        if(FQ->fqWriteOutputBoxes[v]) {
          pio.writeArray(&empty, pathName.str(), FQ->fqNames[v], nDim, localBoxOffset, localBoxSize);
        }
      }

      if(m_boxWriteCoordinates) {
        pio.writeArray(&empty, pathName.str(), "x", nDim, localBoxOffset, localBoxSize);
        pio.writeArray(&empty, pathName.str(), "y", nDim, localBoxOffset, localBoxSize);
        pio.writeArray(&empty, pathName.str(), "z", nDim, localBoxOffset, localBoxSize);
      }
    }
  }
}

/**
 * \brief Saves variables of a given box instead
 *    of whole domain. Box start and end indices
 *    can be given in the property file
 *
 * \author Marian Albers, Pascal Meysonnat (modifications)
 * \date:  Nov 1, 2015,
 */
template <>
void FvStructuredSolver<2>::saveBoxes() {
  constexpr MInt nDim = 2;
  // create a file
  // a) all boxes for each time in one file
  // b) all boxes in one file and one file per output
  stringstream filename;
  filename << m_boxOutputDir << "boxOutput" << m_outputIterationNumber << m_outputFormat;

  ParallelIoHdf5 pio(filename.str(), maia::parallel_io::PIO_REPLACE, m_StructuredComm);

  writeHeaderAttributes(&pio, "boxes");
  writePropertiesAsAttributes(&pio, "");
  pio.setAttribute(m_boxNoBoxes, "noBoxes", "");

  // create datasets
  ParallelIo::size_type localBoxSize[nDim]{0};
  ParallelIo::size_type localBoxOffset[nDim]{0};
  ParallelIo::size_type globalBoxSize[nDim]{0};
  ParallelIo::size_type localDomainBoxOffset[nDim]{0};


  for(MInt i = 0; i < m_noBlocks; ++i) {
    for(MInt b = 0; b < m_boxNoBoxes; ++b) {
      if(m_boxBlock[b] == i) {
        // create a dataset for the solver

        for(MInt dim = 0; dim < nDim; ++dim) {
          globalBoxSize[dim] = m_boxSize[b][dim];
        }

        stringstream pathName;
        pathName << "/box" << b;

        pio.setAttribute(m_boxOffset[b][2], "offseti", pathName.str());
        pio.setAttribute(m_boxOffset[b][1], "offsetj", pathName.str());

        pio.setAttribute(m_boxSize[b][2], "sizei", pathName.str());
        pio.setAttribute(m_boxSize[b][1], "sizej", pathName.str());

        pio.setAttribute(i, "blockId", pathName.str());

        MInt hasCoordinates = 0;

        for(MInt v = 0; v < m_maxNoVariables; v++) {
          pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), m_pvariableNames[v], nDim, globalBoxSize);
        }
        // create datasets for fq-field
        for(MInt v = 0; v < FQ->noFQVariables; v++) {
          if(FQ->fqWriteOutputBoxes[v]) {
            pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), FQ->fqNames[v], nDim, globalBoxSize);
          }
        }
        // create datasets for the variables
        if(m_boxWriteCoordinates) {
          hasCoordinates = 1;
          pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), "x", nDim, globalBoxSize);
          pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName.str(), "y", nDim, globalBoxSize);
        }
        // write output to check if coordinates are contained within the variable list
        pio.setAttribute(hasCoordinates, "hasCoordinates", pathName.str());
      }
    }
  }

  for(MInt b = 0; b < m_boxNoBoxes; ++b) {
    // check if the box is contained your inputsolverId
    if(m_boxBlock[b] == m_blockId
       && ((m_nOffsetCells[1] <= m_boxOffset[b][1] && m_boxOffset[b][1] < m_nOffsetCells[1] + m_nActiveCells[1])
           || (m_boxOffset[b][1] <= m_nOffsetCells[1] && m_nOffsetCells[1] < m_boxOffset[b][1] + m_boxSize[b][1]))
       && ((m_nOffsetCells[0] <= m_boxOffset[b][0] && m_boxOffset[b][0] < m_nOffsetCells[0] + m_nActiveCells[0])
           || (m_boxOffset[b][0] <= m_nOffsetCells[0]
               && m_nOffsetCells[0] < m_boxOffset[b][0] + m_boxSize[b][0]))) { // the box is contained
      // get the size of the box!!!

      for(MInt dim = 0; dim < nDim; ++dim) {
        if(m_nOffsetCells[dim] <= m_boxOffset[b][dim]
           && m_boxOffset[b][dim] + m_boxSize[b][dim] < m_nOffsetCells[dim] + m_nActiveCells[dim]) {
          localBoxSize[dim] = m_boxSize[b][dim];
          localBoxOffset[dim] = 0;
          localDomainBoxOffset[dim] = m_boxOffset[b][dim] - m_nOffsetCells[dim];
        } else if(m_nOffsetCells[dim] <= m_boxOffset[b][dim]) {
          localBoxSize[dim] = (m_nOffsetCells[dim] + m_nActiveCells[dim]) - m_boxOffset[b][dim];
          localBoxOffset[dim] = 0;
          localDomainBoxOffset[dim] = m_boxOffset[b][dim] - m_nOffsetCells[dim];
        } else if(m_boxOffset[b][dim] <= m_nOffsetCells[dim]
                  && m_nOffsetCells[dim] + m_nActiveCells[dim] < m_boxOffset[b][dim] + m_boxSize[b][dim]) {
          localBoxSize[dim] = m_nActiveCells[dim];
          localBoxOffset[dim] = m_nOffsetCells[dim] - m_boxOffset[b][dim];
          localDomainBoxOffset[dim] = 0;
        } else {
          localBoxSize[dim] = (m_boxOffset[b][dim] + m_boxSize[b][dim]) - m_nOffsetCells[dim];
          localBoxOffset[dim] = m_nOffsetCells[dim] - m_boxOffset[b][dim];
          localDomainBoxOffset[dim] = 0;
        }
      }

      stringstream pathName;
      pathName << "/box" << b;
      MInt totalLocalSize = 1;
      for(MInt dim = 0; dim < nDim; ++dim) {
        totalLocalSize *= localBoxSize[dim];
      }
      MInt noFields = m_maxNoVariables;
      if(FQ->noFQBoxOutput > 0) noFields += FQ->noFQBoxOutput;
      if(m_boxWriteCoordinates) noFields += nDim;


      MFloatScratchSpace localBoxVar(noFields * totalLocalSize, AT_, "local Box Variables");

      MInt cellId = 0;
      MInt localId = 0;
      MInt offset = 0;

      MFloat noVars = m_maxNoVariables;
      for(MInt var = 0; var < noVars; ++var) {
        for(MInt j = m_noGhostLayers + localDomainBoxOffset[0];
            j < m_noGhostLayers + localDomainBoxOffset[0] + localBoxSize[0];
            ++j) {
          for(MInt i = m_noGhostLayers + localDomainBoxOffset[1];
              i < m_noGhostLayers + localDomainBoxOffset[1] + localBoxSize[1];
              ++i) {
            cellId = i + j * m_nCells[1];
            MInt boxI = i - m_noGhostLayers - localDomainBoxOffset[1];
            MInt boxJ = j - m_noGhostLayers - localDomainBoxOffset[0];
            localId = var * totalLocalSize + (boxI + boxJ * localBoxSize[1]);
            localBoxVar[localId] = m_cells->pvariables[var][cellId];
          }
        }
      }
      offset += m_maxNoVariables;

      if(FQ->noFQBoxOutput > 0) {
        for(MInt v = 0; v < FQ->noFQVariables; ++v) {
          if(FQ->fqWriteOutputBoxes[v]) {
            for(MInt j = m_noGhostLayers + localDomainBoxOffset[0];
                j < m_noGhostLayers + localDomainBoxOffset[0] + localBoxSize[0];
                ++j) {
              for(MInt i = m_noGhostLayers + localDomainBoxOffset[1];
                  i < m_noGhostLayers + localDomainBoxOffset[1] + localBoxSize[1];
                  ++i) {
                cellId = i + j * m_nCells[1];
                MInt boxI = i - m_noGhostLayers - localDomainBoxOffset[1];
                MInt boxJ = j - m_noGhostLayers - localDomainBoxOffset[0];
                localId = (offset)*totalLocalSize + (boxI + boxJ * localBoxSize[1]);
                localBoxVar[localId] = m_cells->fq[v][cellId];
              }
            }
            ++offset;
          }
        }
      }

      if(m_boxWriteCoordinates) {
        for(MInt dim = 0; dim < nDim; dim++) {
          for(MInt j = m_noGhostLayers + localDomainBoxOffset[0];
              j < m_noGhostLayers + localDomainBoxOffset[0] + localBoxSize[0];
              ++j) {
            for(MInt i = m_noGhostLayers + localDomainBoxOffset[1];
                i < m_noGhostLayers + localDomainBoxOffset[1] + localBoxSize[1];
                ++i) {
              cellId = i + j * m_nCells[1];
              MInt boxI = i - m_noGhostLayers - localDomainBoxOffset[1];
              MInt boxJ = j - m_noGhostLayers - localDomainBoxOffset[0];
              localId = (offset + dim) * totalLocalSize + (boxI + boxJ * localBoxSize[1]);
              localBoxVar[localId] = m_cells->coordinates[dim][cellId];
            }
          }
        }
        offset += nDim;
      }
      ///////////////////////////////////
      ///////write out the data!/////////
      ///////////////////////////////////
      //--> /primitive/conservative variables
      for(MInt v = 0; v < m_maxNoVariables; v++) {
        pio.writeArray(&localBoxVar[v * totalLocalSize], pathName.str(), m_pvariableNames[v], nDim, localBoxOffset,
                       localBoxSize);
      }
      offset = m_maxNoVariables;
      //--> fq field
      for(MInt v = 0; v < FQ->noFQVariables; ++v) {
        if(FQ->fqWriteOutputBoxes[v]) {
          pio.writeArray(&localBoxVar[(offset)*totalLocalSize], pathName.str(), FQ->fqNames[v], nDim, localBoxOffset,
                         localBoxSize);
          offset++;
        }
      }

      if(m_boxWriteCoordinates) {
        pio.writeArray(&localBoxVar[(offset + 0) * totalLocalSize], pathName.str(), "x", nDim, localBoxOffset,
                       localBoxSize);
        pio.writeArray(&localBoxVar[(offset + 1) * totalLocalSize], pathName.str(), "y", nDim, localBoxOffset,
                       localBoxSize);
        offset += nDim;
      }

    } else { // write out nothing as box is not contained
      stringstream pathName;
      pathName << "/box" << b;
      for(MInt dim = 0; dim < nDim; ++dim) {
        localBoxSize[dim] = 0;
        localBoxOffset[dim] = 0;
      }
      MFloat empty = 0;
      for(MInt v = 0; v < m_maxNoVariables; ++v) {
        pio.writeArray(&empty, pathName.str(), m_pvariableNames[v], nDim, localBoxOffset, localBoxSize);
      }
      for(MInt v = 0; v < FQ->noFQVariables; ++v) {
        if(FQ->fqWriteOutputBoxes[v]) {
          pio.writeArray(&empty, pathName.str(), FQ->fqNames[v], nDim, localBoxOffset, localBoxSize);
        }
      }

      if(m_boxWriteCoordinates) {
        pio.writeArray(&empty, pathName.str(), "x", nDim, localBoxOffset, localBoxSize);
        pio.writeArray(&empty, pathName.str(), "y", nDim, localBoxOffset, localBoxSize);
      }
    }
  }
}


/**
 * \brief Overloaded version of writeHeaderAttributes that receives ParallelIoHdf5 object pointer
 *  instead of 'fileId'
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::writeHeaderAttributes(ParallelIoHdf5* pio, MString fileType) {
  stringstream gridFileName;
  MString gridNameStr = "";
  if(m_movingGrid && m_movingGridSaveGrid) {
    gridFileName << "grid" << globalTimeStep << m_outputFormat;
    gridNameStr = gridFileName.str();
  } else {
    gridNameStr = "../" + m_grid->m_gridInputFileName;
  }

  pio->setAttribute(gridNameStr, "gridFile", "");
  pio->setAttribute(m_grid->m_uID, "UID", "");
  pio->setAttribute(fileType, "filetype", "");

  const MInt zonal = (MInt)m_zonal;
  pio->setAttribute(zonal, "zonal", "");
  pio->setAttribute(m_noBlocks, "noBlocks", "");
}


/**
 * \brief Overloaded version of writePropertiesAsAttributes that receives ParallelIoHdf5 object pointer
 *  instead of 'fileId'
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::writePropertiesAsAttributes(ParallelIoHdf5* pio, MString path) {
  pio->setAttribute(m_Ma, "Ma", path);
  pio->setAttribute(m_Re, "Re", path);
  pio->setAttribute(m_Pr, "Pr", path);
  pio->setAttribute(m_timeStep, "timeStep", path);
  pio->setAttribute(m_time, "time", path);
  pio->setAttribute(m_physicalTimeStep, "physicalTimeStep", path);
  pio->setAttribute(m_physicalTime, "physicalTime", path);
  pio->setAttribute(globalTimeStep, "globalTimeStep", path);
  pio->setAttribute(m_firstMaxResidual, "firstMaxResidual", path);
  pio->setAttribute(m_firstAvrgResidual, "firstAvrgResidual", path);

  // save the time(Step) at which the grid motion started
  // does only work safely for constant time step
  if(m_movingGrid) {
    pio->setAttribute(m_movingGridStepOffset, "movingGridStepOffset", path);
    pio->setAttribute(m_movingGridTimeOffset, "movingGridTimeOffset", path);
    // save whether or not the wave time step has been computed
    if(m_travelingWave || m_streamwiseTravelingWave) {
      pio->setAttribute(m_waveTimeStepComputed, "waveTimeStepComputed", path);
      pio->setAttribute(m_waveNoStepsPerCell, "waveNoStepsPerCell", path);
    }
  }

  if(m_stgIsActive) {
    pio->setAttribute(m_stgMaxNoEddies, "stgNRAN", path);
  }

  if(m_volumeForceMethod != 0) {
    pio->setAttribute(m_volumeForce[0], "volumeForce", path);
  }
}


/**
 * \author Pascal S. Meysonnat
 * \date 03.03.2016
 */

template <MInt nDim>
void FvStructuredSolver<nDim>::saveSolverSolution(MBool forceOutput, const MBool finalTimeStep) {
  RECORD_TIMER_START(m_timers[Timers::Run]);
  RECORD_TIMER_START(m_timers[Timers::SaveOutput]);

  // run postprocessing in-solve routines
  this->postprocessInSolve();

  // Function to write the solution to file with iolibrary
  MBool writeSolution = false;
  MBool writeBox = false;
  MBool writeNodalBox = false;
  MBool writeIntpPoints = false;
  MBool writeAux = false;
  MBool computeForces = false;
  MBool writeForces = false;
  MBool computeAsciiCells = false;
  MBool writeAsciiCells = false;

  // first find out which writeOut-mode we use (iteration or convective unit intervals)
  // then check which functions should write out in this timestep
  if(m_useConvectiveUnitWrite) {
    // in this mode we check the convective unit intervals
    // and write out files of each type if activated
    // activation is done by setting the interval
    // to a value greater than 0 (boxOutputInterval = 1)
    if(m_physicalTime - (MFloat)(m_noConvectiveOutputs)*m_convectiveUnitInterval >= m_convectiveUnitInterval) {
      // restart file output is still triggered by iteration counter
      writeSolution = isInInterval(m_solutionInterval);
      forceOutput = writeSolution;

      // activate the desired outputs
      writeSolution = m_sampleSolutionFiles;
      writeBox = (m_boxOutputInterval > 0);
      writeNodalBox = (m_nodalBoxOutputInterval > 0);
      writeIntpPoints = (m_intpPointsOutputInterval > 0);
      writeAux = (m_forceOutputInterval > 0);
      computeForces = (m_forceAsciiComputeInterval > 0);

      m_noConvectiveOutputs++;
      m_outputIterationNumber = m_noConvectiveOutputs;
    }
  } else {
    // in this mode we check the iteration intervals
    // for each writeOut-type (solution, box, line, aux)
    writeSolution = isInInterval(m_solutionInterval);
    writeBox = isInInterval(m_boxOutputInterval);
    writeNodalBox = isInInterval(m_nodalBoxOutputInterval);
    writeIntpPoints = isInInterval(m_intpPointsOutputInterval);
    writeAux = isInInterval(m_forceOutputInterval);
    computeForces = isInInterval(m_forceAsciiComputeInterval);
    computeAsciiCells = isInInterval(m_pointsToAsciiComputeInterval);
    if(writeSolution || finalTimeStep) {
      writeForces = true;
      writeAsciiCells = true;
    }


    m_outputIterationNumber = globalTimeStep;
  }

  // compute vorticity if necessary
  if(m_vorticityOutput && (writeSolution || writeBox || forceOutput)) {
    computeVorticity();
  }

  // compute velocity if wanted
  if(m_computeLambda2 && (writeSolution || writeBox || forceOutput)) {
    computeLambda2Criterion();
  }

  // boxes, auxdata and lines
  // are only available for 3D checked by function pointer

  RECORD_TIMER_START(m_timers[Timers::SaveBoxes]);
  if(writeBox) {
    saveBoxes();
  }
  if(writeNodalBox) {
    saveNodalBoxes();
  }
  RECORD_TIMER_STOP(m_timers[Timers::SaveBoxes]);

  RECORD_TIMER_START(m_timers[Timers::SaveAuxdata]);
  if(writeAux) {
    saveAuxData();
  }
  RECORD_TIMER_STOP(m_timers[Timers::SaveAuxdata]);
  RECORD_TIMER_START(m_timers[Timers::SaveForces]);
  if(computeForces) {
    saveForcesToAsciiFile(writeForces);
  }

  if(computeAsciiCells) {
    savePointsToAsciiFile(writeAsciiCells);
  }

  RECORD_TIMER_STOP(m_timers[Timers::SaveForces]);
  IF_CONSTEXPR(nDim > 2) { // needs also to be implemented for 2d !!!!!!!!!!!!!!
    RECORD_TIMER_START(m_timers[Timers::SaveIntpPoints]);
    if(writeIntpPoints) {
      saveInterpolatedPoints();
    }
    RECORD_TIMER_STOP(m_timers[Timers::SaveIntpPoints]);
  }

  if(writeSolution || forceOutput || finalTimeStep) {
    RECORD_TIMER_START(m_timers[Timers::SaveSolution]);
    // save out the partitions also if desired, i.e, for debugging purposses
    if(m_savePartitionOutput) {
      savePartitions();
    }
    // save postprocessing variables if activated
    saveAverageRestart();
    // save solution/restart file
    saveSolution(forceOutput);
    m_lastOutputTimeStep = globalTimeStep;
    RECORD_TIMER_STOP(m_timers[Timers::SaveSolution]);
  }
  RECORD_TIMER_STOP(m_timers[Timers::SaveOutput]);
  RECORD_TIMER_STOP(m_timers[Timers::Run]);
}

template <MInt nDim>
MBool FvStructuredSolver<nDim>::isInInterval(MInt interval) {
  if(interval > 0) {
    if((globalTimeStep - m_outputOffset) % interval == 0 && globalTimeStep - m_outputOffset >= 0) {
      return true;
    }
  }
  return false;
}

/**
 * \author Pascal S. Meysonnat
 * \brief Saves the soution to hdf5 file
 * \date 17.08.2018
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::saveSolution(MBool forceOutput) {
  stringstream fileName;
  fileName << m_solutionOutput << m_outputIterationNumber << m_outputFormat;

  if(m_movingGrid) {
    if(m_movingGridSaveGrid) {
      m_grid->writeGrid(m_solutionOutput, m_outputFormat);
    }
  }

  m_log << "writing Solution file " << fileName.str() << " ... forceOutput: " << forceOutput << endl;

  // Check if file exists
  MInt fileMode = -1;
  // if (FILE *file = fopen((fileName.str()).c_str(), "r")) { // file does exist
  //   fclose(file);
  //   fileMode = maia::parallel_io::PIO_APPEND;
  // } else { // file does not exist
  fileMode = maia::parallel_io::PIO_REPLACE;
  //}

  ParallelIoHdf5 pio(fileName.str(), fileMode, m_StructuredComm);

  writeHeaderAttributes(&pio, "solution");
  writePropertiesAsAttributes(&pio, "");

  ParallelIo::size_type allCells[3] = {0, 0, 0};
  ParallelIo::size_type stgNoEddieFields = 1200;
  MString stgGlobalPathStr = "stgGlobal";

  for(MInt i = 0; i < m_noBlocks; i++) {
    for(MInt j = 0; j < nDim; j++) {
      allCells[j] = m_grid->getBlockNoCells(i, j);
    }
    // create datasets for the io library
    stringstream path;
    path << i;
    MString blockPathStr = "block";
    blockPathStr += path.str();
    const char* blockPath = blockPathStr.c_str();

    ////////////////////////////////////////////////
    ///////// Create Primitive/Conservative Variables ////////
    ////////////////////////////////////////////////
    m_log << "writing primitive Output" << endl;
    for(MInt v = 0; v < m_maxNoVariables; v++) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, m_pvariableNames[v], nDim, allCells);
    }

    ////////////////////////////////////////////////
    ///////// Create FQ Information ////////////////
    ////////////////////////////////////////////////
    if(FQ->noFQVariables > 0) {
      for(MInt v = 0; v < FQ->noFQVariables; ++v) {
        if(FQ->fqWriteOutput[v]) {
          pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, FQ->fqNames[v], nDim, allCells);
        }
      }
    }

    ////////////////////////////////////////////////
    ///////// Create BC2600 Information ////////////
    ////////////////////////////////////////////////
    if(m_bc2600IsActive) {
      ParallelIo::size_type allCells2600[3] = {allCells[0], allCells[1], allCells[2]};
      allCells2600[nDim - 1] = m_noGhostLayers;
      stringstream path2600Str;
      path2600Str << blockPath << "/bc2600";
      std::string path2600 = path2600Str.str();

      for(MInt var = 0; var < m_maxNoVariables; var++) {
        pio.defineArray(maia::parallel_io::PIO_FLOAT, path2600, m_pvariableNames[var], nDim, allCells2600);
      }
    }

    // ////////////////////////////////////////////////
    // ///////// Create BC2601 Information ////////////
    // ////////////////////////////////////////////////
    if(m_bc2601IsActive) {
      ParallelIo::size_type allCells2601[3] = {allCells[0], allCells[1], allCells[2]};
      allCells2601[nDim - 2] = m_noGhostLayers;
      stringstream path2601Str;
      path2601Str << blockPath << "/bc2601";
      std::string path2601 = path2601Str.str();

      for(MInt var = 0; var < m_maxNoVariables; var++) {
        pio.defineArray(maia::parallel_io::PIO_FLOAT, path2601, m_pvariableNames[var], nDim, allCells2601);
      }
    }

    ////////////////////////////////////////////////
    ///////// Create STG Information ///////////////
    ////////////////////////////////////////////////
    if(m_stgIsActive) {
      ParallelIo::size_type allCells7909[3];
      allCells7909[0] = allCells[0];
      allCells7909[1] = allCells[1];
      allCells7909[2] = 3;
      for(MInt var = 0; var < m_stgNoVariables; var++) {
        stringstream stgPath;
        stgPath << blockPathStr << "/stg";
        stringstream fieldName;
        fieldName << "stgFQ" << var;
        pio.defineArray(maia::parallel_io::PIO_FLOAT, stgPath.str(), fieldName.str(), nDim, allCells7909);
      }
    }
  }

  ////////////////////////////////////////////////
  //////// Create Sandpaper Tripping Info ////////
  ////////////////////////////////////////////////
  if(m_useSandpaperTrip) {
    stringstream tripPath;
    tripPath << "/trip";
    ParallelIo::size_type dataSize = m_tripNoTrips * 2 * m_tripNoModes;

    pio.defineArray(maia::parallel_io::PIO_FLOAT, tripPath.str(), "tripModesG", 1, &dataSize);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, tripPath.str(), "tripModesH1", 1, &dataSize);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, tripPath.str(), "tripModesH2", 1, &dataSize);
  }

  if(m_stgIsActive) {
    stgNoEddieFields = MInt(m_stgNoEddieProperties * m_stgMaxNoEddies);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, stgGlobalPathStr, "FQeddies", 1, &stgNoEddieFields);
  }

  //////////////////////////////////////////////////////////
  ///////// Write Primitive/Conservative Variables /////////
  //////////////////////////////////////////////////////////
  stringstream path;
  path << m_blockId;
  MString blockPathStr = "block";
  blockPathStr += path.str();
  // write out the data
  ParallelIo::size_type ioOffset[3] = {0, 0, 0};
  ParallelIo::size_type ioSize[3] = {0, 0, 0};
  ParallelIo::size_type ioGhost[3] = {m_noGhostLayers, m_noGhostLayers, m_noGhostLayers};
  for(MInt dim = 0; dim < nDim; ++dim) {
    ioOffset[dim] = m_nOffsetCells[dim];
    ioSize[dim] = m_nActiveCells[dim];
  }

  for(MInt v = 0; v < m_maxNoVariables; ++v) {
    pio.writeArray(&(m_cells->pvariables[v][0]), blockPathStr, m_pvariableNames[v], nDim, ioOffset, ioSize, ioGhost);
  }

  ////////////////////////////////////////////////
  ///////// Write FQ Data ////////////////////////
  ////////////////////////////////////////////////
  if(FQ->noFQVariables > 0) {
    for(MInt v = 0; v < FQ->noFQVariables; ++v) {
      if(FQ->fqWriteOutput[v]) {
        pio.writeArray(&m_cells->fq[v][0], blockPathStr, FQ->fqNames[v], nDim, ioOffset, ioSize, ioGhost);
      }
    }
  }


  ////////////////////////////////////////////////
  ///////// Write BC2600 Data ////////////////////
  ////////////////////////////////////////////////
  if(m_bc2600IsActive) {
    stringstream path2600Str;
    path2600Str << blockPathStr << "/bc2600";
    std::string path2600 = path2600Str.str();

    ParallelIo::size_type ioSize2600[3] = {0, 0, 0};
    ParallelIo::size_type ioOffset2600[3] = {0, 0, 0};
    ParallelIo::size_type ioGhost2600[3] = {m_noGhostLayers, m_noGhostLayers, m_noGhostLayers};
    for(MInt dim = 0; dim < nDim; ++dim) {
      ioSize2600[dim] = m_bc2600noActiveCells[dim];
      ioOffset2600[dim] = m_bc2600noOffsetCells[dim];
    }

    ioGhost2600[nDim - 1] = 0;
    if(m_bc2600) {
      for(MInt var = 0; var < m_maxNoVariables; var++) {
        pio.writeArray(&m_bc2600Variables[var][0], path2600, m_pvariableNames[var], nDim, ioOffset2600, ioSize2600,
                       ioGhost2600);
      }
    } else {
      MFloat empty = 0;
      for(MInt var = 0; var < m_maxNoVariables; var++) {
        pio.writeArray(&empty, path2600, m_pvariableNames[var], nDim, ioOffset2600, ioSize2600, ioGhost2600);
      }
    }
  }

  ////////////////////////////////////////////////
  ///////// Write BC2601 Data ////////////////////
  ////////////////////////////////////////////////
  if(m_bc2601IsActive) {
    stringstream path2601Str;
    path2601Str << blockPathStr << "/bc2601";
    std::string path2601 = path2601Str.str();

    ParallelIo::size_type ioSize2601[3] = {0, 0, 0};
    ParallelIo::size_type ioOffset2601[3] = {0, 0, 0};
    ParallelIo::size_type ioGhost2601[3] = {m_noGhostLayers, m_noGhostLayers, m_noGhostLayers};
    for(MInt dim = 0; dim < nDim; ++dim) {
      ioSize2601[dim] = m_bc2601noActiveCells[dim];
      ioOffset2601[dim] = m_bc2601noOffsetCells[dim];
    }

    ioGhost2601[nDim - 2] = 0;
    if(m_bc2601) {
      for(MInt var = 0; var < m_maxNoVariables; var++) {
        pio.writeArray(&m_bc2601Variables[var][0], path2601, m_pvariableNames[var], nDim, ioOffset2601, ioSize2601,
                       ioGhost2601);
      }
    } else {
      MFloat empty = 0;
      for(MInt var = 0; var < m_maxNoVariables; var++) {
        pio.writeArray(&empty, path2601, m_pvariableNames[var], nDim, ioOffset2601, ioSize2601, ioGhost2601);
      }
    }
  }

  ////////////////////////////////////////////////
  ///////// Write STG Information ////////////////
  ////////////////////////////////////////////////
  if(m_stgIsActive) {
    ParallelIo::size_type VBOffset = 0;
    if(m_stgRootRank) {
      pio.writeArray(&m_stgEddies[0][0], stgGlobalPathStr, "FQeddies", 1, &VBOffset, &stgNoEddieFields);
    } else {
      stgNoEddieFields = 0;
      MFloat empty = 0;
      pio.writeArray(&empty, stgGlobalPathStr, "FQeddies", 1, &VBOffset, &stgNoEddieFields);
    }

    // if this domain has part of the STG bc write value, otherwise only write nullptr
    if(m_stgLocal) {
      MInt noActiveStgCells = (m_stgBoxSize[0] - 2 * m_noGhostLayers) * (m_stgBoxSize[1] - 2 * m_noGhostLayers) * 3;
      MFloatScratchSpace stgFqDummy(m_stgNoVariables, noActiveStgCells, AT_, "stgFqDummy");

      for(MInt var = 0; var < m_stgNoVariables; var++) {
        for(MInt k = m_noGhostLayers; k < m_stgBoxSize[0] - m_noGhostLayers; k++) {
          for(MInt j = m_noGhostLayers; j < m_stgBoxSize[1] - m_noGhostLayers; j++) {
            for(MInt i = 0; i < m_stgBoxSize[2]; i++) {
              MInt cellIdBC = i + (j + k * m_stgBoxSize[1]) * 3;
              MInt cellIdDummy =
                  i + ((j - m_noGhostLayers) + (k - m_noGhostLayers) * (m_stgBoxSize[1] - 2 * m_noGhostLayers)) * 3;
              stgFqDummy(var, cellIdDummy) = m_cells->stg_fq[var][cellIdBC];
            }
          }
        }
      }

      ParallelIo::size_type bcOffset[3] = {m_nOffsetCells[0], m_nOffsetCells[1], 0};
      ParallelIo::size_type bcCells[3] = {m_stgBoxSize[0] - 2 * m_noGhostLayers, m_stgBoxSize[1] - 2 * m_noGhostLayers,
                                          m_stgBoxSize[2]};

      for(MInt var = 0; var < m_stgNoVariables; var++) {
        stringstream fieldName;
        stringstream stgPath;
        stgPath << blockPathStr << "/stg";
        fieldName << "stgFQ" << var;
        pio.writeArray(&stgFqDummy(var, 0), stgPath.str(), fieldName.str(), nDim, bcOffset, bcCells);
      }
    } else {
      ParallelIo::size_type bcOffset[3] = {0, 0, 0};
      ParallelIo::size_type bcCells[3] = {0, 0, 0};
      MFloat empty = 0;

      for(MInt var = 0; var < m_stgNoVariables; var++) {
        stringstream fieldName;
        stringstream stgPath;
        stgPath << blockPathStr << "/stg";
        fieldName << "stgFQ" << var;
        pio.writeArray(&empty, stgPath.str(), fieldName.str(), nDim, bcOffset, bcCells);
      }
    }
  }

  ////////////////////////////////////////////////
  ///////// Sandpaper Tripping ///////////////////
  ////////////////////////////////////////////////
  if(m_useSandpaperTrip) {
    stringstream tripPath;
    tripPath << "trip";

    if(domainId() == 0) {
      ParallelIo::size_type offset = 0;
      ParallelIo::size_type dataSize = m_tripNoTrips * m_tripNoModes * 2;

      pio.writeArray(m_tripModesG, tripPath.str(), "tripModesG", 1, &offset, &dataSize);
      pio.writeArray(m_tripModesH1, tripPath.str(), "tripModesH1", 1, &offset, &dataSize);
      pio.writeArray(m_tripModesH2, tripPath.str(), "tripModesH2", 1, &offset, &dataSize);
    } else {
      ParallelIo::size_type offset = 0;
      ParallelIo::size_type dataSize = 0;
      MFloat empty = 0;
      pio.writeArray(&empty, tripPath.str(), "tripModesG", 1, &offset, &dataSize);
      pio.writeArray(&empty, tripPath.str(), "tripModesH1", 1, &offset, &dataSize);
      pio.writeArray(&empty, tripPath.str(), "tripModesH2", 1, &offset, &dataSize);
    }
  }

  m_log << "...-> OK " << endl;
}


/**
 * \brief Saves the partitioned grid into an HDF5 file.
 *        Not used in production use but useful for debugging.
 *
 * \author: Pascal Meysonnat
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::savePartitions() {
  // Function to write the solution to file with iolibrary
  stringstream fileName;
  ParallelIo::size_type noCells[3] = {0, 0, 0};

  fileName << m_solutionOutput << "partitioned" << globalTimeStep << m_outputFormat;
  ParallelIoHdf5 pio(fileName.str(), maia::parallel_io::PIO_REPLACE, m_StructuredComm);
  writeHeaderAttributes(&pio, "solution");
  writePropertiesAsAttributes(&pio, "");

  // save with ghostcells ==> multiple solvers;
  for(MInt i = 0; i < noDomains(); i++) {
    // create datasets for the io library
    for(MInt j = 0; j < nDim; j++) {
      noCells[j] = m_grid->getActivePoints(i, j) - 1 + 2 * m_noGhostLayers;
    }
    stringstream path;
    path << i;
    MString partitionPathStr = "block";
    partitionPathStr += path.str();
    const char* partitionPath = partitionPathStr.c_str();
    // create dataset for primitive/conservative variables
    for(MInt v = 0; v < m_maxNoVariables; v++) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, partitionPath, m_pvariableNames[v], nDim, noCells);
    }
    // create dataset for fq field variables
    for(MInt v = 0; v < FQ->noFQVariables; v++) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, partitionPath, FQ->fqNames[v], nDim, noCells);
    }

    if(m_stgIsActive) {
      ParallelIo::size_type allCells7909[3];
      allCells7909[0] = m_grid->getActivePoints(i, 0) - 1 + 2 * m_noGhostLayers;
      allCells7909[1] = m_grid->getActivePoints(i, 1) - 1 + 2 * m_noGhostLayers;
      allCells7909[2] = 3;

      for(MInt var = 0; var < m_stgNoVariables; var++) {
        stringstream fieldName;
        stringstream stgPath;
        stgPath << partitionPathStr << "/stg";
        fieldName << "stgFQ" << var;
        pio.defineArray(maia::parallel_io::PIO_FLOAT, stgPath.str(), fieldName.str(), nDim, allCells7909);
      }
    }
  }

  // write the values into the array so that we can visualize it
  ParallelIo::size_type ioOffset[3] = {0, 0, 0};
  ParallelIo::size_type ioSize[3] = {m_nCells[0], m_nCells[1], m_nCells[2]};
  stringstream path;
  path << domainId();
  MString partitionPathStr = "block";
  partitionPathStr += path.str();

  // write primitive variables
  for(MInt v = 0; v < m_maxNoVariables; ++v) {
    pio.writeArray(&m_cells->pvariables[v][0], partitionPathStr, m_pvariableNames[v], nDim, ioOffset, ioSize);
  }

  for(MInt v = 0; v < FQ->noFQVariables; v++) {
    pio.writeArray(&m_cells->fq[v][0], partitionPathStr, FQ->fqNames[v], nDim, ioOffset, ioSize);
  }

  ////////////////////////////////////////////////
  ///////// Write STG Information ////////////////
  ////////////////////////////////////////////////
  if(m_stgIsActive) {
    ParallelIo::size_type bcOffset[3] = {0, 0, 0};
    ParallelIo::size_type bcCells[3] = {m_nCells[0], m_nCells[1], 3};

    for(MInt var = 0; var < m_stgNoVariables; var++) {
      stringstream fieldName;
      stringstream stgPath;
      stgPath << partitionPathStr << "/stg";
      fieldName << "stgFQ" << var;
      pio.writeArray(&m_cells->stg_fq[var][0], stgPath.str(), fieldName.str(), nDim, bcOffset, bcCells);
    }
  }
}

/**
 * \brief Load the restart time step from the restart file
 *         (useNonSpecifiedRestartFile enabled)
 */
template <MInt nDim>
MInt FvStructuredSolver<nDim>::determineRestartTimeStep() const {
  TRACE();

  if(!m_useNonSpecifiedRestartFile) {
    TERMM(1, "determineRestartTimeStep should only be used with useNonSpecifiedRestartFile enabled!");
  }

  MString restartFile = Context::getSolverProperty<MString>("restartVariablesFileName", m_solverId, AT_);
  std::stringstream restartFileName;
  restartFileName << outputDir() << restartFile;
  ParallelIoHdf5 pio(restartFileName.str(), maia::parallel_io::PIO_READ, MPI_COMM_SELF);
  MInt timeStep = -1;
  pio.getAttribute(&timeStep, "globalTimeStep", "");

  return timeStep;
}

/**
 * \brief Load Restart File (primitive and conservative output)
 *        general formulation
 *
 * \author Pascal Meysonnat
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::loadRestartFile() {
  TRACE();
  RECORD_TIMER_START(m_timers[Timers::LoadRestart]);
  m_log << "loading Restart file ... " << endl;
  stringstream restartFileName;

  if(m_useNonSpecifiedRestartFile) {
    MString restartFile = "restart.hdf5";
    /*! \property
      \page propertiesFVSTRCTRD
      \section restartVariableFileName
      <code>MInt FvStructuredSolver::loadRestartFile </code>\n
      default = <code> 0 </code>\n \n
      Name of the specific restart file.\n
      Possible values are:\n
      <ul>
      <li>string</li>
      </ul>
      Keywords: <i>RESTART, STRUCTURED</i>
    */
    restartFile =
        Context::getSolverProperty<MString>("restartVariablesFileName", m_solverId, AT_); // this should be removed
    restartFileName << outputDir() << restartFile;
  } else {
    MString restartFile = "restart";
    /*! \property
      \page propertiesFVSTRCTRD
      \section restartTimeStep
      <code>MInt FvStructuredSolver::loadRestartFile </code>\n
      default = <code> 0 </code>\n \n
      Start Iteration of the specific restart file.\n
      Possible values are:\n
      <ul>
      <li>integer</li>
      </ul>
      Keywords: <i>RESTART, STRUCTURED</i>
    */

    MInt restartTimeStep = Context::getSolverProperty<MInt>("restartTimeStep", m_solverId, AT_);

    restartFileName << outputDir() << restartFile << restartTimeStep << ".hdf5";
  }

  MBool restartFromSA = false;
  restartFromSA = Context::getSolverProperty<MBool>("restartFromSA", m_solverId, AT_, &restartFromSA);
  if(restartFromSA && m_ransMethod != RANS_KEPSILON) mTerm(1, "Are you sick!");

  ParallelIoHdf5 pio(restartFileName.str(), maia::parallel_io::PIO_READ, m_StructuredComm);

  // check if restart and grid do fit together through UID
  if(!m_ignoreUID) {
    MString aUID = "";
    pio.getAttribute(&aUID, "UID", "");

    if(aUID.compare(m_grid->m_uID) != 0) {
      mTerm(1, AT_, "FATAL: the files do not match each other according to the attribute UID");
    }
  }
  // check general attributes
  pio.getAttribute(&m_time, "time", "");
  pio.getAttribute(&m_physicalTime, "physicalTime", "");
  pio.getAttribute(&m_physicalTimeStep, "physicalTimeStep", "");
  pio.getAttribute(&m_firstMaxResidual, "firstMaxResidual", "");
  pio.getAttribute(&m_firstAvrgResidual, "firstAvrgResidual", "");
  pio.getAttribute(&m_timeStep, "timeStep", "");
  // check if primitive or conservative output is given
  MInt isPrimitiveOutput = 1;
  if(pio.hasAttribute("primitiveOutput", "")) {
    pio.getAttribute(&isPrimitiveOutput, "primitiveOutput", "");
  }
  // if moving Grid is actived read the moving grid time offset (if it exists)
  // otherwise assume this is an initial start and set the current restart time
  // as the moving grid time offset
  if(m_movingGrid || m_bodyForce) {
    if(pio.hasAttribute("movingGridStepOffset", "")) {
      pio.getAttribute(&m_movingGridTimeOffset, "movingGridTimeOffset", "");
      pio.getAttribute(&m_movingGridStepOffset, "movingGridStepOffset", "");
      m_movingGridInitialStart = false;
    } else {
      m_movingGridTimeOffset = m_time;
      m_movingGridStepOffset = globalTimeStep;
      m_movingGridInitialStart = true;
    }
    // check if the wave time step has already been computed
    if(pio.hasAttribute("waveTimeStepComputed", "")) {
      pio.getAttribute(&m_waveTimeStepComputed, "waveTimeStepComputed", "");
    } else {
      m_waveTimeStepComputed = false;
    }

    if(pio.hasAttribute("waveNoStepsPerCell", "")) {
      pio.getAttribute(&m_waveNoStepsPerCell, "waveNoStepsPerCell", "");
    } else {
      m_waveNoStepsPerCell = 1;
    }
  }

  // check for convective unit output
  if(m_useConvectiveUnitWrite) {
    m_noConvectiveOutputs = (MInt)(m_physicalTime / m_convectiveUnitInterval);
    m_log << "Convective unit output iteration counter: " << m_noConvectiveOutputs << endl;
  }
  // check for moving grid initial start
  if(m_movingGridInitialStart) {
    m_movingGridTimeOffset = m_time;
  }
  if(domainId() == 0) {
    cout << "Restarting at GlobalTimeStep " << globalTimeStep << endl;
  }
  m_restartTimeStep = globalTimeStep;
  // now read in the data!
  m_log << "-> reading in the data ... " << endl;
  m_log << "Loading restart variables..." << endl;
  if(domainId() == 0) {
    cout << "Loading restart variables..." << endl;
  }
  stringstream blockNumber;
  blockNumber << m_blockId;
  MString blockPathStr = "/block";
  blockPathStr += blockNumber.str();
  const char* blockPath = blockPathStr.c_str();

  // Check if restart file has correct data for restart from SA
  if(restartFromSA) {
    vector<MString> variableNames = pio.getDatasetNames(blockPath);

    MBool rans0 = false;
    for(MUint i = 0; i < variableNames.size(); ++i) {
      if(variableNames[i].find("rans1") != std::string::npos) {
        cout << variableNames[i] << endl;
        mTerm(1, "Restart file contains the variable rans1, but restartFromSA was set!!!");
      }
      if(variableNames[i].find("rans0") != std::string::npos) rans0 = true;
    }
    if(!rans0) mTerm(1, "Restart file does not contain variable 'rans0'!");
  }

  // record tim to load restart file members
  RECORD_TIMER_START(m_timers[Timers::LoadVariables]);
  // check for primitive input or conservative input
  ParallelIo::size_type ioOffset[3] = {0, 0, 0};
  ParallelIo::size_type ioSize[3] = {0, 0, 0};
  for(MInt dim = 0; dim < nDim; ++dim) {
    ioOffset[dim] = m_nOffsetCells[dim];
    ioSize[dim] = m_nActiveCells[dim];
  }

  if(isPrimitiveOutput) {
    for(MInt var = 0; var < m_maxNoVariables - restartFromSA; var++) {
      m_cells->pvariables[var][0] = -1.0;
      pio.readArray(m_cells->pvariables[var], blockPath, m_pvariableNames[var], nDim, ioOffset, ioSize);
      m_log << "Reading " << m_pvariableNames[var] << endl;
    }
  } else {
    for(MInt var = 0; var < CV->noVariables - restartFromSA; var++) {
      pio.readArray(m_cells->variables[var], blockPath, m_variableNames[var], nDim, ioOffset, ioSize);
    }
  }
  if(m_zonal) {
    pio.readArray(m_cells->fq[FQ->AVG_U], blockPath, FQ->fqNames[FQ->AVG_U], nDim, ioOffset, ioSize);
    pio.readArray(m_cells->fq[FQ->AVG_V], blockPath, FQ->fqNames[FQ->AVG_V], nDim, ioOffset, ioSize);
    pio.readArray(m_cells->fq[FQ->AVG_W], blockPath, FQ->fqNames[FQ->AVG_W], nDim, ioOffset, ioSize);
    pio.readArray(m_cells->fq[FQ->AVG_RHO], blockPath, FQ->fqNames[FQ->AVG_RHO], nDim, ioOffset, ioSize);
    pio.readArray(m_cells->fq[FQ->AVG_P], blockPath, FQ->fqNames[FQ->AVG_P], nDim, ioOffset, ioSize);
    pio.readArray(m_cells->fq[FQ->NU_T], blockPath, FQ->fqNames[FQ->NU_T], nDim, ioOffset, ioSize);

    FQ->loadedFromRestartFile[FQ->AVG_U] = true;
    FQ->loadedFromRestartFile[FQ->AVG_V] = true;
    FQ->loadedFromRestartFile[FQ->AVG_W] = true;
    FQ->loadedFromRestartFile[FQ->AVG_RHO] = true;
    FQ->loadedFromRestartFile[FQ->AVG_P] = true;
    FQ->loadedFromRestartFile[FQ->NU_T] = true;
  }

  RECORD_TIMER_STOP(m_timers[Timers::LoadVariables]);
  m_log << "Loading restart variables... SUCCESSFUL!" << endl;
  if(domainId() == 0) {
    cout << "Loading restart variables... SUCCESSFUL!" << endl;
  }
  m_log << "-> reading in auxilliary data for restart ..." << endl;
  m_log << "--> ... sponge ..." << endl;
  ////////////////////////////////////////////////////////////
  ///////////////// SPONGE ///////////////////////////////////
  ////////////////////////////////////////////////////////////

  RECORD_TIMER_START(m_timers[Timers::LoadSponge]);
  if(m_useSponge) {
    m_log << "--> ... sponge ..." << endl;
    if(domainId() == 0) {
      cout << "Loading sponge data..." << endl;
    }
    if(m_computeSpongeFactor == false) {
      pio.readArray(m_cells->fq[FQ->SPONGE_FACTOR], blockPath, FQ->fqNames[FQ->SPONGE_FACTOR], nDim, ioOffset, ioSize);
      FQ->loadedFromRestartFile[FQ->SPONGE_FACTOR] = true;
    }
    if(m_spongeLayerType == 2) {
      pio.readArray(m_cells->fq[FQ->SPONGE_RHO], blockPath, FQ->fqNames[FQ->SPONGE_RHO], nDim, ioOffset, ioSize);
      FQ->loadedFromRestartFile[FQ->SPONGE_RHO] = true;
      pio.readArray(m_cells->fq[FQ->SPONGE_RHO_E], blockPath, FQ->fqNames[FQ->SPONGE_RHO_E], nDim, ioOffset, ioSize);
      FQ->loadedFromRestartFile[FQ->SPONGE_RHO_E] = true;
    }
    if(m_spongeLayerType == 4) {
      pio.readArray(m_cells->fq[FQ->SPONGE_RHO], blockPath, FQ->fqNames[FQ->SPONGE_RHO], nDim, ioOffset, ioSize);
      FQ->loadedFromRestartFile[FQ->SPONGE_RHO] = true;
    }
    if(domainId() == 0) {
      cout << "Loading sponge data... SUCCESSFUL!" << endl;
    }
    m_log << "--> ... sponge ... SUCCESSFUL" << endl;
  }
  RECORD_TIMER_STOP(m_timers[Timers::LoadSponge]);

  ////////////////////////////////////////////////////////////
  ///////////////// CELL SHIFTING ////////////////////////////
  ////////////////////////////////////////////////////////////
  if(isPrimitiveOutput) {
    this->shiftCellValuesRestart(true);
    computeConservativeVariables();
  } else {
    this->shiftCellValuesRestart(false);
    computePrimitiveVariables();
  }

  // fill the ghost-cells for the
  // averaged cells with exchange or
  // simple extrapolation
  if(m_zonal) {
    vector<MFloat*> zonalVars;
    zonalVars.push_back(m_cells->fq[FQ->AVG_U]);
    zonalVars.push_back(m_cells->fq[FQ->AVG_V]);
    zonalVars.push_back(m_cells->fq[FQ->AVG_W]);
    zonalVars.push_back(m_cells->fq[FQ->AVG_P]);
    zonalVars.push_back(m_cells->fq[FQ->AVG_RHO]);
    zonalVars.push_back(m_cells->fq[FQ->NU_T]);
    gcFillGhostCells(zonalVars);
  }

  // load special variables

  IF_CONSTEXPR(nDim == 3) { // only implemented for 3d or does not work in 2d.
    loadRestartBC2601();
    if(m_stgIsActive) {
      RECORD_TIMER_START(m_timers[Timers::LoadSTG]);
      loadRestartSTG(isPrimitiveOutput);
      RECORD_TIMER_STOP(m_timers[Timers::LoadSTG]);
    }
  }

  ////////////////////////////////////////////////////////////
  ///////////////// CHANGE MACH NUMBER ///////////////////////
  ////////////////////////////////////////////////////////////
  if(m_changeMa) {
    MFloat oldMa = F0;
    pio.getAttribute(&oldMa, "Ma", "");
    convertRestartVariables(oldMa);
    // if we use the STG also convert the vars in the STG fields
    if(m_stgIsActive) {
      convertRestartVariablesSTG(oldMa);
    }
  }

  ///////////////////////////////////////////////////////////
  //////////////// VOLUME FORCING ///////////////////////////
  ///////////////////////////////////////////////////////////
  if(m_volumeForceMethod != 0) {
    //    if(pio.hasAttribute("volumeForce", "")) {
    pio.getAttribute(&m_volumeForce[0], "volumeForce", "");
    //    }
  }

  m_log << "-> reading in auxilliary data for restart ...SUCCESSFUL" << endl;
  m_log << "loading Restart file ... SUCCESSFUL " << endl;

  loadRestartBC2600();

  RECORD_TIMER_STOP(m_timers[Timers::LoadRestart]);
}

/**
 * \brief Function to shift the values in the cell after restart
 *        to correct position (reading in does not take into account
 *        the ghost cells)
 *
 */
template <>
void FvStructuredSolver<2>::shiftCellValuesRestart(MBool isPrimitive) {
  TRACE();
  for(MInt j = (m_nActiveCells[0] - 1); j >= 0; j--) {
    for(MInt i = (m_nActiveCells[1] - 1); i >= 0; i--) {
      const MInt cellId_org = i + j * m_nActiveCells[1];
      const MInt i_new = i + m_noGhostLayers;
      const MInt j_new = j + m_noGhostLayers;
      const MInt cellId = i_new + j_new * m_nCells[1];
      if(!isPrimitive) {
        for(MInt var = 0; var < CV->noVariables; var++) {
          m_cells->variables[var][cellId] = m_cells->variables[var][cellId_org];
          m_cells->variables[var][cellId_org] = F0;
        }
      } else {
        for(MInt var = 0; var < m_maxNoVariables; var++) {
          m_cells->pvariables[var][cellId] = m_cells->pvariables[var][cellId_org];
          m_cells->pvariables[var][cellId_org] = F0;
        }
      }
      // also shift values in the FQ field
      if(FQ->noFQVariables > 0) {
        for(MInt var = 0; var < FQ->noFQVariables; var++) {
          if(FQ->loadedFromRestartFile[var]) {
            m_cells->fq[var][cellId] = m_cells->fq[var][cellId_org];
            m_cells->fq[var][cellId_org] = F0;
          }
        }
      }
    }
  }
}

template <>
void FvStructuredSolver<3>::shiftCellValuesRestart(MBool isPrimitive) {
  TRACE();
  // accounting for the ghost layers and shift the values to the right place
  for(MInt k = (m_nActiveCells[0] - 1); k >= 0; k--) {
    for(MInt j = (m_nActiveCells[1] - 1); j >= 0; j--) {
      for(MInt i = (m_nActiveCells[2] - 1); i >= 0; i--) {
        const MInt cellId_org = i + (j + k * m_nActiveCells[1]) * m_nActiveCells[2];
        const MInt i_new = i + m_noGhostLayers;
        const MInt j_new = j + m_noGhostLayers;
        const MInt k_new = k + m_noGhostLayers;
        const MInt cellId = i_new + (j_new + k_new * m_nCells[1]) * m_nCells[2];
        if(!isPrimitive) {
          for(MInt var = 0; var < CV->noVariables; var++) {
            m_cells->variables[var][cellId] = m_cells->variables[var][cellId_org];
            m_cells->variables[var][cellId_org] = F0;
          }
        } else {
          for(MInt var = 0; var < m_maxNoVariables; var++) {
            m_cells->pvariables[var][cellId] = m_cells->pvariables[var][cellId_org];
            m_cells->pvariables[var][cellId_org] = F0;
          }
        }
        // also shift values in the FQ field
        if(FQ->noFQVariables > 0) {
          for(MInt var = 0; var < FQ->noFQVariables; var++) {
            if(FQ->loadedFromRestartFile[var]) {
              m_cells->fq[var][cellId] = m_cells->fq[var][cellId_org];
              m_cells->fq[var][cellId_org] = F0;
            }
          }
        }
      }
    }
  }
}

/**
 * \brief Load Box file
 *         general formulation
 *
 * \author Marian Albers
 * \date: 02.10.2019
 */
template <MInt nDim>
MBool FvStructuredSolver<nDim>::loadBoxFile(MString fileDir, MString filePrefix, MInt currentStep, MInt boxNr) {
  TRACE();

  stringstream fileName;
  fileName << fileDir << "/" << filePrefix << currentStep << ".hdf5";
  if(FILE* file = fopen((fileName.str()).c_str(), "r")) {
    fclose(file);
    if(domainId() == 0) {
      cout << "Loading box file " << fileName.str() << endl;
    }
  } else {
    if(domainId() == 0) {
      cout << "Box file " << fileName.str() << " does not exist" << endl;
    }
    return false;
  }


  ParallelIoHdf5 pio(fileName.str(), maia::parallel_io::PIO_READ, m_StructuredComm);
  stringstream boxPath;
  boxPath << "t" << currentStep << "/box" << boxNr;

  MInt boxOffset[3] = {0, 0, 0};
  pio.getAttribute(&boxOffset[0], "offsetk", boxPath.str());
  pio.getAttribute(&boxOffset[1], "offsetj", boxPath.str());
  pio.getAttribute(&boxOffset[2], "offseti", boxPath.str());
  MInt boxSize[3] = {0, 0, 0};
  pio.getAttribute(&boxSize[0], "sizek", boxPath.str());
  pio.getAttribute(&boxSize[1], "sizej", boxPath.str());
  pio.getAttribute(&boxSize[2], "sizei", boxPath.str());

  ParallelIo::size_type localBoxSize[3] = {0, 0, 0};
  ParallelIo::size_type localBoxOffset[3] = {0, 0, 0};
  ParallelIo::size_type localDomainBoxOffset[3] = {0, 0, 0};

  if(((m_nOffsetCells[2] <= boxOffset[2] && boxOffset[2] < m_nOffsetCells[2] + m_nActiveCells[2])
      || (boxOffset[2] <= m_nOffsetCells[2] && m_nOffsetCells[2] < boxOffset[2] + boxSize[2]))
     && ((m_nOffsetCells[1] <= boxOffset[1] && boxOffset[1] < m_nOffsetCells[1] + m_nActiveCells[1])
         || (boxOffset[1] <= m_nOffsetCells[1] && m_nOffsetCells[1] < boxOffset[1] + boxSize[1]))
     && ((m_nOffsetCells[0] <= boxOffset[0] && boxOffset[0] < m_nOffsetCells[0] + m_nActiveCells[0])
         || (boxOffset[0] <= m_nOffsetCells[0]
             && m_nOffsetCells[0] < boxOffset[0] + boxSize[0]))) { // the box is contained

    // get the size of the box
    for(MInt dim = 0; dim < nDim; ++dim) {
      if(m_nOffsetCells[dim] <= boxOffset[dim]
         && boxOffset[dim] + boxSize[dim] < m_nOffsetCells[dim] + m_nActiveCells[dim]) {
        localBoxSize[dim] = boxSize[dim];
        localBoxOffset[dim] = 0;
        localDomainBoxOffset[dim] = boxOffset[dim] - m_nOffsetCells[dim];
      } else if(m_nOffsetCells[dim] <= boxOffset[dim]) {
        localBoxSize[dim] = (m_nOffsetCells[dim] + m_nActiveCells[dim]) - boxOffset[dim];
        localBoxOffset[dim] = 0;
        localDomainBoxOffset[dim] = boxOffset[dim] - m_nOffsetCells[dim];
      } else if(boxOffset[dim] <= m_nOffsetCells[dim]
                && m_nOffsetCells[dim] + m_nActiveCells[dim] < boxOffset[dim] + boxSize[dim]) {
        localBoxSize[dim] = m_nActiveCells[dim];
        localBoxOffset[dim] = m_nOffsetCells[dim] - boxOffset[dim];
        localDomainBoxOffset[dim] = 0;
      } else {
        localBoxSize[dim] = (boxOffset[dim] + boxSize[dim]) - m_nOffsetCells[dim];
        localBoxOffset[dim] = m_nOffsetCells[dim] - boxOffset[dim];
        localDomainBoxOffset[dim] = 0;
      }
    }

    MInt totalLocalSize = 1;
    for(MInt dim = 0; dim < nDim; ++dim) {
      totalLocalSize *= localBoxSize[dim];
    }

    MFloatScratchSpace localBoxVar(totalLocalSize, AT_, "local Box Variables");
    for(MInt var = 0; var < PV->noVariables; var++) {
      pio.readArray(&localBoxVar[0], boxPath.str(), m_pvariableNames[var], nDim, localBoxOffset, localBoxSize);

      for(MInt k = m_noGhostLayers + localDomainBoxOffset[0];
          k < m_noGhostLayers + localDomainBoxOffset[0] + localBoxSize[0];
          ++k) {
        for(MInt j = m_noGhostLayers + localDomainBoxOffset[1];
            j < m_noGhostLayers + localDomainBoxOffset[1] + localBoxSize[1];
            ++j) {
          for(MInt i = m_noGhostLayers + localDomainBoxOffset[2];
              i < m_noGhostLayers + localDomainBoxOffset[2] + localBoxSize[2];
              ++i) {
            const MInt cellId = i + (j + k * m_nCells[1]) * m_nCells[2];
            const MInt boxI = i - m_noGhostLayers - localDomainBoxOffset[2];
            const MInt boxJ = j - m_noGhostLayers - localDomainBoxOffset[1];
            const MInt boxK = k - m_noGhostLayers - localDomainBoxOffset[0];
            const MInt localId = boxI + (boxJ + boxK * localBoxSize[1]) * localBoxSize[2];
            m_cells->pvariables[var][cellId] = localBoxVar[localId];
          }
        }
      }
    }
  } else {
    for(MInt var = 0; var < PV->noVariables; var++) {
      ParallelIo::size_type offset[3] = {0, 0, 0};
      ParallelIo::size_type size[3] = {0, 0, 0};
      MFloat empty = 0;
      pio.readArray(&empty, boxPath.str(), m_pvariableNames[var], nDim, offset, size);
    }
  }


  if(domainId() == 0) {
    cout << "Done loading box file " << currentStep << endl;
  }

  return true;
}


template <MInt nDim>
void FvStructuredSolver<nDim>::saveAuxData() {
  computeAuxData();
  stringstream fileName;
  MChar gridFile[25];
  MString tempG;

  fileName << m_auxOutputDir << "auxData" << m_outputIterationNumber << m_outputFormat;

  strcpy(gridFile, tempG.c_str());
  ParallelIoHdf5 pio(fileName.str(), maia::parallel_io::PIO_REPLACE, m_StructuredComm);
  writeHeaderAttributes(&pio, "auxdata");
  writePropertiesAsAttributes(&pio, "");
  const MString powerNamesVisc[3] = {"Pxv", "Pyv", "Pzv"};
  const MString powerNamesPres[3] = {"Pxp", "Pyp", "Pzp"};

  const MString dataNames3D[] = {"cfx", "cfy", "cfz", "ax", "ay", "az", "x", "y", "z"};
  MString dataNames[3 * nDim];
  MInt cnt = 0;
  for(MInt i = 0; i < 3; ++i) {
    for(MInt dim = 0; dim < nDim; ++dim)
      dataNames[cnt++] = dataNames3D[i * 3 + dim];
  }
  MInt noFields = nDim;
  if(m_detailAuxData) {
    noFields = 3 * nDim;
  }

  for(auto it = m_windowInfo->m_auxDataWindowIds.cbegin(); it != m_windowInfo->m_auxDataWindowIds.cend(); ++it) {
    const MInt i = it->first;
    ParallelIo::size_type datasetSize[nDim - 1];
    MInt dim1 = nDim - 2;
    for(MInt dim = 0; dim < nDim; ++dim) {
      if(m_windowInfo->globalStructuredBndryCndMaps[i]->end2[dim]
         == m_windowInfo->globalStructuredBndryCndMaps[i]->start2[dim]) {
        continue;
      }
      datasetSize[dim1--] = m_windowInfo->globalStructuredBndryCndMaps[i]->end2[dim]
                            - m_windowInfo->globalStructuredBndryCndMaps[i]->start2[dim];
    }

    if(m_bForce) {
      stringstream datasetId;
      datasetId << it->second; //==m_windowInfo->globalStructuredBndryCndMaps[i]->Id2;
      MString pathName = "window";
      pathName += datasetId.str();
      MString dataNamesCp = "cp";
      for(MInt j = 0; j < noFields; j++) {
        pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName, dataNames[j], nDim - 1, datasetSize);
      }
      pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName, dataNamesCp, nDim - 1, datasetSize);
    }

    if(m_bPower) {
      stringstream datasetId;
      datasetId << it->second; //==m_windowInfo->globalStructuredBndryCndMaps[i]->Id2;
      MString pathName = "window";
      pathName += datasetId.str();
      for(MInt j = 0; j < nDim; j++) {
        pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName, powerNamesVisc[j], nDim - 1, datasetSize);
        pio.defineArray(maia::parallel_io::PIO_FLOAT, pathName, powerNamesPres[j], nDim - 1, datasetSize);
      }
    }
  }

  for(auto it = m_windowInfo->m_auxDataWindowIds.cbegin(); it != m_windowInfo->m_auxDataWindowIds.cend(); ++it) {
    stringstream datasetname;
    datasetname << it->second; //==m_windowInfo->globalStructuredBndryCndMaps[i]->Id2;

    MBool isLocalAuxMap = false;
    MInt localAuxMapId = 0;
    const MUint noAuxDataMaps = m_windowInfo->physicalAuxDataMap.size();

    // loop through all local auxDataMaps
    // to find if this partition shares a part of it
    // and then save the local id
    for(MUint j = 0; j < noAuxDataMaps; ++j) {
      stringstream datasetId;
      datasetId << m_windowInfo->physicalAuxDataMap[j]->Id2;
      if(datasetId.str() == datasetname.str()) {
        isLocalAuxMap = true;
        localAuxMapId = j;
        break;
      }
    }

    if(isLocalAuxMap) {
      stringstream datasetId;
      datasetId << m_windowInfo->physicalAuxDataMap[localAuxMapId]->Id2;

      ParallelIo::size_type offset[nDim - 1] = {};
      ParallelIo::size_type dataSize[nDim - 1] = {};
      MInt dataSetSize = 1;
      MInt dim1 = nDim - 2;
      for(MInt j = 0; j < nDim; ++j) {
        if(m_windowInfo->physicalAuxDataMap[localAuxMapId]->start2[j]
           == m_windowInfo->physicalAuxDataMap[localAuxMapId]->end2[j])
          continue;
        dataSize[dim1] = m_windowInfo->physicalAuxDataMap[localAuxMapId]->end2[j]
                         - m_windowInfo->physicalAuxDataMap[localAuxMapId]->start2[j];
        offset[dim1] = m_windowInfo->physicalAuxDataMap[localAuxMapId]->start2[j];
        dataSetSize *= dataSize[dim1];
        dim1--;
      }

      if(m_bForce) {
        const MInt mapOffset = m_cells->cfOffsets[localAuxMapId];
        MString pathName = "window" + datasetId.str();
        for(MInt j = 0; j < noFields; j++) {
          pio.writeArray(&m_cells->cf[mapOffset + dataSetSize * j], pathName, dataNames[j], nDim - 1, offset, dataSize);
        }
        const MInt mapOffsetCp = m_cells->cpOffsets[localAuxMapId];
        MString dataname = "cp";
        pio.writeArray(&m_cells->cp[mapOffsetCp], pathName, dataname, nDim - 1, offset, dataSize);
      }

      if(m_bPower) {
        const MInt mapOffset = m_cells->powerOffsets[localAuxMapId];
        MString pathName = "window" + datasetId.str();
        for(MInt j = 0; j < nDim; j++) {
          pio.writeArray(&m_cells->powerVisc[mapOffset + dataSetSize * j], pathName, powerNamesVisc[j], nDim - 1,
                         offset, dataSize);
          pio.writeArray(&m_cells->powerPres[mapOffset + dataSetSize * j], pathName, powerNamesPres[j], nDim - 1,
                         offset, dataSize);
        }
      }
    } else {
      ParallelIo::size_type offset[nDim] = {};
      ParallelIo::size_type dataSize[nDim] = {};
      for(MInt j = 0; j < nDim - 1; ++j) {
        offset[j] = 0;
        dataSize[j] = 0;
      }
      // skin-friction and pressure coefficient
      if(m_bForce) {
        MString pathName = "window" + datasetname.str();
        MFloat empty = 0;
        for(MInt j = 0; j < noFields; j++) {
          pio.writeArray(&empty, pathName, dataNames[j], nDim - 1, offset, dataSize);
        }
        MString dataname = "cp";
        pio.writeArray(&empty, pathName, dataname, nDim - 1, offset, dataSize);
      }
      // power
      if(m_bPower) {
        MString pathName = "window" + datasetname.str();
        MFloat empty = 0;
        for(MInt j = 0; j < nDim; j++) {
          pio.writeArray(&empty, pathName, powerNamesVisc[j], nDim - 1, offset, dataSize);
          pio.writeArray(&empty, pathName, powerNamesPres[j], nDim - 1, offset, dataSize);
        }
      }
    }
  }

  if(m_bForce) {
    saveForceCoefficient(&pio);
  }
}


/**
 * \brief Saves force coefficients to an HDF5 file
 *
 * \authors Marian Albers, Pascal Meysonnat
 *
 * \param[in] Open ParallelIOHdf5 object to which attributes should be written
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::saveForceCoefficient(ParallelIoHdf5* pio) {
  TRACE();

  MInt noWalls = m_windowInfo->m_auxDataWindowIds.size();
  MFloatScratchSpace force(noWalls, nDim, AT_, "force");
  MFloatScratchSpace forceP(noWalls, nDim, AT_, "forceP");
  MFloatScratchSpace forceC(noWalls, nDim, AT_, "forceC");
  MFloatScratchSpace forceV(noWalls, nDim, AT_, "forceV");
  MFloatScratchSpace area(noWalls, nDim, AT_, "area");
  MFloatScratchSpace power(noWalls, nDim, AT_, "Ptot");
  MFloatScratchSpace powerV(noWalls, nDim, AT_, "Pvisc");
  MFloatScratchSpace powerP(noWalls, nDim, AT_, "Ppres");

  for(MInt i = 0; i < noWalls; ++i) {
    for(MInt dim = 0; dim < nDim; dim++) {
      forceV(i, dim) = m_forceCoef[i * m_noForceDataFields + dim];
      forceC(i, dim) = m_forceCoef[i * m_noForceDataFields + nDim + dim];
      forceP(i, dim) = m_forceCoef[i * m_noForceDataFields + 2 * nDim + dim];
      force(i, dim) = forceV(i, dim) + forceC(i, dim) + forceP(i, dim);
      powerV(i, dim) = m_forceCoef[i * m_noForceDataFields + 3 * nDim + 1 + dim];
      powerP(i, dim) = m_forceCoef[i * m_noForceDataFields + 4 * nDim + 1 + dim];
      power(i, dim) = powerV(i, dim) + powerP(i, dim);
    }
    area[i] = m_forceCoef[i * m_noForceDataFields + 3 * nDim];
  }

  MInt count = 0;
  for(auto it = m_windowInfo->m_auxDataWindowIds.cbegin(); it != m_windowInfo->m_auxDataWindowIds.cend(); ++it) {
    stringstream datasetname;
    datasetname << it->second; //==m_windowInfo->globalStructuredBndryCndMaps[it->first]->Id2;
    MString pathName = "window" + datasetname.str();

    pio->setAttribute(force(count, 0), "forceX", pathName);
    pio->setAttribute(forceV(count, 0), "forceVX", pathName);
    pio->setAttribute(forceP(count, 0), "forcePX", pathName);
    pio->setAttribute(forceC(count, 0), "forceCX", pathName);

    pio->setAttribute(force(count, 1), "forceY", pathName);
    pio->setAttribute(forceV(count, 1), "forceVY", pathName);
    pio->setAttribute(forceP(count, 1), "forcePY", pathName);
    pio->setAttribute(forceC(count, 1), "forceCY", pathName);

    IF_CONSTEXPR(nDim == 3) {
      pio->setAttribute(force(count, 2), "forceZ", pathName);
      pio->setAttribute(forceV(count, 2), "forceVZ", pathName);
      pio->setAttribute(forceP(count, 2), "forcePZ", pathName);
      pio->setAttribute(forceC(count, 2), "forceCZ", pathName);
    }

    pio->setAttribute(area[count], "area", pathName);

    if(m_bPower) {
      pio->setAttribute(power(count, 0), "powerX", pathName);
      pio->setAttribute(power(count, 1), "powerY", pathName);
      pio->setAttribute(power(count, 2), "powerZ", pathName);

      pio->setAttribute(powerP(count, 0), "powerPX", pathName);
      pio->setAttribute(powerP(count, 1), "powerPY", pathName);
      pio->setAttribute(powerP(count, 2), "powerPZ", pathName);

      pio->setAttribute(powerV(count, 0), "powerVX", pathName);
      pio->setAttribute(powerV(count, 1), "powerVY", pathName);
      pio->setAttribute(powerV(count, 2), "powerVZ", pathName);
    }

    count++;
  }
}


/**
 * \brief Function to save the force coefficients and power to an ASCII file
 *
 * \author Marian Albers
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::saveForcesToAsciiFile(MBool forceWrite) {
  TRACE();
  // compute the data
  if(m_bForce) {
    computeAuxDataRoot();

    // write all the data to files if domainId()==0
    if(domainId() == 0) {
      MInt noWalls = m_windowInfo->m_auxDataWindowIds.size();

      MFloatScratchSpace cForce(noWalls, nDim, AT_, "cForce");
      MFloatScratchSpace cForceP(noWalls, nDim, AT_, "cForceP");
      MFloatScratchSpace cForceC(noWalls, nDim, AT_, "cForceC");
      MFloatScratchSpace cForceV(noWalls, nDim, AT_, "cForceV");
      MFloatScratchSpace area(noWalls, AT_, "area");
      // power consumption
      MFloatScratchSpace cPower(noWalls, nDim, AT_, "cPower");
      MFloatScratchSpace cPowerV(noWalls, nDim, AT_, "cPower");
      MFloatScratchSpace cPowerP(noWalls, nDim, AT_, "cPower");

      for(MInt i = 0; i < noWalls; ++i) {
        for(MInt dim = 0; dim < nDim; dim++) {
          cForceV(i, dim) = m_forceCoef[i * m_noForceDataFields + dim];
          cForceC(i, dim) = m_forceCoef[i * m_noForceDataFields + nDim + dim];
          cForceP(i, dim) = m_forceCoef[i * m_noForceDataFields + 2 * nDim + dim];
          cForce(i, dim) = cForceV(i, dim) + cForceP(i, dim) + cForceC(i, dim);
          cPowerV(i, dim) = m_forceCoef[i * m_noForceDataFields + 3 * nDim + 1 + dim];
          cPowerP(i, dim) = m_forceCoef[i * m_noForceDataFields + 4 * nDim + 1 + dim];
          cPower(i, dim) = cPowerV(i, dim) + cPowerP(i, dim);
        }
        area[i] = m_forceCoef[i * m_noForceDataFields + 3 * nDim];
      }

      if(m_lastForceComputationTimeStep != globalTimeStep) {
        for(MInt i = 0; i < noWalls; ++i) {
          MInt cnt = 0;
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = globalTimeStep;
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = m_time;
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = m_physicalTime;
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForce(i, 0);
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceV(i, 0);
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceP(i, 0);
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceC(i, 0);
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForce(i, 1);
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceV(i, 1);
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceP(i, 1);
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceC(i, 1);
          IF_CONSTEXPR(nDim == 3) {
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForce(i, 2);
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceV(i, 2);
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceP(i, 2);
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cForceC(i, 2);
          }
          m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = area[i];

          if(m_bPower) {
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPower(i, 0);
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPowerV(i, 0);
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPowerP(i, 0);
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPower(i, 1);
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPowerV(i, 1);
            m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPowerP(i, 1);
            IF_CONSTEXPR(nDim == 3) {
              m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPower(i, 2);
              m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPowerV(i, 2);
              m_forceData[m_forceCounter][i * m_noForceDataFields + cnt++] = cPowerP(i, 2);
            }
          }
        }
        m_forceCounter++;
        m_lastForceComputationTimeStep = globalTimeStep;
      }

      if((m_forceCounter >= m_forceAsciiOutputInterval || forceWrite) && m_lastForceOutputTimeStep != globalTimeStep) {
        cout << "globalTimeStep: " << globalTimeStep << " writing out to ascii file" << endl;
        // TODO_SS labels:FV why not looping over the elements of m_windowInfo->m_auxDataWindowIds and choosing windowId
        // for iWall
        for(MInt i = 0; i < noWalls; ++i) {
          stringstream iWall;
          iWall << i;
          MString filename = "./forces." + iWall.str() + ".dat";

          FILE* f_forces;
          f_forces = fopen(filename.c_str(), "a+");

          for(MInt j = 0; j < m_forceCounter; j++) {
            fprintf(f_forces, "%d ", (MInt)m_forceData[j][i * m_noForceDataFields + 0]);
            for(MInt k = 1; k < m_noForceDataFields; k++) {
              fprintf(f_forces, " %.8f ", m_forceData[j][i * m_noForceDataFields + k]);
            }
            fprintf(f_forces, "\n");
          }
          fclose(f_forces);
        }

        m_forceCounter = 0;
        m_lastForceOutputTimeStep = globalTimeStep;
      }
    }
  }
}


/**
 * \brief Saves the averaged (mean) variables from postprocessing to an HDF5 file
 *
 * \author Marian Albers, Nov 15, 2015
 * \date 15.11.2015
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::saveAveragedVariables(MString name, MInt noVars, MFloat** summedVars) {
  // Function to write the solution to file with iolibrary
  stringstream fileName;

  m_log << "writing Averaged Variables to file " << name << m_outputFormat << " ... " << endl;
  fileName << name << m_outputFormat;

  ParallelIoHdf5 pio(fileName.str(), maia::parallel_io::PIO_REPLACE, m_StructuredComm);

  writeHeaderAttributes(&pio, "solution");
  writePropertiesAsAttributes(&pio, "");

  pio.setAttribute(m_averageStartTimestep, "averageStartTimeStep", "");
  pio.setAttribute(m_averageInterval, "averageSampleInterval", "");
  pio.setAttribute(m_noSamples, "noSamples", "");

  for(MInt i = 0; i < m_noBlocks; i++) {
    ParallelIo::size_type noCells[nDim] = {};
    for(MInt j = 0; j < nDim; j++) {
      noCells[j] = m_grid->getBlockNoCells(i, j);
    }
    // create datasets for the io library
    stringstream path;
    path << i;
    MString blockPathStr = "block";
    blockPathStr += path.str();

    // create dataset and write
    for(MInt var = 0; var < noVars; var++) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, m_avgVariableNames[var], nDim, noCells);
    }

    if(m_averagingFavre) {
      for(MInt var = 0; var < PV->noVariables; var++) {
        pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, m_avgFavreNames[var], nDim, noCells);
      }
    }
  }

  ParallelIo::size_type ioOffset[3] = {0, 0, 0};
  ParallelIo::size_type ioSize[3] = {0, 0, 0};
  ParallelIo::size_type ioGhost[3] = {m_noGhostLayers, m_noGhostLayers, m_noGhostLayers};
  for(MInt dim = 0; dim < nDim; ++dim) {
    ioOffset[dim] = m_nOffsetCells[dim];
    ioSize[dim] = m_nActiveCells[dim];
  }

  stringstream path;
  path << m_blockId;
  MString blockPathStr = "block";
  blockPathStr += path.str();
  for(MInt var = 0; var < noVars; var++) {
    pio.writeArray(&summedVars[var][0], blockPathStr, m_avgVariableNames[var], nDim, ioOffset, ioSize, ioGhost);
  }

  if(m_averagingFavre) {
    for(MInt var = 0; var < PV->noVariables; var++) {
      pio.writeArray(&m_favre[var][0], blockPathStr, m_avgFavreNames[var], nDim, ioOffset, ioSize, ioGhost);
    }
  }
}

/**
 * \brief Writes an restart file for postprocessing
 * \author Frederik Temme
 * \date 12.01.2015
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::saveAverageRestart(MString name, MInt noVars, MFloat** summedVars, MFloat** square,
                                                  MFloat** cube, MFloat** fourth) {
  (void)noVars;

  // Function to write the solution to file with iolibrary
  stringstream fileName;

  m_log << "writing Averaged Restart Variables to file " << name << m_outputFormat << " ... " << endl;
  fileName << name << m_outputFormat;

  ParallelIoHdf5 pio(fileName.str(), maia::parallel_io::PIO_REPLACE, m_StructuredComm);

  writeHeaderAttributes(&pio, "solution");
  writePropertiesAsAttributes(&pio, "");

  pio.setAttribute(m_averageStartTimestep, "averageStartTimeStep", "");
  pio.setAttribute(m_averageInterval, "averageSampleInterval", "");
  pio.setAttribute(m_noSamples, "noSamples", "");

  ParallelIo::size_type allCells[3]; // not nDim because of compiler warning0
  for(MInt i = 0; i < m_noBlocks; i++) {
    for(MInt j = 0; j < nDim; j++) {
      allCells[j] = m_grid->getBlockNoCells(i, j);
    }
    // create datasets for the io library
    stringstream path;
    path << i; // m_blockId;
    MString blockPathStr = "block";
    blockPathStr += path.str();

    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "u", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "v", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "w", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "rho", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "p", 3, allCells);

    if(m_averagingFavre) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "um_favre", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vm_favre", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "wm_favre", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "rhom_favre", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "pm_favre", 3, allCells);
    }

    if(m_averageVorticity) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vortx", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vorty", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vortz", 3, allCells);
    }

    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "uu", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vv", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "ww", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "uv", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vw", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "uw", 3, allCells);
    pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "pp", 3, allCells);

    if(m_averageVorticity) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vortxvortx", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vortyvorty", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vortzvortz", 3, allCells);
    }

    if(m_kurtosis || m_skewness) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "uuu", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vvv", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "www", 3, allCells);
    }

    if(m_kurtosis) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "uuuu", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "vvvv", 3, allCells);
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "wwww", 3, allCells);
    }
  }

  ////////////////////////////////////////////////
  ///////////// Write Variables //////////////////
  ////////////////////////////////////////////////
  ParallelIo::size_type ioOffset[3] = {0, 0, 0};
  ParallelIo::size_type ioSize[3] = {0, 0, 0};
  ParallelIo::size_type ioGhost[3] = {m_noGhostLayers, m_noGhostLayers, m_noGhostLayers};
  for(MInt dim = 0; dim < nDim; ++dim) {
    ioOffset[dim] = m_nOffsetCells[dim];
    ioSize[dim] = m_nActiveCells[dim];
  }

  stringstream path;
  path << m_blockId;
  MString blockPathStr = "block";
  blockPathStr += path.str();
  MInt offset = 0;
  pio.writeArray(&summedVars[0][0], blockPathStr, "u", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&summedVars[1][0], blockPathStr, "v", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&summedVars[2][0], blockPathStr, "w", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&summedVars[3][0], blockPathStr, "rho", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&summedVars[4][0], blockPathStr, "p", nDim, ioOffset, ioSize, ioGhost);
  offset = noVariables();

  if(m_averagingFavre) {
    pio.writeArray(&m_favre[0][0], blockPathStr, "um_favre", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&m_favre[1][0], blockPathStr, "vm_favre", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&m_favre[2][0], blockPathStr, "wm_favre", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&m_favre[3][0], blockPathStr, "rhom_favre", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&m_favre[4][0], blockPathStr, "pm_favre", nDim, ioOffset, ioSize, ioGhost);
  }

  if(m_averageVorticity) {
    pio.writeArray(&summedVars[offset + 0][0], blockPathStr, "vortx", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&summedVars[offset + 1][0], blockPathStr, "vorty", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&summedVars[offset + 2][0], blockPathStr, "vortz", nDim, ioOffset, ioSize, ioGhost);
  }

  pio.writeArray(&square[0][0], blockPathStr, "uu", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&square[1][0], blockPathStr, "vv", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&square[2][0], blockPathStr, "ww", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&square[3][0], blockPathStr, "uv", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&square[4][0], blockPathStr, "vw", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&square[5][0], blockPathStr, "uw", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&square[6][0], blockPathStr, "pp", nDim, ioOffset, ioSize, ioGhost);

  if(m_averageVorticity) {
    pio.writeArray(&square[7][0], blockPathStr, "vortxvortx", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&square[8][0], blockPathStr, "vortyvorty", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&square[9][0], blockPathStr, "vortzvortz", nDim, ioOffset, ioSize, ioGhost);
  }

  if(m_kurtosis || m_skewness) {
    pio.writeArray(&cube[0][0], blockPathStr, "uuu", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&cube[1][0], blockPathStr, "vvv", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&cube[2][0], blockPathStr, "www", nDim, ioOffset, ioSize, ioGhost);
  }

  if(m_kurtosis) {
    pio.writeArray(&fourth[0][0], blockPathStr, "uuuu", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&fourth[1][0], blockPathStr, "vvvv", nDim, ioOffset, ioSize, ioGhost);
    pio.writeArray(&fourth[2][0], blockPathStr, "wwww", nDim, ioOffset, ioSize, ioGhost);
  }
}


/**
 * \brief Writes the production terms into a given file
 * \author Marian Albers
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::saveProductionTerms(MString name, MFloat** production) {
  // Function to write the solution to file with iolibrary

  m_log << "writing production terms to file " << name << " ... " << endl;

  ParallelIoHdf5 pio(name, maia::parallel_io::PIO_APPEND, m_StructuredComm);

  writeHeaderAttributes(&pio, "solution");
  writePropertiesAsAttributes(&pio, "");

  ParallelIo::size_type allCells[3];
  for(MInt i = 0; i < m_noBlocks; i++) {
    for(MInt j = 0; j < nDim; j++) {
      allCells[j] = m_grid->getBlockNoCells(i, j);
    }
    // create datasets for the io library
    stringstream path;
    path << i;
    MString blockPathStr = "block";
    blockPathStr += path.str();

    // create dataset and write
    if(!pio.hasDataset("p1j", blockPathStr)) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "p1j", nDim, allCells);
    } else {
      cout << "Dataset p1j exists, not creating new" << endl;
    }
    if(!pio.hasDataset("p2j", blockPathStr)) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "p2j", nDim, allCells);
    }
    if(!pio.hasDataset("p3j", blockPathStr)) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "p3j", nDim, allCells);
    }
  }

  stringstream path;
  path << m_blockId;
  MString blockPathStr = "block";
  blockPathStr += path.str();

  ParallelIo::size_type ioOffset[3] = {0, 0, 0};
  ParallelIo::size_type ioSize[3] = {0, 0, 0};
  ParallelIo::size_type ioGhost[3] = {m_noGhostLayers, m_noGhostLayers, m_noGhostLayers};
  for(MInt dim = 0; dim < nDim; ++dim) {
    ioOffset[dim] = m_nOffsetCells[dim];
    ioSize[dim] = m_nActiveCells[dim];
  }

  pio.writeArray(&production[0][0], blockPathStr, "p1j", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&production[1][0], blockPathStr, "p2j", nDim, ioOffset, ioSize, ioGhost);
  pio.writeArray(&production[2][0], blockPathStr, "p3j", nDim, ioOffset, ioSize, ioGhost);
}

/**
 * \brief Writes the dissipation into a given file
 * \author Marian Albers
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::saveDissipation(MString name, MFloat* dissipation) {
  // Function to write the solution to file with iolibrary

  m_log << "writing dissipation terms to file " << name << " ... " << endl;

  ParallelIoHdf5 pio(name, maia::parallel_io::PIO_APPEND, m_StructuredComm);

  writeHeaderAttributes(&pio, "solution");
  writePropertiesAsAttributes(&pio, "");

  ParallelIo::size_type allCells[3];
  for(MInt i = 0; i < m_noBlocks; i++) {
    for(MInt j = 0; j < nDim; j++) {
      allCells[j] = m_grid->getBlockNoCells(i, j);
    }
    // create datasets for the io library
    stringstream path;
    path << i;
    MString blockPathStr = "block";
    blockPathStr += path.str();

    // create dataset and write
    if(!pio.hasDataset("diss", blockPathStr)) {
      pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, "diss", nDim, allCells);
    } else {
      cout << "Dataset diss exists, not creating new" << endl;
    }
  }

  stringstream path;
  path << m_blockId;
  MString blockPathStr = "block";
  blockPathStr += path.str();

  ParallelIo::size_type ioOffset[3] = {0, 0, 0};
  ParallelIo::size_type ioSize[3] = {0, 0, 0};
  ParallelIo::size_type ioGhost[3] = {m_noGhostLayers, m_noGhostLayers, m_noGhostLayers};
  for(MInt dim = 0; dim < nDim; ++dim) {
    ioOffset[dim] = m_nOffsetCells[dim];
    ioSize[dim] = m_nActiveCells[dim];
  }

  pio.writeArray(&dissipation[0], blockPathStr, "diss", nDim, ioOffset, ioSize, ioGhost);
}


/**
 * \brief Writes the gradients into a given file
 * \author Marian Albers
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::saveGradients(MString name, MFloat** gradients, MString* gradientNames) {
  // Function to write the solution to file with iolibrary

  m_log << "writing gradients to file " << name << " ... " << endl;

  ParallelIoHdf5 pio(name, maia::parallel_io::PIO_APPEND, m_StructuredComm);

  writeHeaderAttributes(&pio, "solution");
  writePropertiesAsAttributes(&pio, "");

  ParallelIo::size_type allCells[3];
  MInt noVars = 3 * 9;
  for(MInt i = 0; i < m_noBlocks; i++) {
    for(MInt j = 0; j < nDim; j++) {
      allCells[j] = m_grid->getBlockNoCells(i, j);
    }
    // create datasets for the io library
    stringstream path;
    path << i;
    MString blockPathStr = "block";
    blockPathStr += path.str();

    // create dataset and write
    for(MInt var = 0; var < noVars; var++) {
      if(!pio.hasDataset(gradientNames[var], blockPathStr)) {
        pio.defineArray(maia::parallel_io::PIO_FLOAT, blockPathStr, gradientNames[var], nDim, allCells);
      } else {
        cout << "Dataset " << gradientNames[var] << " exists, not creating new" << endl;
      }
    }
  }


  stringstream path;
  path << m_blockId;
  MString blockPathStr = "block";
  blockPathStr += path.str();

  ParallelIo::size_type ioOffset[3] = {0, 0, 0};
  ParallelIo::size_type ioSize[3] = {0, 0, 0};
  ParallelIo::size_type ioGhost[3] = {m_noGhostLayers, m_noGhostLayers, m_noGhostLayers};
  for(MInt dim = 0; dim < nDim; ++dim) {
    ioOffset[dim] = m_nOffsetCells[dim];
    ioSize[dim] = m_nActiveCells[dim];
  }

  for(MInt var = 0; var < noVars; var++) {
    pio.writeArray(&gradients[var][0], blockPathStr, gradientNames[var].c_str(), nDim, ioOffset, ioSize, ioGhost);
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::exchangeTimeStep() {
  TRACE();

  MFloat globalTimeStepMin = F0;

  MPI_Allreduce(&m_timeStep, &globalTimeStepMin, 1, MPI_DOUBLE, MPI_MIN, m_StructuredComm, AT_, "m_timeStep",
                "globalTimeStepMin");

  m_timeStep = globalTimeStepMin;
}


/**
 *  \brief Householder Reduction according to
 *         Numercial Recipies in C: The Art of Scientific Computing
 *
 *     Householder reduction of a real, symmetric matrix A.
 *     On output A is replaced by the orthogonla matrix Q (omitted here)
 *     diag  returns the diagonla elements of the tridiagonal matrix
 *     offdiag the off-diagonal elements with offdiag[1]=0;
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::tred2(MFloatScratchSpace& A, MInt dim, MFloat* diag, MFloat* offdiag) {
  // formulation from book pp. 582
  MInt n = dim;
  MInt l = 0, k = 0, j = 0, i = 0;
  MFloat scale = F0, hh = F0, h = F0, g = F0, f = F0;

  for(i = dim - 1; i > 0; i--) {
    l = i - 1;
    h = F0;
    scale = F0;
    if(l > 0) {
      for(k = 0; k < i; ++k) {
        scale += abs(A(i, k));
      }
      if(approx(scale, F0, m_eps)) {
        offdiag[i] = A(i, l);
      } else {
        for(k = 0; k < i; ++k) {
          A(i, k) /= scale;
          h += A(i, k) * A(i, k);
        }
        f = A(i, l);
        g = (f >= F0 ? -sqrt(h) : sqrt(h));
        offdiag[i] = scale * g;
        h -= f * g;
        A(i, l) = f - g;
        f = F0;
        for(j = 0; j < i; ++j) {
          g = F0;
          for(k = 0; k < j + 1; ++k) {
            g += A(j, k) * A(i, k);
          }
          for(k = j + 1; k < i; ++k) {
            g += A(k, j) * A(i, k);
          }
          offdiag[j] = g / h;
          f += offdiag[j] * A(i, j);
        }
        hh = f / (h + h);
        for(j = 0; j < i; j++) {
          f = A(i, j);
          g = offdiag[j] - hh * f;
          offdiag[j] = g;
          for(k = 0; k < j + 1; k++) {
            A(j, k) -= f * offdiag[k] + g * A(i, k);
          }
        }
      }
    } else {
      offdiag[i] = A(i, l);
    }
    diag[i] = h;
  }

  offdiag[0] = F0;
  for(i = 0; i < n; ++i) {
    diag[i] = A(i, i);
  }
}


/**
 * \brief  Compute Eigenvalues with implicit shift according to
 *     Numercial Recipies in C: The Art of Scientific Computing
 *
 *      QL algorithm with implicit shifts, to determine the eigenvalues of
 *      a real symmetric matrix previously reduced by tred2. diag is a vector
 *      of length np. On input, its first n elements are the diagonal
 *      elements of the tridiagonal matrix. On output, it returns the
 *      eigenvalues. The vector offdiag inputs the subdiagonal elements of the
 *      tridiagonal matrix, with offidag(0) arbitrary. On output offdiag
 *      is destroyed.
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::tqli2(MFloat* diag, MFloat* offdiag, MInt dim) //, MFloat** z)
{
  const MFloat eps = numeric_limits<MFloat>::epsilon();
  MInt m, l, iter, i, k;
  MFloat s, r, p, g, f, dd, c, b;
  MInt n = dim;
  for(i = 1; i < n; ++i)
    offdiag[i - 1] = offdiag[i];
  offdiag[n - 1] = 0.0;
  for(l = 0; l < n; ++l) {
    iter = 0;
    do {
      for(m = l; m < n - 1; m++) {
        dd = fabs(diag[m]) + fabs(diag[m + 1]);
        if(fabs(offdiag[m]) <= eps * dd) break;
      }
      if(m != l) {
        if(iter++ == 30) {
          for(k = 0; k < dim; ++k) {
            diag[k] = F0;
          }
          return;
        }
        g = (diag[l + 1] - diag[l]) / (2.0 * offdiag[l]);
        r = pythag(g, F1);
        r = abs(r);
        if(g < F0) r *= -F1;
        g = diag[m] - diag[l] + offdiag[l] / (g + r);
        s = c = F1;
        p = F0;
        for(i = m - 1; i >= l; i--) {
          f = s * offdiag[i];
          b = c * offdiag[i];
          offdiag[i + 1] = (r = pythag(f, g));
          if(approx(r, F0, eps)) {
            diag[i + 1] -= p;
            offdiag[m] = F0;
            break;
          }
          s = f / r;
          c = g / r;
          g = diag[i + 1] - p;
          r = (diag[i] - g) * s + 2.0 * c * b;
          diag[i + 1] = g + (p = s * r);
          g = c * r - b;
        }
        if(approx(r, F0, eps) && i >= l) continue;
        diag[l] -= p;
        offdiag[l] = g;
        offdiag[m] = F0;
      }
    } while(m != l);
  }
}

/**
 * \brief Sorting function to sort list in ascending order
 *
 *  sorts an array list into ascending numerical order, by straight
 *  insertion. dim is input(size of list); list is replaced on
 *  output by its sorted rearrangement.
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::insertSort(MInt dim, MFloat* list) {
  MFloat temp = F0;
  MInt j = 0;
  for(MInt i = 1; i < dim; i++) {
    temp = list[i];
    j = i - 1;
    while(j >= 0 && temp < list[j]) {
      list[j + 1] = list[j];
      j = j - 1;
    }
    list[j + 1] = temp;
  }
}

template <MInt nDim>
MFloat FvStructuredSolver<nDim>::pythag(MFloat a, MFloat b) {
  MFloat absa = F0, absb = F0;
  absa = fabs(a);
  absb = fabs(b);
  if(absa > absb)
    return absa * sqrt(1.0 + POW2(absb / absa));
  else
    return (approx(absb, F0, m_eps) ? F0 : absb * sqrt(1.0 + POW2(absa / absb)));
}

/**
 * \brief Checks whole domain for NaNs and adds the number of NaNs globally
 * \author Marian Albers
 * \date 19.08.2015
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::checkNans() {
  TRACE();
  MInt noNansLocal = 0;
  for(MInt cellid = 0; cellid < m_noCells; cellid++) {
    // go through every cell
    for(MInt i = 0; i < CV->noVariables; i++) {
      if(std::isnan(m_cells->variables[i][cellid])) {
        noNansLocal++;
      }
    }
  }

  MInt noNansGlobal = 0;
  MPI_Allreduce(&noNansLocal, &noNansGlobal, 1, MPI_INT, MPI_SUM, m_StructuredComm, AT_, "noNansLocal", "noNansGlobal");

  if(domainId() == 0) {
    cout << "GlobalTimeStep: " << globalTimeStep << " , noNansGlobal: " << noNansGlobal << endl;
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::setTimeStep() {
  TRACE();
  if(m_travelingWave || m_streamwiseTravelingWave) return;

  if((!m_constantTimeStep && globalTimeStep % m_timeStepComputationInterval == 0) || globalTimeStep == 0) {
    computeTimeStep();
    if(noDomains() > 1) {
      exchangeTimeStep();
    }

    m_physicalTimeStep = m_timeStep * m_timeRef;

    setLimiterVisc();
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::setLimiterVisc() {
  TRACE();

  if(m_limiterVisc) {
    // NOTE: currently only implemented in 2D
    ASSERT(nDim == 2, "Not implemented for 3D yet!");
    // xsd = 0, ysd = 1

    for(MInt cellId = 0; cellId < m_noCells; cellId++) {
      const MFloat metricScale =
          POW2(m_cells->cellMetrics[0 * 2 + 0][cellId]) + POW2(m_cells->cellMetrics[0 * 2 + 1][cellId])
          + POW2(m_cells->cellMetrics[1 * 2 + 0][cellId]) + POW2(m_cells->cellMetrics[1 * 2 + 1][cellId])
          + 2.0
                * (m_cells->cellMetrics[0 * 2 + 0][cellId] * m_cells->cellMetrics[1 * 2 + 0][cellId]
                   + m_cells->cellMetrics[0 * 2 + 1][cellId] * m_cells->cellMetrics[1 * 2 + 1][cellId]);
      // TODO_SS labels:FV,totest with localTimeStepping m_physicalTimeStep might not be set properly
      // TODO_SS labels:FV in some cases if Jacobian is negative the time step can also get negative -> think if current
      // implementation
      //       would also work in those cases
      m_cells->fq[FQ->LIMITERVISC][cellId] =
          m_CFLVISC * POW2(m_cells->cellJac[cellId]) / (m_physicalTimeStep * metricScale);
    }
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::fixTimeStepTravelingWave() {
  const MFloat waveTransSpeed = fabs(m_waveSpeed);

  MFloat travelDist = F0;
  if(m_travelingWave) {
    if(nDim == 3) {
      travelDist = abs(m_grid->m_coordinates[2][0] - m_grid->m_coordinates[2][m_nPoints[2] * m_nPoints[1]]);
    } else {
      travelDist = abs(m_grid->m_coordinates[0][0] - m_grid->m_coordinates[0][1]);
    }
  } else if(m_streamwiseTravelingWave) {
    travelDist = m_waveLength;
  } else {
    m_log << "WARNING!!!!!!!!!!!!!!!!!!!!: dz was fixed and hardcoded !!!!!!!!!! " << endl;
    travelDist = 0.33200866159432146;
  }

  if(!m_waveTimeStepComputed) {
    // first compute time step as usual and exchange it
    computeTimeStep();
    if(noDomains() > 1) {
      exchangeTimeStep();
    }

    // number of timeSteps the wave needs to move one cell width further
    m_waveNoStepsPerCell = ceil((travelDist / waveTransSpeed) / (m_timeStep));
    if(m_waveNoStepsPerCell < 2) {
      m_waveNoStepsPerCell = 2; // although highly improbable, see Nyquist-Shannon-Theorem
    }

    cout.precision(18);
    if(domainId() == 0) {
      cout << "Old time step = " << m_timeStep << endl;
    }
    m_log << "/////////////////// TRAVELING WAVE /////////////////////////////" << endl;
    m_log << "Old time step = " << m_timeStep << endl;

    m_timeStep = (travelDist / waveTransSpeed) / (m_waveNoStepsPerCell);
    m_physicalTimeStep = m_timeStep * m_timeRef;

    MBool syncSolutionInterval = false;
    MBool syncForceInterval = false;
    MBool syncBoxInterval = false;
    MBool syncIntpPointsInterval = false;


    // fix output writing functions in case
    if(m_synchronizedMGOutput) {
      // change the output frequency of the different io routines to be in accordance with the new time step such that
      // only in full number of iterations in which the moving grid has traveld one cell distance is ensured
      // if(m_outputOffset){
      //  m_log << "ERROR: m_outputOffset and synchronized output cannot be combined (not implemented yet)" << endl;
      //  mTerm(1, AT_, "ERROR: m_outputOffset and synchronized output cannot be combined (not implemented yet)");
      //}

      if(domainId() == 0) {
        cout << "m_movingGridStepOffset: " << m_movingGridStepOffset
             << " m_movingGridInitialStart: " << m_movingGridInitialStart << endl;
      }


      if(m_outputOffset > m_movingGridStepOffset) {
        const MInt offsetCounter =
            ceil(((MFloat)m_outputOffset - (MFloat)m_movingGridStepOffset) / (MFloat)m_waveNoStepsPerCell);
        m_outputOffset = m_movingGridStepOffset + offsetCounter * m_waveNoStepsPerCell;
      } else {
        m_outputOffset = m_movingGridStepOffset; // for input output such that it works propberly
      }

      if(m_useConvectiveUnitWrite) {
        m_log << "ERROR: m_useConvectiveUnitWrite and synchronized output cannot be combined (not implemented yet)"
              << endl;
        mTerm(1, AT_,
              "ERROR: m_useConvetiveUnitWrite and synchronized output cannot be combined (not implemented yet)");
      }
      if(m_solutionInterval) {
        m_solutionInterval =
            mMax(m_waveNoStepsPerCell,
                 m_waveNoStepsPerCell * (MInt)floor((MFloat)m_solutionInterval / (MFloat)m_waveNoStepsPerCell));
        syncSolutionInterval = true;
      }
      if(m_forceOutputInterval) {
        m_forceOutputInterval =
            mMax(m_waveNoStepsPerCell,
                 m_waveNoStepsPerCell * (MInt)floor((MFloat)m_forceOutputInterval / (MFloat)m_waveNoStepsPerCell));
        syncForceInterval = true;
      }
      if(m_boxOutputInterval) {
        m_boxOutputInterval =
            mMax(m_waveNoStepsPerCell,
                 m_waveNoStepsPerCell * (MInt)floor((MFloat)m_boxOutputInterval / (MFloat)m_waveNoStepsPerCell));

        syncBoxInterval = true;
      }
      if(m_intpPointsOutputInterval) {
        m_intpPointsOutputInterval =
            mMax(m_waveNoStepsPerCell,
                 m_waveNoStepsPerCell * (MInt)floor((MFloat)m_boxOutputInterval / (MFloat)m_waveNoStepsPerCell));
        syncIntpPointsInterval = true;
      }
    }

    if(domainId() == 0) {
      cout << "New time step: " << m_timeStep << " new physical time step: " << m_physicalTimeStep << endl;
      cout << "Number of steps to move one cell width: " << travelDist / (m_waveSpeed * m_timeStep) << " time steps"
           << endl;
      cout << "Number of steps to move one physical time step: " << F1 / m_physicalTimeStep << endl;
      cout << "Number of steps to move one wave length: " << m_waveLength / (m_waveSpeed * m_timeStep) << endl;
      cout << "Cells per wavelength: " << m_waveLength / travelDist << " travelDist: " << travelDist << endl;
      cout << "Time for wave to travel one wave length: " << m_waveLength / m_waveSpeed << endl;
      cout << "Physical time for wave to travel one wave length: " << m_waveLength / m_waveSpeed * m_timeRef << endl;
      cout << "solution output interval was reset: " << syncSolutionInterval << " and changed to " << m_solutionInterval
           << endl;
      cout << "box output interval was reset: " << syncBoxInterval << " and changed to " << m_boxOutputInterval << endl;
      cout << "force output interval was reset: " << syncForceInterval << " and changed to " << m_forceOutputInterval
           << endl;
      cout << "intpPoints output interval was reset: " << syncIntpPointsInterval << " and changed to "
           << m_intpPointsOutputInterval << endl;
    }

    m_log << "New time step: " << m_timeStep << " new physical time step: " << m_physicalTimeStep << endl;
    m_log << "Number of steps to move one cell width: " << travelDist / (m_waveSpeed * m_timeStep) << " time steps"
          << endl;
    m_log << "Number of steps to move one physical time step: " << F1 / m_physicalTimeStep << endl;
    m_log << "Number of steps to move one wave length: " << m_waveLength / (m_waveSpeed * m_timeStep) << endl;
    m_log << "Cells per wavelength: " << m_waveLength / travelDist << " travelDist: " << travelDist << endl;
    m_log << "Time for wave to travel one wave length: " << m_waveLength / m_waveSpeed << endl;
    m_log << "Physical time for wave to travel one wave length: " << m_waveLength / m_waveSpeed * m_timeRef << endl;
    m_log << "solution output interval was reset: " << syncSolutionInterval << " and changed to " << m_solutionInterval
          << endl;
    m_log << "box output interval was reset: " << syncBoxInterval << " and changed to " << m_boxOutputInterval << endl;
    m_log << "force output interval was reset: " << syncForceInterval << " and changed to " << m_forceOutputInterval
          << endl;
    m_log << "intpPoints output interval was reset: " << syncIntpPointsInterval << " and changed to "
          << m_intpPointsOutputInterval << endl;
    m_log << "solution writing out will start at" << m_outputOffset << endl;
    m_log << "////////////////////////////////////////////////////////////////" << endl;

    m_waveTimeStepComputed = true;
  }

  if(globalTimeStep == 0 || globalTimeStep == m_restartTimeStep) {
    MBool syncSolutionInterval = false;
    MBool syncForceInterval = false;
    MBool syncBoxInterval = false;
    MBool syncIntpPointsInterval = false;

    // fix output writing functions in case
    if(m_synchronizedMGOutput) {
      // change the output frequency of the different io routines to be in accordance with the new time step such that
      // only in full number of iterations in which the moving grid has traveld one cell distance is ensured
      // if(m_outputOffset!=m_movingGridStepOffset){
      //  m_log << "ERROR: m_outputOffset and synchronized output cannot be combined (not implemented yet)" << endl;
      //  mTerm(1, AT_, "ERROR: m_outputOffset and synchronized output cannot be combined (not implemented yet)");
      //}
      // shift the starting point of the outputs for the moving grid

      if(m_outputOffset > m_movingGridStepOffset) {
        const MInt offsetCounter =
            ceil(((MFloat)m_outputOffset - (MFloat)m_movingGridStepOffset) / (MFloat)m_waveNoStepsPerCell);
        m_outputOffset = m_movingGridStepOffset + offsetCounter * m_waveNoStepsPerCell;
      } else {
        m_outputOffset = m_movingGridStepOffset; // for input output such that it works propberly
      }


      if(m_useConvectiveUnitWrite) {
        m_log << "ERROR: m_useConvectiveUnitWrite and synchronized output cannot be combined (not implemented yet)"
              << endl;
        mTerm(1, AT_,
              "ERROR: m_useConvetiveUnitWrite and synchronized output cannot be combined (not implemented yet)");
      }
      if(m_solutionInterval) {
        m_solutionInterval =
            mMax(m_waveNoStepsPerCell,
                 m_waveNoStepsPerCell * (MInt)floor((MFloat)m_solutionInterval / (MFloat)m_waveNoStepsPerCell));
        syncSolutionInterval = true;
      }
      if(m_forceOutputInterval) {
        m_forceOutputInterval =
            mMax(m_waveNoStepsPerCell,
                 m_waveNoStepsPerCell * (MInt)floor((MFloat)m_forceOutputInterval / (MFloat)m_waveNoStepsPerCell));
        syncForceInterval = true;
      }
      if(m_boxOutputInterval) {
        m_boxOutputInterval =
            mMax(m_waveNoStepsPerCell,
                 m_waveNoStepsPerCell * (MInt)floor((MFloat)m_boxOutputInterval / (MFloat)m_waveNoStepsPerCell));
        syncBoxInterval = true;
      }
      if(m_intpPointsOutputInterval) {
        m_intpPointsOutputInterval =
            mMax(m_waveNoStepsPerCell,
                 m_waveNoStepsPerCell * (MInt)floor((MFloat)m_boxOutputInterval / (MFloat)m_waveNoStepsPerCell));
        syncIntpPointsInterval = true;
      }
    }

    m_log << "/////////////////// TRAVELING WAVE /////////////////////////////" << endl;
    m_log << "New time step: " << m_timeStep << " new physical time step: " << m_physicalTimeStep << endl;
    m_log << "Number of steps to move one cell width: " << travelDist / (m_waveSpeed * m_timeStep) << " time steps"
          << endl;
    m_log << "Number of steps to move one physical time step: " << F1 / m_physicalTimeStep << endl;
    m_log << "Number of steps to move one wave length: " << m_waveLength / (m_waveSpeed * m_timeStep) << endl;
    m_log << "Time for wave to travel one wave length: " << m_waveLength / m_waveSpeed << endl;
    m_log << "Cells per wavelength: " << m_waveLength / travelDist << " travelDist: " << travelDist << endl;
    m_log << "solution output interval was reset: " << syncSolutionInterval << " and changed to " << m_solutionInterval
          << endl;
    m_log << "box output interval was reset: " << syncBoxInterval << " and changed to " << m_boxOutputInterval << endl;
    m_log << "force output interval was reset: " << syncForceInterval << " and changed to " << m_forceOutputInterval
          << endl;
    m_log << "intpPoints output interval was reset: " << syncIntpPointsInterval << " and changed to "
          << m_intpPointsOutputInterval << endl;
    m_log << "solution writing out will start at" << m_outputOffset << endl;
    m_log << "////////////////////////////////////////////////////////////////" << endl;
    // correct Averaging Time Steps
    if(m_postprocessing) {
      const MInt waveNoStepsPerCell = m_waveNoStepsPerCell;
      if(m_averageInterval % waveNoStepsPerCell != 0) {
        m_log << "Changed averageInterval from " << m_averageInterval;
        const MInt minAverageInterval = waveNoStepsPerCell;
        m_averageInterval =
            mMax(minAverageInterval,
                 (MInt)(waveNoStepsPerCell * floor((MFloat)m_averageInterval / (MFloat)waveNoStepsPerCell)));
        m_log << " to " << m_averageInterval << ", every " << m_averageInterval * m_physicalTimeStep
              << " convective units" << endl;
      }

      const MInt waveStepOffset = m_movingGridStepOffset;
      if((m_averageStartTimestep - waveStepOffset) % m_averageInterval != 0) {
        m_log << "Changed averageStartTimeStep from " << m_averageStartTimestep;
        MInt offsetCounter =
            ceil(((MFloat)m_averageStartTimestep - (MFloat)waveStepOffset) / (MFloat)m_averageInterval);
        m_averageStartTimestep = waveStepOffset + offsetCounter * m_averageInterval;
        m_log << " to " << m_averageStartTimestep << ". Averaging every " << m_averageInterval << " time steps" << endl;
      }

      if((m_averageStopTimestep - waveStepOffset) % m_averageInterval != 0) {
        m_log << "Changed averageStopTimeStep from " << m_averageStopTimestep;
        MInt offsetCounter = ceil(((MFloat)m_averageStopTimestep - (MFloat)waveStepOffset) / (MFloat)m_averageInterval);
        m_averageStopTimestep = waveStepOffset + offsetCounter * m_averageInterval;
        m_log << " to " << m_averageStopTimestep << ". Averaging every " << m_averageInterval << " time steps" << endl;
      }
    }
  }
}

template <MInt nDim>
void FvStructuredSolver<nDim>::convertRestartVariables(MFloat oldMa) {
  TRACE();
  const MFloat eps = pow(10.0, -5.0);
  if(ABS(oldMa - m_Ma) > eps) {
    m_log << "converting restart variables from old Ma: " << oldMa << " to new Ma: " << m_Ma << " ..." << endl;
    const MFloat gammaMinusOne = m_gamma - 1.0;
    // old references
    MFloat T8old = 1.0 / (1.0 + 0.5 * gammaMinusOne * POW2(oldMa));
    MFloat p8old = pow(T8old, (m_gamma / gammaMinusOne)) / m_gamma;
    MFloat u8old = oldMa * sqrt(T8old);
    MFloat rho8old = pow(T8old, (1.0 / gammaMinusOne));
    // MFloat rhoU8old = rho8old*u8old;
    MFloat rhoE8old = p8old / gammaMinusOne + rho8old * (F1B2 * POW2(u8old));
    // new references
    MFloat T8new = 1.0 / (1.0 + 0.5 * gammaMinusOne * POW2(m_Ma));
    MFloat p8new = pow(T8new, (m_gamma / gammaMinusOne)) / m_gamma;
    MFloat u8new = m_Ma * sqrt(T8new);
    MFloat rho8new = pow(T8new, (1.0 / gammaMinusOne));
    // MFloat rhoU8new = rho8new*u8new;
    MFloat rhoE8new = p8new / gammaMinusOne + rho8new * (F1B2 * POW2(u8new));
    // ratios
    MFloat velRatio = u8new / u8old;
    MFloat rhoRatio = rho8new / rho8old;
    MFloat pRatio = p8new / p8old;
    MFloat rhoERatio = rhoE8new / rhoE8old;

    // conversion
    for(MInt cellId = 0; cellId < m_noCells; ++cellId) {
      // density
      m_cells->pvariables[PV->RHO][cellId] *= rhoRatio;
      // velocities
      for(MInt i = 0; i < nDim; ++i) {
        m_cells->pvariables[PV->VV[i]][cellId] *= velRatio;
      }
      // energy
      m_cells->pvariables[PV->P][cellId] *= pRatio;
    }

    if(m_useSponge) {
      if(m_spongeLayerType == 2) {
        for(MInt cellId = 0; cellId < m_noCells; ++cellId) {
          m_cells->fq[FQ->SPONGE_RHO][cellId] *= rhoRatio;
          m_cells->fq[FQ->SPONGE_RHO_E][cellId] *= rhoERatio;
        }
      }

      if(m_spongeLayerType == 4) {
        for(MInt cellId = 0; cellId < m_noCells; ++cellId) {
          m_cells->fq[FQ->SPONGE_RHO][cellId] *= rhoRatio;
        }
      }
    }

    computeConservativeVariables();

    m_log << "converting restart variables from old Ma: " << oldMa << " to new Ma: " << m_Ma << " ... SUCCESSFUL!"
          << endl;
  }
  return;
}


/**
 * \brief  Return decomposition information, i.e. number of local elements,...
 * \author Marian Albers
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::getDomainDecompositionInformation(std::vector<std::pair<MString, MInt>>& domainInfo) {
  TRACE();

  const MString namePrefix = "b" + std::to_string(solverId()) + "_";

  // Number of Cells
  const MInt noCells = m_noCells;

  domainInfo.emplace_back(namePrefix + "noStructuredCells", noCells);
}


/// Get solver timings
template <MInt nDim>
void FvStructuredSolver<nDim>::getSolverTimings(std::vector<std::pair<MString, MFloat>>& solverTimings,
                                                const MBool NotUsed(allTimings)) {
  TRACE();
  const MString namePrefix = "b" + std::to_string(solverId()) + "_";


  solverTimings.emplace_back(namePrefix + "Viscous Flux", RETURN_TIMER_TIME(m_timers[Timers::ViscousFlux]));
  solverTimings.emplace_back(namePrefix + "ConvectiveFlux", RETURN_TIMER_TIME(m_timers[Timers::ConvectiveFlux]));
  solverTimings.emplace_back(namePrefix + "Exchange", RETURN_TIMER_TIME(m_timers[Timers::Exchange]));
  solverTimings.emplace_back(namePrefix + "BoundaryCondition", RETURN_TIMER_TIME(m_timers[Timers::BoundaryCondition]));
  solverTimings.emplace_back(namePrefix + "MLCoupling", RETURN_TIMER_TIME(m_timers[Timers::MLCoupling]));
  solverTimings.emplace_back(namePrefix + "RungeKutta Step", RETURN_TIMER_TIME(m_timers[Timers::RungeKutta]));
  solverTimings.emplace_back(namePrefix + "Save output", RETURN_TIMER_TIME(m_timers[Timers::SaveOutput]));
  solverTimings.emplace_back(namePrefix + "Save forces", RETURN_TIMER_TIME(m_timers[Timers::SaveForces]));
  solverTimings.emplace_back(namePrefix + "Save auxdata", RETURN_TIMER_TIME(m_timers[Timers::SaveAuxdata]));
  solverTimings.emplace_back(namePrefix + "Save boxes", RETURN_TIMER_TIME(m_timers[Timers::SaveBoxes]));
  solverTimings.emplace_back(namePrefix + "Save intp points", RETURN_TIMER_TIME(m_timers[Timers::SaveIntpPoints]));
  solverTimings.emplace_back(namePrefix + "Save solution", RETURN_TIMER_TIME(m_timers[Timers::SaveSolution]));
  solverTimings.emplace_back(namePrefix + "Sandpaper tripping", RETURN_TIMER_TIME(m_timers[Timers::SandpaperTrip]));
  solverTimings.emplace_back(namePrefix + "Moving Grid", RETURN_TIMER_TIME(m_timers[Timers::MovingGrid]));
  solverTimings.emplace_back(namePrefix + "Moving grid volume flux", RETURN_TIMER_TIME(m_timers[Timers::MGVolumeFlux]));
  solverTimings.emplace_back(namePrefix + "MG Move Grid", RETURN_TIMER_TIME(m_timers[Timers::MGMoveGrid]));
}


/**
 * \brief Init for Falkner-Skan-Cooke flow
 *
 * \author T.Schilden
 * \date 11.12.2013
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::initFsc() {
  TRACE();

  m_log << "initialization of falkner scan cooke " << endl;

  // create file to read the fsc velocity distribution
  ParallelIo parallelIo("fsc.Netcdf", maia::parallel_io::PIO_READ, mpiComm());

  // get the size of the arrays
  vector<ParallelIo::size_type> dims = parallelIo.getArrayDims("eta");
  m_fsc_noPoints = (MInt)(dims[0]);
  parallelIo.setOffset(m_fsc_noPoints, 0);

  // allocate memory for the arrays
  mAlloc(m_fsc_eta, m_fsc_noPoints, "m_fsc_eta", AT_);
  mAlloc(m_fsc_fs, m_fsc_noPoints, "m_fsc_fs", AT_);
  mAlloc(m_fsc_f, m_fsc_noPoints, "m_fsc_f", AT_);
  mAlloc(m_fsc_g, m_fsc_noPoints, "m_fsc_g", AT_);

  // read the arrays
  parallelIo.readArray(m_fsc_eta, "eta");
  parallelIo.readArray(m_fsc_fs, "fprim");
  parallelIo.readArray(m_fsc_g, "g");
  parallelIo.readScalar(&m_fsc_m, "m");

  // FSC Parameters
  m_fsc_Re = cos(m_angle[1]) * m_Re;

  /*! \property
    \page propertiesFVSTRCTRD
    \section fscX0
    <code>MFloat FvStructuredSolver::m_fsc_x0 </code>\n
    default = <code>none</code>\n \n
    x_0 of the fsc boundary layer, e.g., $(x/x_0)^m$
    Possible values are:
    <ul>
    <li>larger zero</li>
    </ul>
    Keywords: <i>FINITE_VOLUME, LAMINAR_BOUNDARY_LAYER</i>
  */
  m_fsc_x0 = Context::getSolverProperty<MFloat>("fscX0", m_solverId, AT_, &m_fsc_x0);


  m_fsc_dx0 = F0;
  /*! \property
    \page propertiesFVSTRCTRD
    \section fscDX0
    <code>MFloat FvStructuredSolver::m_fsc_dx0 </code>\n
    default = <code>F0</code>\n \n
    x coordinate of x_0 in maia coordinates
    Possible values are:
    <ul>
    <li> whatever you want </li>
    </ul>
    Keywords: <i>FINITE_VOLUME, LAMINAR_BOUNDARY_LAYER</i>
  */
  m_fsc_dx0 = Context::getSolverProperty<MFloat>("fscDX0", m_solverId, AT_, &m_fsc_dx0);

  m_fsc_y0 = F0;
  /*! \property
    \page propertiesFVSTRCTRD
    \section fscY0
    <code>MFloat FvStructuredSolver::m_fsc_y0 </code>\n
    default = <code>F0</code>\n \n
    y position of the fsc boundary layer no-slip surface
    Possible values are:
    <ul>
    <li> whatever you want </li>
    </ul>
    Keywords: <i>FINITE_VOLUME, LAMINAR_BOUNDARY_LAYER</i>
  */
  m_fsc_y0 = Context::getSolverProperty<MFloat>("fscY0", m_solverId, AT_, &m_fsc_y0);


  m_log << "fsc Reynolds number " << m_fsc_Re << endl;
  m_log << "acceleration parameter " << m_fsc_m << endl;
  m_log << "reference location x " << m_fsc_x0 << endl;
  m_log << "reference location in maia coords " << m_fsc_dx0 << endl;
  m_log << "y position of no-slip " << m_fsc_y0 << endl;

  // calculate the integral of fs
  const MFloat detaB2 = m_fsc_eta[1] / F2;
  m_fsc_f[0] = F0;
  for(MInt i = 1; i < m_fsc_noPoints; i++)
    m_fsc_f[i] = m_fsc_f[i - 1] + detaB2 * (m_fsc_fs[i - 1] + m_fsc_fs[i]);

  // debug raw distributions
  if(true && !domainId()) {
    ofstream fscf;
    // write pressure to file
    fscf.open("fsc_raw.dat", ios::trunc);
    if(fscf) {
      fscf << "#eta fs f g" << endl;
      for(MInt i = 0; i < m_fsc_noPoints; i++) {
        fscf << m_fsc_eta[i] << " " << m_fsc_fs[i] << " " << m_fsc_f[i] << " " << m_fsc_g[i] << endl;
      }
      fscf.close();
    }
  }
  m_log << " done " << endl;
}


template <MInt nDim>
MFloat FvStructuredSolver<nDim>::getFscPressure(MInt cellId) {
  return getFscPressure(m_cells->coordinates[0][cellId]);
}

template <MInt nDim>
MFloat FvStructuredSolver<nDim>::getFscPressure(MFloat coordX) {
  const MFloat dx = coordX - m_fsc_dx0;
  const MFloat x = m_fsc_x0 + dx;

  const MFloat fac = CV->rhoInfinity * POW2(PV->UInfinity) * F1B2;
  return PV->PInfinity + fac * (F1 - pow(x / m_fsc_x0, F2 * m_fsc_m));
}

/**
 * \brief Load variables for the specified timeStep
 *
 * \author T.Schilden
 * \date 11.12.2013
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::getFscVelocity(MInt cellId, MFloat* const vel) {
  getFscVelocity(m_cells->coordinates[0][cellId], m_cells->coordinates[1][cellId], vel);
}

template <MInt nDim>
MFloat FvStructuredSolver<nDim>::getFscEta(MFloat coordX, MFloat coordY) {
  const MFloat dx = coordX - m_fsc_dx0;
  const MFloat x = m_fsc_x0 + dx;
  const MFloat y = coordY - m_fsc_y0;
  return y * sqrt((m_fsc_m + F1) * m_fsc_Re * pow(x / m_fsc_x0, m_fsc_m) / (F2 * x));
}


template <MInt nDim>
void FvStructuredSolver<nDim>::getFscVelocity(MFloat coordX, MFloat coordY, MFloat* const vel) {
  // coordinates
  const MFloat eta = getFscEta(coordX, coordY);
  const MFloat dx = coordX - m_fsc_dx0;
  const MFloat x = m_fsc_x0 + dx;

  // get f, fs
  MFloat fs, f;
  if(eta >= m_fsc_eta[m_fsc_noPoints - 1]) { // freestream
    fs = m_fsc_fs[m_fsc_noPoints - 1];
    f = m_fsc_f[m_fsc_noPoints - 1] + m_fsc_fs[m_fsc_noPoints - 1] * (eta - m_fsc_eta[m_fsc_noPoints - 1]);
  } else if(eta <= F0) { // set zero, this is wrong in case of blowing or suction
    fs = F0;
    f = F0;
  } else { // interpolate
    // get index
    const MFloat Deta = m_fsc_eta[1];
    const MInt etai = eta / Deta;
    ASSERT((etai + 1) <= (m_fsc_noPoints - 1), "error computing the index for fsc distributions");
    const MFloat deta = eta - m_fsc_eta[etai];
    const MFloat phi = deta / Deta;
    fs = m_fsc_fs[etai] * (F1 - phi) + m_fsc_fs[etai + 1] * phi;
    f = m_fsc_f[etai] * (F1 - phi) + m_fsc_f[etai + 1] * phi;
  }

  // "streamwise"
  vel[0] = PV->UInfinity * pow(x / m_fsc_x0, m_fsc_m) * fs;
  // normal, computet as inside the eta range
  const MFloat fac = sqrt(F2 * pow(x / m_fsc_x0, m_fsc_m) / ((m_fsc_m + F1) * m_fsc_x0 * m_fsc_Re)) / F2;
  //   eta might be < 0 but then, fs and f are zero
  vel[1] = PV->UInfinity * fac * ((F1 - m_fsc_m) * fs * eta - (F1 + m_fsc_m) * f);

  // spanwise
  IF_CONSTEXPR(nDim > 2) {
    MFloat g;
    if(eta >= m_fsc_eta[m_fsc_noPoints - 1]) { // freestream
      g = m_fsc_g[m_fsc_noPoints - 1];
    } else if(eta <= F0) { // set zero, this is wrong in case of blowing or suction
      g = F0;
    } else { // interpolate
      // get index
      const MFloat Deta = m_fsc_eta[1];
      const MInt etai = eta / Deta;
      ASSERT((etai + 1) <= (m_fsc_noPoints - 1), "error computing the index for fsc distributions");
      const MFloat deta = eta - m_fsc_eta[etai];
      const MFloat phi = deta / Deta;
      g = m_fsc_g[etai] * (F1 - phi) + m_fsc_g[etai + 1] * phi;
    }
    vel[2] = PV->WInfinity * g;
  }
}


/**
 * \brief Init for Blasius boundary layer
 *
 * \author Marian Albers
 * \date 03.06.2020
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::initBlasius() {
  TRACE();


  const MFloat fppwall = 0.332051914927913096446;

  // Calculate Blasius solution
  const MFloat etamax = 20.0;
  const MFloat deta = 0.001;
  const MInt steps = etamax / deta;

  MFloatScratchSpace blasius(steps + 1, 4, "m_blasius", AT_);
  blasius.fill(F0);
  blasius(0, 3) = fppwall;

  // Lets use the shooting method for getting the Blasius solution
  for(MInt i = 0; i < steps; i++) {
    blasius(i + 1, 0) = blasius(i, 0) + deta;
    blasius(i + 1, 1) = blasius(i, 1) + blasius(i, 2) * deta;
    blasius(i + 1, 2) = blasius(i, 2) + blasius(i, 3) * deta;
    blasius(i + 1, 3) = blasius(i, 3) + (-0.5 * blasius(i, 1) * blasius(i, 3)) * deta;
  }

  m_blasius_noPoints = steps;

  // allocate memory for the arrays
  mAlloc(m_blasius_eta, m_blasius_noPoints, "m_blasius_eta", AT_);
  mAlloc(m_blasius_fp, m_blasius_noPoints, "m_blasius_fp", AT_);
  mAlloc(m_blasius_f, m_blasius_noPoints, "m_blasius_f", AT_);

  const MFloat finalfp = blasius(steps - 1, 2);

  for(MInt i = 0; i < steps; i++) {
    m_blasius_eta[i] = blasius(i, 0);
    m_blasius_f[i] = blasius(i, 1) / finalfp;
    m_blasius_fp[i] = blasius(i, 2) / finalfp;
  }

  // debug raw distributions
  if(domainId() == 0) {
    ofstream blasiusf;
    // write pressure to file
    blasiusf.open("blasius_raw.dat", ios::trunc);
    if(blasiusf) {
      blasiusf << "#eta f fp" << endl;
      for(MInt i = 0; i < m_blasius_noPoints; i++) {
        blasiusf << m_blasius_eta[i] << " " << m_blasius_f[i] << " " << m_blasius_fp[i] << endl;
      }
      blasiusf.close();
    }
  }

  m_blasius_dx0 = 0.0;
  m_blasius_y0 = 0.0; // position of the wall
}


/**
 * \brief Load variables for the specified timeStep
 *
 * \author T.Schilden
 * \date 11.12.2013
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::getBlasiusVelocity(MInt cellId, MFloat* const vel) {
  getBlasiusVelocity(m_cells->coordinates[0][cellId], m_cells->coordinates[1][cellId], vel);
}

template <MInt nDim>
MFloat FvStructuredSolver<nDim>::getBlasiusEta(MFloat coordX, MFloat coordY) {
  const MFloat nu8 = SUTHERLANDLAW(PV->TInfinity) / CV->rhoInfinity;
  m_blasius_x0 = F1 / POW2(1.7208) * PV->UInfinity / nu8 * m_Re0;

  const MFloat dx = coordX - m_blasius_dx0;
  const MFloat x = m_blasius_x0 + dx;
  const MFloat y = coordY - m_blasius_y0;

  return y * sqrt(PV->UInfinity * m_Re0 / (nu8 * x));
}


template <MInt nDim>
void FvStructuredSolver<nDim>::getBlasiusVelocity(MFloat coordX, MFloat coordY, MFloat* const vel) {
  // coordinates
  const MFloat nu8 = SUTHERLANDLAW(PV->TInfinity) / CV->rhoInfinity;
  m_blasius_x0 = F1 / POW2(1.7208) * PV->UInfinity / nu8 * m_Re0;
  const MFloat eta = getBlasiusEta(coordX, coordY);
  const MFloat dx = coordX - m_blasius_dx0;
  const MFloat x = m_blasius_x0 + dx;

  // get f, fp, and g
  MFloat fp, f;
  if(eta >= m_blasius_eta[m_blasius_noPoints - 1]) { // freestream
    fp = m_blasius_fp[m_blasius_noPoints - 1];
    f = m_blasius_f[m_blasius_noPoints - 1]
        + m_blasius_fp[m_blasius_noPoints - 1] * (eta - m_blasius_eta[m_blasius_noPoints - 1]);
  } else if(eta <= F0) { // set zero, this is wrong in case of blowing or suction
    fp = F0;
    f = F0;
  } else { // interpolate
    // get index
    const MFloat Deta = m_blasius_eta[1];
    const MInt etai = eta / Deta;
    ASSERT((etai + 1) <= (m_blasius_noPoints - 1), "error computing the index for Blasius distributions");
    const MFloat deta = eta - m_blasius_eta[etai];
    const MFloat phi = deta / Deta;
    fp = m_blasius_fp[etai] * (F1 - phi) + m_blasius_fp[etai + 1] * phi;
    f = m_blasius_f[etai] * (F1 - phi) + m_blasius_f[etai + 1] * phi;
  }

  vel[0] = PV->UInfinity * fp;
  vel[1] = F1B2 * sqrt(PV->UInfinity * nu8 / (x * m_Re0)) * (eta * fp - f);
  IF_CONSTEXPR(nDim == 3) { vel[2] = 0.0; }
}


/// Initialize solver
/// moved from methods
/// \author Sven Berger
/// \tparam nDim
template <MInt nDim>
void FvStructuredSolver<nDim>::initSolver() {
  TRACE();
  RECORD_TIMER_START(m_timers[Timers::RunInit]);
  initializeRungeKutta();

  // Note: timers needed here for subtimers started in exchange() called from initSolutionStep()
  RECORD_TIMER_START(m_timers[Timers::Run]);
  RECORD_TIMER_START(m_timers[Timers::MainLoop]);
  initSolutionStep(-1);
  RECORD_TIMER_STOP(m_timers[Timers::MainLoop]);
  RECORD_TIMER_STOP(m_timers[Timers::Run]);

  setTimeStep();
  RECORD_TIMER_STOP(m_timers[Timers::RunInit]);
}

template <MInt nDim>
void FvStructuredSolver<nDim>::finalizeInitSolver() {
  this->postprocessPreInit();
  this->postprocessPreSolve();
}
///////////////////////////////////////////////////////////////////////////////
/// AUX DATA //////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <MInt nDim>
void FvStructuredSolver<nDim>::allocateAuxDataMaps() {
  TRACE();

  if(m_ransMethod == RANS_KEPSILON) {
    m_bForce = true;
    m_detailAuxData = true;
  }

  if(m_bForce) {
    const MInt noFields = nDim + m_detailAuxData * 2 * nDim;

    MInt noAuxDataMaps = m_windowInfo->physicalAuxDataMap.size();
    if(noAuxDataMaps > 0) {
      mAlloc(m_cells->cfOffsets, noAuxDataMaps, "m_cells->cfOffsets", 0, AT_);
      mAlloc(m_cells->cpOffsets, noAuxDataMaps, "m_cells->cpOffsets", 0, AT_);
      mAlloc(m_cells->powerOffsets, noAuxDataMaps, "m_cells->powerOffsets", 0, AT_);
    }
    MInt totalSizeCf = 0, totalSizeCp = 0, totalSizePower = 0;
    for(MInt i = 0; i < noAuxDataMaps; ++i) {
      MInt dataSize = 1;
      for(MInt j = 0; j < nDim; ++j) {
        if(m_windowInfo->physicalAuxDataMap[i]->end1[j] == m_windowInfo->physicalAuxDataMap[i]->start1[j]) {
          continue;
        }
        dataSize *= m_windowInfo->physicalAuxDataMap[i]->end1[j] - m_windowInfo->physicalAuxDataMap[i]->start1[j];
      }
      m_cells->cfOffsets[i] = totalSizeCf;
      m_cells->cpOffsets[i] = totalSizeCp;
      m_cells->powerOffsets[i] = totalSizePower;
      totalSizeCf += dataSize * noFields;
      totalSizeCp += dataSize;
      totalSizePower += dataSize * nDim;
    }

    if(totalSizeCf > 0) {
      mAlloc(m_cells->cf, totalSizeCf, "m_cells->cf", -1.23456123456, AT_);
    }
    if(totalSizeCp > 0) {
      mAlloc(m_cells->cp, totalSizeCp, "m_cells->cp", -1.23456123456, AT_);
    }

    if(m_bPower) {
      if(totalSizePower > 0) {
        mAlloc(m_cells->powerVisc, totalSizePower, "m_cells->power", -1.23456123456, AT_);
        mAlloc(m_cells->powerPres, totalSizePower, "m_cells->power", -1.23456123456, AT_);
      }
    }

    m_forceHeaderNames.clear();
    m_forceHeaderNames.push_back("n");
    m_forceHeaderNames.push_back("t_ac");
    m_forceHeaderNames.push_back("t_conv");
    m_forceHeaderNames.push_back("f_x");
    m_forceHeaderNames.push_back("f_x,v");
    m_forceHeaderNames.push_back("f_x,p");
    m_forceHeaderNames.push_back("f_x,c");
    m_forceHeaderNames.push_back("f_y");
    m_forceHeaderNames.push_back("f_y,v");
    m_forceHeaderNames.push_back("f_y,p");
    m_forceHeaderNames.push_back("f_y,c");
    IF_CONSTEXPR(nDim == 3) {
      m_forceHeaderNames.push_back("f_z");
      m_forceHeaderNames.push_back("f_z,v");
      m_forceHeaderNames.push_back("f_z,p");
      m_forceHeaderNames.push_back("f_z,c");
    }
    m_forceHeaderNames.push_back("area");
    if(m_bPower) {
      m_forceHeaderNames.push_back("P_x");
      m_forceHeaderNames.push_back("P_x,v");
      m_forceHeaderNames.push_back("P_x,p");
      m_forceHeaderNames.push_back("P_y");
      m_forceHeaderNames.push_back("P_y,v");
      m_forceHeaderNames.push_back("P_y,p");
      IF_CONSTEXPR(nDim == 3) {
        m_forceHeaderNames.push_back("P_z");
        m_forceHeaderNames.push_back("P_z,v");
        m_forceHeaderNames.push_back("P_z,p");
      }
    }

    m_noForceDataFields = (MInt)m_forceHeaderNames.size();
    m_forceCounter = 0;

    const MInt noWalls = m_windowInfo->m_auxDataWindowIds.size();
    if(m_forceAsciiOutputInterval > 0 && noWalls * m_noForceDataFields > 0) {
      mAlloc(m_forceData, m_forceAsciiOutputInterval, noWalls * m_noForceDataFields, "m_forceData", F0, AT_);
    }

    // Allocate memory for integral coefficients
    if(noWalls * m_noForceDataFields > 0) {
      mAlloc(m_forceCoef, noWalls * m_noForceDataFields, "m_forceCoef", F0, AT_);
    }
  }


  IF_CONSTEXPR(nDim == 3) {
    if(m_bForceLineAverage) {
      // compute Domain Width for Averaging
      computeDomainWidth();
    }
  }

  if(domainId() == 0 && m_forceAsciiOutputInterval != 0) {
    // we need to decide how many files to open
    const MInt noWalls = m_windowInfo->m_auxDataWindowIds.size();

    m_lastForceOutputTimeStep = -1;
    m_lastForceComputationTimeStep = -1;

    // create file header if file does not exist
    for(MInt i = 0; i < noWalls; ++i) {
      stringstream iWall;
      iWall << i;
      MString filename = "./forces." + iWall.str() + ".dat";


      if(FILE* file = fopen(filename.c_str(), "r")) {
        fclose(file);
      } else {
        FILE* f_forces;
        f_forces = fopen(filename.c_str(), "a+");

        fprintf(f_forces,
                "# Force coefficient file, the force coefficents in the 2 or 3 space directions\n"
                "# are listed in columns, the subscript v denotes a viscous force, p a pressure force,\n"
                "# and c the compressible contribution of the stress tensor\n");
        fprintf(f_forces, "# 1:%s ", m_forceHeaderNames[0].c_str());
        for(MInt j = 1; j < (MInt)m_forceHeaderNames.size(); j++) {
          fprintf(f_forces, " %d:%s ", (j + 1), m_forceHeaderNames[j].c_str());
        }
        fprintf(f_forces, "\n");
        fclose(f_forces);
      }
    }
  }
}

/**
 *  \brief   Function to compute the force coefficient cl, split
 *           split into the viscous part cLv and the pressure part cLp
 *  \authors Marian Albers, Pascal Meysonnat
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::computeForceCoef() {
  const MInt noWalls = m_windowInfo->m_auxDataWindowIds.size();
  MPI_Allreduce(MPI_IN_PLACE, m_forceCoef, noWalls * m_noForceDataFields, MPI_DOUBLE, MPI_SUM, m_StructuredComm, AT_,
                "MPI_IN_PLACE", "m_forceCoef");

  if(m_bForceLineAverage) {
    for(MInt i = 0; i < noWalls * m_noForceDataFields; ++i) {
      m_forceCoef[i] /= m_globalDomainWidth;
    }
  }
}


/**
 * \brief    Function to compute the coefficient, split
 *           split into the viscous part cLv and the pressure part cLp
 *           The ROOT version is faster due to an MPI_Reduce instead
 *           of an MPI_Allreduce, but only root rank has data
 * \author Marian Albers
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::computeForceCoefRoot() {
  const MInt noWalls = m_windowInfo->m_auxDataWindowIds.size();

  MFloatScratchSpace force(noWalls * m_noForceDataFields, AT_, "force");
  for(MInt i = 0; i < noWalls * m_noForceDataFields; ++i) {
    force(i) = m_forceCoef[i];
  }


  MPI_Reduce(force.begin(), m_forceCoef, noWalls * m_noForceDataFields, MPI_DOUBLE, MPI_SUM, 0, m_StructuredComm, AT_,
             "force", "forceCoef");
  if(m_bForceLineAverage) {
    for(MInt i = 0; i < noWalls * m_noForceDataFields; ++i) {
      m_forceCoef[i] /= m_globalDomainWidth;
    }
  }
}


template <MInt nDim>
void FvStructuredSolver<nDim>::computeAuxData() {
  if(m_bForce && m_bPower) {
    computeFrictionPressureCoef(true);
  } else {
    computeFrictionPressureCoef(false);
  }

  if(m_bForce) computeForceCoef();
}


template <MInt nDim>
void FvStructuredSolver<nDim>::computeAuxDataRoot() {
  if(m_bForce && m_bPower) {
    computeFrictionPressureCoef(true);
  } else {
    computeFrictionPressureCoef(false);
  }

  if(m_bForce) computeForceCoefRoot();
}
///////////////////////////////////////////////////////////////////////////////
/// AUX DATA ENDS /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// TODO_SS labels:FV SVD stuff belongs into some math cpp file
////////////////////////////////////////////////////////////////////////////////
/// SVD STUFF //////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// this one is for the singularities
template <MInt nDim>
MFloat FvStructuredSolver<nDim>::computeRecConstSVD(const MInt ijk, const MInt noNghbrIds, MInt* nghbr, MInt ID,
                                                    MInt sID, MFloatScratchSpace& tmpA, MFloatScratchSpace& tmpC,
                                                    MFloatScratchSpace& weights, const MInt recDim) {
  if(noNghbrIds == 0) return F0;

  const MFloat normalizationFactor = 1 / 0.01; // reduces the condition number of the eq system

  for(MInt n = 0; n < noNghbrIds; n++) {
    MInt nghbrId = nghbr[n];
    MFloat dx[nDim];
    for(MInt i = 0; i < nDim; i++) {
      dx[i] = (m_cells->coordinates[i][nghbrId] - m_grid->m_coordinates[i][ijk]) * normalizationFactor;
    }

    tmpA(n, 0) = F1 * normalizationFactor;
    for(MInt i = 0; i < nDim; i++) {
      tmpA(n, i + 1) = dx[i];
    }
  }

  maia::math::invert(tmpA, weights, tmpC, noNghbrIds, recDim);

  // condition number could be calculated using:
  // condNum = svd.singularValues()(0)/svd.singularValues()(size-1);
  // but would double the computational effort...

  for(MInt n = 0; n < noNghbrIds; n++) {
    for(MInt i = 0; i < nDim + 1; i++) {
      m_singularity[sID].ReconstructionConstants[i][ID + n] = tmpC(i, n) * normalizationFactor;
    }
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// SVD STUFF ENDS /////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


template <MInt nDim>
void FvStructuredSolver<nDim>::exchange() {
  exchange(m_sndComm, m_rcvComm);
}

/**
 * \brief Parallel exchange of primitive variables between partitions with MPI
 *
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::exchange(std::vector<std::unique_ptr<StructuredComm<nDim>>>& sndComm,
                                        std::vector<std::unique_ptr<StructuredComm<nDim>>>& rcvComm) {
  RECORD_TIMER_START(m_timers[Timers::Exchange]);


  ///////////////////////////////////////////////
  ///////////// NORMAL EXCHANGE /////////////////
  ///////////////////////////////////////////////
  if(noDomains() > 1) {
    std::vector<MPI_Request> sndRequests;
    std::vector<MPI_Request> rcvRequests;
    std::vector<MPI_Status> sndStatus;
    std::vector<MPI_Status> rcvStatus;
    sndRequests.reserve(sndComm.size());
    rcvRequests.reserve(rcvComm.size());

    RECORD_TIMER_START(m_timers[Timers::Gather]);
    gather(false, sndComm);
    RECORD_TIMER_STOP(m_timers[Timers::Gather]);

    RECORD_TIMER_START(m_timers[Timers::Send]);
    send(false, sndComm, sndRequests);
    RECORD_TIMER_STOP(m_timers[Timers::Send]);

    RECORD_TIMER_START(m_timers[Timers::Receive]);
    receive(false, rcvComm, rcvRequests);
    RECORD_TIMER_STOP(m_timers[Timers::Receive]);

    RECORD_TIMER_START(m_timers[Timers::SendWait]);
    sndStatus.resize(sndRequests.size());
    MPI_Waitall(sndRequests.size(), &sndRequests[0], &sndStatus[0], AT_);
    RECORD_TIMER_STOP(m_timers[Timers::SendWait]);

    RECORD_TIMER_START(m_timers[Timers::ReceiveWait]);
    rcvStatus.resize(rcvRequests.size());
    MPI_Waitall(rcvRequests.size(), &rcvRequests[0], &rcvStatus[0], AT_);
    RECORD_TIMER_STOP(m_timers[Timers::ReceiveWait]);

    RECORD_TIMER_START(m_timers[Timers::Scatter]);
    scatter(false, rcvComm);
    RECORD_TIMER_STOP(m_timers[Timers::Scatter]);
  }

  ///////////////////////////////////////////////
  ///////////// PERIODIC EXCHANGE ///////////////
  ///////////////////////////////////////////////

  for(MInt periodicDir = 0; periodicDir < nDim; periodicDir++) {
    std::vector<MPI_Request> sndRequests;
    std::vector<MPI_Request> rcvRequests;
    std::vector<MPI_Status> sndStatus;
    std::vector<MPI_Status> rcvStatus;
    sndRequests.reserve(sndComm.size());
    rcvRequests.reserve(rcvComm.size());

    m_currentPeriodicDirection = periodicDir;
    RECORD_TIMER_START(m_timers[Timers::Gather]);
    gather(true, sndComm);
    RECORD_TIMER_STOP(m_timers[Timers::Gather]);

    RECORD_TIMER_START(m_timers[Timers::Send]);
    send(true, sndComm, sndRequests);
    RECORD_TIMER_STOP(m_timers[Timers::Send]);

    RECORD_TIMER_START(m_timers[Timers::Receive]);
    receive(true, rcvComm, rcvRequests);
    RECORD_TIMER_STOP(m_timers[Timers::Receive]);

    RECORD_TIMER_START(m_timers[Timers::SendWait]);
    sndStatus.resize(sndRequests.size());
    MPI_Waitall(sndRequests.size(), &sndRequests[0], &sndStatus[0], AT_);
    RECORD_TIMER_STOP(m_timers[Timers::SendWait]);

    RECORD_TIMER_START(m_timers[Timers::ReceiveWait]);
    rcvStatus.resize(rcvRequests.size());
    MPI_Waitall(rcvRequests.size(), &rcvRequests[0], &rcvStatus[0], AT_);
    RECORD_TIMER_STOP(m_timers[Timers::ReceiveWait]);

    RECORD_TIMER_START(m_timers[Timers::Scatter]);
    scatter(true, rcvComm);
    RECORD_TIMER_STOP(m_timers[Timers::Scatter]);
  }

  RECORD_TIMER_STOP(m_timers[Timers::Exchange]);
}


template <MInt nDim>
void FvStructuredSolver<nDim>::send(const MBool periodicExchange,
                                    std::vector<std::unique_ptr<StructuredComm<nDim>>>& sndComm,
                                    std::vector<MPI_Request>& sndRequests) {
  for(auto& snd : sndComm) {
    if(periodicExchange && skipPeriodicDirection(snd)) continue;

    MPI_Request request{};
    const MInt tag = domainId() + (snd->tagHelper) * noDomains();
    const MInt err = MPI_Isend((void*)&snd->cellBuffer[0], snd->cellBufferSize, MPI_DOUBLE, snd->nghbrId, tag,
                               m_StructuredComm, &request, AT_, "snd->cellBuffer");
    sndRequests.push_back(request);
    if(err) cout << "rank " << domainId() << " sending throws error " << endl;
  }
}

template <MInt nDim>
void FvStructuredSolver<nDim>::receive(const MBool periodicExchange,
                                       std::vector<std::unique_ptr<StructuredComm<nDim>>>& rcvComm,
                                       std::vector<MPI_Request>& rcvRequests) {
  for(auto& rcv : rcvComm) {
    if(periodicExchange && skipPeriodicDirection(rcv)) continue;

    MPI_Request request{};
    const MInt tag = rcv->nghbrId + (rcv->tagHelper) * noDomains();
    const MInt err = MPI_Irecv((void*)&rcv->cellBuffer[0], rcv->cellBufferSize, MPI_DOUBLE, rcv->nghbrId, tag,
                               m_StructuredComm, &request, AT_, "rcv->cellBuffer");
    rcvRequests.push_back(request);
    if(err) cout << "rank " << domainId() << " sending throws error " << endl;
  }
}


template <MInt nDim>
MBool FvStructuredSolver<nDim>::skipPeriodicDirection(unique_ptr<StructuredComm<nDim>>& comm) {
  const MInt currentDirection = (comm->bcId - 4401) / 2;
  return ((MBool)(m_currentPeriodicDirection - currentDirection));
}

template <MInt nDim>
MBool FvStructuredSolver<nDim>::isPeriodicComm(unique_ptr<StructuredComm<nDim>>& comm) {
  if(comm->commType == PERIODIC_BC || comm->commType == PERIODIC_BC_SINGULAR) {
    return true;
  } else {
    return false;
  }
}


/// Compute right-hand side.
/// moved from methods
/// \author Sven Berger
/// \tparam nDim
template <MInt nDim>
void FvStructuredSolver<nDim>::rhs() {
  TRACE();

  resetRHS();

  Muscl();

  if(!m_euler) {
    RECORD_TIMER_START(m_timers[Timers::ViscousFlux]);
    viscousFlux();
    RECORD_TIMER_STOP(m_timers[Timers::ViscousFlux]);
  }

  if(m_considerVolumeForces) {
    setVolumeForce();
    computeVolumeForces();
  }

  if(m_blockType == "porous") {
    if(m_rans)
      computePorousRHS(true);
    else
      computePorousRHS(false);
  }
}

/// Apply boundary condition
/// moved from methods
/// \author Sven Berger
/// \tparam nDim
template <MInt nDim>
void FvStructuredSolver<nDim>::rhsBnd() {
  RECORD_TIMER_START(m_timers[Timers::UpdateSponge]);
  updateSpongeLayer();
  RECORD_TIMER_STOP(m_timers[Timers::UpdateSponge]);
}

template <MInt nDim>
void FvStructuredSolver<nDim>::lhsBnd() {
  computePrimitiveVariables();

  exchange();

  if(m_zonal) {
    computeCumulativeAverage(false);
    if(globalTimeStep % m_zonalExchangeInterval == 0 && m_RKStep == 0) {
      spanwiseAvgZonal(m_zonalSpanwiseAvgVars);
      zonalExchange();
    }
  }

  RECORD_TIMER_START(m_timers[Timers::BoundaryCondition]);
  applyBoundaryCondition();
  RECORD_TIMER_STOP(m_timers[Timers::BoundaryCondition]);
}

template <MInt nDim>
MBool FvStructuredSolver<nDim>::solutionStep() {
  RECORD_TIMER_START(m_timers[Timers::Run]);
  RECORD_TIMER_START(m_timers[Timers::MainLoop]);

  MBool step = false;
  
  RECORD_TIMER_START(m_timers[Timers::MLCoupling]);

  // Copy solver double* U/V/W to float input buffers
  MLong totalCells = static_cast<MLong>(m_nCells[0]) * static_cast<MLong>(m_nCells[1]) * static_cast<MLong>(m_nCells[2]);
  MFloat* src[3] = {
    m_cells->pvariables[PV->U],
    m_cells->pvariables[PV->V],
    m_cells->pvariables[PV->W]
  };
  for (int f = 0; f < 3; ++f) {
    for (MLong i = 0; i < totalCells; ++i) {
      m_mlInputBuf[f][i] = static_cast<float>(src[f][i]);
    }
  }

  // Snapshot: capture "sent" fields (before ML inference)
  snapshot::write_step(
    src[0], src[1], src[2],
    m_nCells[0], m_nCells[1], m_nCells[2],
    m_nOffsetCells[0], m_nOffsetCells[1], m_nOffsetCells[2],
    globalTimeStep, "sent");

  int delta = m_mlCoupler->step();

  if (delta > 0) {
    // Copy float output buffers back to solver double* fields
    for (int f = 0; f < 3; ++f) {
      for (MLong i = 0; i < totalCells; ++i) {
        src[f][i] = static_cast<MFloat>(m_mlOutputBuf[f][i]);
      }
    }

    // Snapshot: capture "received" fields (after ML inference)
    snapshot::write_step(
      src[0], src[1], src[2],
      m_nCells[0], m_nCells[1], m_nCells[2],
      m_nOffsetCells[0], m_nOffsetCells[1], m_nOffsetCells[2],
      globalTimeStep, "received");

    m_physicalTime += delta * m_timeStep * m_timeRef;
    m_time += delta * m_timeStep;
    globalTimeStep += delta;
    step = true;
  }
  RECORD_TIMER_STOP(m_timers[Timers::MLCoupling]);

  if (delta == 0) {
    rhs();
    rhsBnd();
    RECORD_TIMER_START(m_timers[Timers::RungeKutta]);
    step = rungeKuttaStep();
    RECORD_TIMER_STOP(m_timers[Timers::RungeKutta]);
    RECORD_TIMER_START(m_timers[Timers::SetTimeStep]);
    if(step) setTimeStep();
    RECORD_TIMER_STOP(m_timers[Timers::SetTimeStep]);
    lhsBnd();
  }

  RECORD_TIMER_STOP(m_timers[Timers::MainLoop]);
  RECORD_TIMER_STOP(m_timers[Timers::Run]);
  return step;
}


/**
 * \brief: Performs the post time step
 * \author Thomas Hoesgen
 */
template <MInt nDim>
void FvStructuredSolver<nDim>::postTimeStep() {
  TRACE();
  RECORD_TIMER_START(m_timers[Timers::Run]);
  RECORD_TIMER_START(m_timers[Timers::MainLoop]);
  m_timeStepConverged = maxResidual();
  RECORD_TIMER_STOP(m_timers[Timers::MainLoop]);
  RECORD_TIMER_STOP(m_timers[Timers::Run]);
}


template <MInt nDim>
void FvStructuredSolver<nDim>::cleanUp() {
  RECORD_TIMER_START(m_timers[Timers::Run]);
  // write a solution file before the solution run ends

  if(m_lastOutputTimeStep != globalTimeStep) {
    if(m_forceAsciiOutputInterval != 0) {
      saveForcesToAsciiFile(true);
    }
    if(m_pointsToAsciiOutputInterval != 0) {
      savePointsToAsciiFile(true);
    }
    saveSolution(true);
    saveAverageRestart();
  }

  this->postprocessPostSolve();
  RECORD_TIMER_STOP(m_timers[Timers::Run]);
}


// Explicit instantiations for 2D and 3D
template class FvStructuredSolver<2>;
template class FvStructuredSolver<3>;
