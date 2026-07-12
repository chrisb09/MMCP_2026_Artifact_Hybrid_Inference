// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

#include "mpioverride.h"
#include <map>
#include "IO/infoout.h"
#include "UTIL/functions.h"
#include "globalvariables.h"

// TO USE THIS FUNCTIONALITY, ENABLE MAIA_MPI_DEBUG IN CONFIG.H //
// You can find details and instructions about the wrapper functions
// in the corresponding header file.

// Allow the use of MPI functions marked deprecated in mpioverride.h in the MPI wrapper functions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

using namespace std;

// keeps track of the number of MPI communicators
map<MPI_Comm, MInt> mpi_dbg_lst{};
MInt mpi_dbg_cnt = 0;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Helper functions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/** \brief Determine communicator ID
 *
 * \author Philipp Schleich
 * \date 24.09.2019
 *
 * \param[in] MPI communicator
 * \return communicator id
 *
 **/
MInt getCommId(const MPI_Comm comm) {
  // Check for nullptr-communicator
  if(comm == MPI_COMM_NULL) {
    return -1;
  }

  MInt id;
  // search communicator in mapping
  auto it = mpi_dbg_lst.find(comm);
  // If communicator cannot be found, it is the world communicator
  if(it == mpi_dbg_lst.end()) {
    //      mTerm (1, AT_, "Error: Communicator not found.");
    id = 0;
  } else {
    id = it->second;
  }

  return id;
}

/** \brief Convert returncode to string
 *
 * \author Philipp Schleich
 * \date 25.09.2019
 *
 * \param[in] returncode as int
 * \return returncode as a string
 *
 **/
MString return2string(const MInt returncode) {
  MString returnString;
  switch(returncode) {
    case MPI_SUCCESS:
      returnString = "MPI_SUCCESS";
      break;
    case MPI_ERR_COMM:
      returnString = "MPI_ERR_COMM";
      break;
    case MPI_ERR_GROUP:
      returnString = "MPI_ERR_GROUP";
      break;
    case MPI_ERR_INTERN:
      returnString = "MPI_ERR_INTERN";
      break;
    case MPI_ERR_ARG:
      returnString = "MPI_ERR_ARG";
      break;
    case MPI_ERR_COUNT:
      returnString = "MPI_ERR_COUNT";
      break;
    case MPI_ERR_TYPE:
      returnString = "MPI_ERR_TYPE";
      break;
    case MPI_ERR_TAG:
      returnString = "MPI_ERR_TAG";
      break;
    case MPI_ERR_RANK:
      returnString = "MPI_ERR_RANK";
      break;
    case MPI_ERR_REQUEST:
      returnString = "MPI_ERR_REQUEST";
      break;
    case MPI_ERR_IN_STATUS:
      returnString = "MPI_ERR_IN_STATUS";
      break;
    case MPI_ERR_BUFFER:
      returnString = "MPI_ERR_BUFFER";
      break;
    case MPI_ERR_OP:
      returnString = "MPI_ERR_OP";
      break;
    case MPI_ERR_ROOT:
      returnString = "MPI_ERR_ROOT";
      break;
    case MPI_ERR_OTHER:
      returnString = "MPI_ERR_OTHER";
      break;
    case MPI_ERR_INFO:
      returnString = "MPI_ERR_INFO";
      break;
    case MPI_ERR_INFO_KEY:
      returnString = "MPI_ERR_INFO_KEY";
      break;
    case MPI_ERR_INFO_VALUE:
      returnString = "MPI_ERR_INFO_VALUE";
      break;
    case MPI_ERR_TRUNCATE:
      returnString = "MPI_ERR_TRUNCATE: Message truncated on receive. Receive buffer too small for message.";
      break;
    default:
      returnString = "UNKNOWN";
  }

  return returnString;
}

/** \brief Print debug output in m_log
 *
 * \author Philipp Schleich
 * \date 24.09.2019
 *
 * \param[in] returncode
 * \param[in] list of admissible returncodes
 * \param[in] number of admissible returncodes
 *
 **/
void debugResult(const MInt result, const MInt returncodes[], const MInt len) {
  MBool found = false;

  // If returncode in list of admissible returncodes, print it
  for(MInt cnt = 0; cnt < len; cnt++) {
    if(result == returncodes[cnt]) {
      m_log << "    - execution code:         ";
      m_log << return2string(result) << endl;
      found = true;
    }
  }

  // If returncode unknown or not in list of admissible returncodes,
  // print unknown
  if(!found) {
    m_log << "    - execution code:         UNKNOWN" << endl;
  }
}

/** \brief Raise and error and terminate if MPI returns an errorcode
 *
 * \author Philipp Schleich
 * \date 24.09.2019
 *
 * \param[in] returncode
 * \param[in] name of the MPI function
 * \param[in] list of admissible returncodes
 * \param[in] number of admissible mpi functions
 * \param[in] location of function (name)
 * \param[in] function-specific additional output
 *            (e.g. name of sent or received variable, size, ...)
 *
 **/
void raiseMPIerror(const MInt result, const MString& mpiFunction, const MInt returncodes[], const MInt len,
                   const MString& name, const MString& customString) {
  const MString errorMsg =
      "MPI error: " + mpiFunction + " called from '" + name + customString + "' returned error code: ";
  // MInt cnt = 1; // MPI_SUCCESS always first one, check only unsuccessful ones
  MBool found = false;

  char error_string[256];
  int length_of_error_string;
  MPI_Error_string(result, error_string, &length_of_error_string);
  std::cerr << "error string " << error_string << std::endl;

  // If returncode in list of admissible returncodes, terminate with
  // the corresponding error
  for(MInt cnt = 1; cnt < len; cnt++) {
    if(result == returncodes[cnt]) {
      TERMM(1, errorMsg + return2string(result));
      found = true;
    }
  }

  // If returncode unknown or not in list of admissible returncodes,
  // terminate with error unknown
  if(!found) {
    TERMM(1, errorMsg + "UNKNOWN Error. MPI Error is " + error_string);
  }
}


/// \brief Return the name of the given MPI datatype
MString mpiTypeName(MPI_Datatype datatype) {
  char name[128];
  int namelen;
  MPI_Type_get_name(datatype, name, &namelen);
  return std::string(name);
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Wrapper functions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline void startDlbIdleTimer(const MString& name) { maia::dlb::g_dlbTimerController.stopLoadStartIdleTimer(name); }

inline void stopDlbIdleTimer(const MString& name) { maia::dlb::g_dlbTimerController.stopIdleStartLoadTimer(name); }

// ----------------------------------------------------------------------------
// MPI Communicator functions
// ----------------------------------------------------------------------------

/** \brief same as MPI_Comm_create, but updates the number of MPI communicators
 *
 * \author Andreas Lintermann
 * \date 07.05.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Comm_create.
 *
 * \param[in] comm the parent communicator
 * \param[in] group MPI group
 * \param[in] newcomm the new MPI communicator
 * \param[in] name the name of the calling function
 * \param[in] varname name of the variable
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm* newcomm, const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Comm_create called from function " << name << endl;
  m_log << "    - variable name:          " << varname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  int result = MPI_Comm_create(comm, group, newcomm);
  // Note: do not change error handling for Paraview, at the moment this results in "MPI_ERR_COMM:
  // invalid communicator" erros when loading multisolver grids (with inactive ranks for a solver)
#ifndef PVPLUGIN
  if(comm != MPI_COMM_NULL && *newcomm != MPI_COMM_NULL) {
    // Set error handling for new communicator
    MPI_Comm_set_errhandler(*newcomm, MPI_ERRORS_RETURN);
  }
#endif

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_GROUP};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);

  const MBool isNullComm = (*newcomm == MPI_COMM_NULL);
  if(result == MPI_SUCCESS && !isNullComm) {
    mpi_dbg_lst.insert(pair<MPI_Comm, MInt>(*newcomm, ++mpi_dbg_cnt));
  }

  m_log << "    - new communicator id:    " << getCommId(*newcomm) << endl;
  m_log << "    - total number of comms.: " << mpi_dbg_lst.size() << endl;
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Comm_create", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Comm_split, but updates the number of MPI communicators
 *
 * \author Andreas Lintermann
 * \date 07.05.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Comm_split.
 *
 * \param[in] comm the parent communicator
 * \param[in] color control of subset assignment
 * \param[in] key control of rank assignment
 * \param[in] newcomm the new MPI communicator
 * \param[in] name the name of the calling function
 * \param[in] varname name of the variable
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm* newcomm, const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Comm_split called from function " << name << endl;
  m_log << "    - variable name:          " << varname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  int result = MPI_Comm_split(comm, color, key, newcomm);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_INTERN};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);

  m_log << "    - new communicator id:    " << getCommId(*newcomm) << endl;
  m_log << "    - total number of comms.: " << mpi_dbg_lst.size() << endl;
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Comm_split", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Comm_free, but updates the number of MPI communicators
 *
 * \author Andreas Lintermann
 * \date 07.05.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Comm_split.
 *
 * \param[in] comm the MPI communicator to free
 * \param[in] name the name of the calling function
 * \param[in] varname name of the variable
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Comm_free(MPI_Comm* comm, const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Comm_free called from " << name << endl;
  m_log << "    - variable name:          " << varname << endl;

  // search communicator in mapping
  map<MPI_Comm, MInt>::iterator it = mpi_dbg_lst.find(*comm);
  if(it == mpi_dbg_lst.end()) {
    TERMM(1, "Error: Communicator to free not found.");
  }

  MInt id = it->second;
  mpi_dbg_lst.erase(it);

  m_log << "    - communicator id:        " << id << endl;
  m_log << "    - total number of comms.: " << mpi_dbg_lst.size() << endl;
#endif

  int result = MPI_Comm_free(comm);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Comm_free", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Comm_group
 *
 * \author Philipp Schleich
 * \date 23.09.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Comm_split.
 *
 * \param[in] comm the MPI communicator to free
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Comm_group(MPI_Comm comm, MPI_Group* group, const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Comm_group called from " << name << endl;
  m_log << "    - variable name:          " << varname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  int result = MPI_Comm_group(comm, group);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Comm_group", returncodes, 3, name, customString);
  }

  return result;
}

// ----------------------------------------------------------------------------
// Point-to-point communication
// ----------------------------------------------------------------------------


/** \brief same as MPI_Send
 *
 * \author Philipp Schleich
 * \date 03.06.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Send.
 *
 * \additional param[in] name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, const MString& name,
             const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Send called from " << name << endl;
  m_log << "    - variable name / type:   " << varname << " / " << mpiTypeName(datatype) << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
  m_log << "    - destination / tag:      " << dest << " / " << tag << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Send(buf, count, datatype, dest, tag, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_TAG, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 6);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Send", returncodes, 6, name, customString);
  }

  return result;
}

/** \brief same as MPI_Isend
 *
 * \author Philipp Schleich
 * \date 03.06.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Isend.
 *
 * \additional param[in] name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request,
              const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Isend called from " << name << endl;
  m_log << "    - variable name / type:   " << varname << " / " << mpiTypeName(datatype) << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
  m_log << "    - destination / tag:      " << dest << " / " << tag << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Isend(buf, count, datatype, dest, tag, comm, request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE,
                              MPI_ERR_TAG, MPI_ERR_RANK, MPI_ERR_INTERN};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 7);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Isend", returncodes, 7, name, customString);
  }

  return result;
}

/** \brief same as MPI_Issend
 *
 * \author Philipp Schleich
 * \date 03.06.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Issend.
 *
 * \additional param[in] name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Issend(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
               MPI_Request* request, const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Issend called from " << name << endl;
  m_log << "    - variable name / type:   " << varname << " / " << mpiTypeName(datatype) << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
  m_log << "    - destination / tag:      " << dest << " / " << tag << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Issend(buf, count, datatype, dest, tag, comm, request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE,
                              MPI_ERR_TAG, MPI_ERR_RANK, MPI_ERR_INTERN};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 7);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Issend", returncodes, 7, name, customString);
  }

  return result;
}

/** \brief same as MPI_Recv
 *
 * \author Philipp Schleich
 * \date 04.06.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Recv.
 *
 * \additional param[in] name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status,
             const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Recv called from " << name << endl;
  m_log << "    - variable name / type:   " << varname << " / " << mpiTypeName(datatype) << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
  m_log << "    - source / tag:           " << source << " / " << tag << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Recv(buf, count, datatype, source, tag, comm, status);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_TAG, MPI_ERR_RANK};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 6);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Recv", returncodes, 6, name, customString);
  }

  return result;
}

/** \brief same as MPI_Irecv
 *
 * \author Philipp Schleich
 * \date 04.06.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Irecv.
 *
 * \additional param[in] name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request* request,
              const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Irecv called from " << name << endl;
  m_log << "    - variable name / type:   " << varname << " / " << mpiTypeName(datatype) << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
  m_log << "    - source / tag:           " << source << " / " << tag << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Irecv(buf, count, datatype, source, tag, comm, request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE,
                              MPI_ERR_TAG, MPI_ERR_RANK, MPI_ERR_INTERN};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 7);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Irecv", returncodes, 7, name, customString);
  }

  return result;
}

/** \brief same as MPI_Send_init
 *
 * \author Philipp Schleich
 * \date 06.06.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Send_init.
 *
 * \additional param[in] name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Send_init(const void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
                  MPI_Request* request, const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Send_init called from " << name << endl;
  m_log << "    - variable name:          " << varname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Send_init(buf, count, datatype, dest, tag, comm, request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE,
                              MPI_ERR_TAG, MPI_ERR_RANK, MPI_ERR_INTERN};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 7);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Send_init", returncodes, 7, name, customString);
  }

  return result;
}

/** \brief same as MPI_Recv_init
 *
 * \author Philipp Schleich
 * \date 06.06.2019
 *
 * Also delivers output to m_log on the execution status of MPI_Recv_init.
 *
 * \additional param[in] name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Recv_init(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* request,
                  const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Recv_init called from " << name << endl;
  m_log << "    - variable name:          " << varname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Recv_init(buf, count, datatype, dest, tag, comm, request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE,
                              MPI_ERR_TAG, MPI_ERR_RANK, MPI_ERR_INTERN};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 7);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Recv_init", returncodes, 7, name, customString);
  }

  return result;
}

// ----------------------------------------------------------------------------
// Wait / Test
// ----------------------------------------------------------------------------


/** \brief same as MPI_Wait
 * \author Philipp Schleich
 * \date 18.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Wait(MPI_Request* request, MPI_Status* status, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Wait called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Wait(request, status);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_REQUEST, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Wait", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Waitall
 * \author Philipp Schleich
 * \date 18.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Waitall(int count, MPI_Request* request, MPI_Status* status, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Waitall called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Waitall(count, request, status);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_REQUEST, MPI_ERR_ARG, MPI_ERR_IN_STATUS};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 4);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Waitall", returncodes, 4, name, customString);
  }

  return result;
}


/** \brief same as MPI_Waitsome
 * \author Philipp Schleich
 * \date 18.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int* outcount, int array_of_indices[],
                 MPI_Status array_of_statuses[], const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Waitsome called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_REQUEST, MPI_ERR_ARG, MPI_ERR_IN_STATUS};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 4);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Waitsome", returncodes, 4, name, customString);
  }

  return result;
}

/** \brief same as MPI_Test
 * \author Philipp Schleich
 * \date 18.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Test(MPI_Request* request, int* flag, MPI_Status* status, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Test called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Test(request, flag, status);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_REQUEST, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Test", returncodes, 3, name, customString);
  }

  return result;
}

// ----------------------------------------------------------------------------
// Collectives
// ----------------------------------------------------------------------------

/** \brief same as MPI_Barrier
 * \author Philipp Schleich
 * \date 05.07.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Barrier(MPI_Comm comm, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Barrier called from " << name << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Barrier(comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Barrier", returncodes, 2, name, customString);
  }

  return result;
}


/** \brief same as MPI_Reduce
 * \author Philipp Schleich
 * \date 06.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Reduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
               const MString& name, const MString& sndvarname, const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Reduce called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Reduce", returncodes, 5, name, customString);
  }

  return result;
}

/** \brief same as MPI_Allreduce
 * \author Philipp Schleich
 * \date 06.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                  const MString& name, const MString& sndvarname, const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Allreduce called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER, MPI_ERR_OP};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 6);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Allreduce", returncodes, 6, name, customString);
  }

  return result;
}


/** \brief same as MPI_Iallreduce
 * \author Philipp Schleich
 * \date 23.09.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                   MPI_Request* request, const MString& name, const MString& sndvarname, const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Iallreduce called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER, MPI_ERR_OP};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 6);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Iallreduce", returncodes, 6, name, customString);
  }

  return result;
}

/** \brief same as MPI_Scatter
 * \author Philipp Schleich
 * \date 07.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                MPI_Datatype recvtype, int root, MPI_Comm comm, const MString& name, const MString& sndvarname,
                const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Scatter called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Scatter", returncodes, 5, name, customString);
  }

  return result;
}

/** \brief same as MPI_Scatterv
 * \author Miro Gondrum
 * \date 10.01.2022
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Scatterv(const void* sendbuf, const int sendcount[], const int displs[], MPI_Datatype sendtype, void* recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm, const MString& name,
                 const MString& sndvarname, const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Scatter called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Scatterv(sendbuf, sendcount, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Scatter", returncodes, 5, name, customString);
  }

  return result;
}

/** \brief same as MPI_Bcast
 * \author Philipp Schleich
 * \date 07.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] varname name of the variable
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, const MString& name,
              const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Bcast called from " << name << endl;
  m_log << "    - variable name:          " << varname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Bcast(buffer, count, datatype, root, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER, MPI_ERR_ROOT};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 6);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Bcast", returncodes, 6, name, customString);
  }

  return result;
}

/** \brief same as MPI_Ibcast
 * \author Sven Berger
 * \date 30.09.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] varname name of the variable
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Ibcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request* request,
               const MString& name, const MString& varname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Ibcast called from " << name << endl;
  m_log << "    - variable name:          " << varname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Ibcast(buffer, count, datatype, root, comm, request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER, MPI_ERR_ROOT};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 6);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for variable '" + varname;
    raiseMPIerror(result, "MPI_Ibcast", returncodes, 6, name, customString);
  }

  return result;
}

/** \brief same as MPI_Gather
 * \author Philipp Schleich
 * \date 07.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Gather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
               MPI_Datatype recvtype, int root, MPI_Comm comm, const MString& name, const MString& sndvarname,
               const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Gather called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Gather", returncodes, 5, name, customString);
  }

  return result;
}

/** \brief same as MPI_Gatherv
 * \author Philipp Schleich
 * \date 17.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
                const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm, const MString& name,
                const MString& sndvarname, const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Gatherv called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 4);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Gatherv", returncodes, 4, name, customString);
  }


  return result;
}

/** \brief same as MPI_Allgather
 * \author Philipp Schleich
 * \date 07.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                  MPI_Datatype recvtype, MPI_Comm comm, const MString& name, const MString& sndvarname,
                  const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Allgather called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Allgather", returncodes, 5, name, customString);
  }

  return result;
}

/** \brief same as MPI_Allgatherv
 * \author Philipp Schleich
 * \date 07.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Allgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
                   const int displs[], MPI_Datatype recvtype, MPI_Comm comm, const MString& name,
                   const MString& sndvarname, const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Allgatherv called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 4);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Allgatherv", returncodes, 4, name, customString);
  }

  return result;
}

/** \brief same as MPI_Alltoall
 * \author Philipp Schleich
 * \date 17.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Alltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                 MPI_Datatype recvtype, MPI_Comm comm, const MString& name, const MString& sndvarname,
                 const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Alltoall called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COMM, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Alltoall", returncodes, 5, name, customString);
  }

  return result;
}

/** \brief same as MPI_Alltoallv
 * \author Philipp Schleich
 * \date 17.06.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Alltoallv(const void* sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype,
                  void* recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
                  const MString& name, const MString& sndvarname, const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Alltoallv called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Alltoallv", returncodes, 5, name, customString);
  }

  return result;
}


/** \brief same as MPI_Exscan
 * \author Philipp Schleich
 * \date 04.07.2019
 *
 * \param[in] name the name of the calling function
 * \param[in] sndvarname name of the sendbuf
 * \param[in] rcvvarname name of the recvbuf
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Exscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
               const MString& name, const MString& sndvarname, const MString& rcvvarname) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Exscan called from " << name << endl;
  m_log << "    - snd variable name:      " << sndvarname << endl;
  m_log << "    - rcv variable name:      " << rcvvarname << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_COUNT, MPI_ERR_TYPE, MPI_ERR_BUFFER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString = "' for sent variable '" + sndvarname + "' for received variable '" + rcvvarname;
    raiseMPIerror(result, "MPI_Exscan", returncodes, 5, name, customString);
  }

  return result;
}

// ----------------------------------------------------------------------------
// Derived Datatypes
// ----------------------------------------------------------------------------

/** \brief same as MPI_Type_commit
 * \author Philipp Schleich
 * \date 19.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Type_commit(MPI_Datatype* datatype, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Type_commit called from " << name << endl;
#endif

  int result = MPI_Type_commit(datatype);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_TYPE};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Type_commit", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief same as MPI_Type_free
 * \author Philipp Schleich
 * \date 19.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Type_free(MPI_Datatype* datatype, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Type_free called from " << name << endl;
#endif

  int result = MPI_Type_free(datatype);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_TYPE, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Type_free", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Type_create_hindexed
 * \author Philipp Schleich
 * \date 19.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Type_create_hindexed(int count, const int array_of_solverlengths[], const MPI_Aint array_of_displacements[],
                             MPI_Datatype oldtype, MPI_Datatype* newtype, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Type_create_hindexed called from " << name << endl;
#endif

  int result = MPI_Type_create_hindexed(count, array_of_solverlengths, array_of_displacements, oldtype, newtype);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_TYPE, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Type_create_hindexed", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Type_contiguous
 * \author Philipp Schleich
 * \date 19.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Type_contiguous(int count, MPI_Datatype old_type, MPI_Datatype* new_type_p, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Type_contiguous from " << name << endl;
#endif

  int result = MPI_Type_contiguous(count, old_type, new_type_p);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_TYPE, MPI_ERR_COUNT, MPI_ERR_INTERN};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 4);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Type_contiguous", returncodes, 4, name, customString);
  }

  return result;
}


// ----------------------------------------------------------------------------
// MPI Group
// ----------------------------------------------------------------------------

/** \brief same as MPI_Group_incl
 * \author Philipp Schleich
 * \date 19.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Group_incl(MPI_Group group, int n, const int ranks[], MPI_Group* newgroup, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Group_incl called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Group_incl(group, n, ranks, newgroup);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_GROUP, MPI_ERR_ARG, MPI_ERR_INTERN, MPI_ERR_RANK};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Group_incl", returncodes, 5, name, customString);
  }

  return result;
}

/** \brief same as MPI_Group_free
 * \author Philipp Schleich
 * \date 19.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Group_free(MPI_Group* group, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Group_free called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Group_free(group);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Group_free", returncodes, 2, name, customString);
  }

  return result;
}


// ----------------------------------------------------------------------------
// MISC
// ----------------------------------------------------------------------------
/** \brief same as MPI_Start
 * \author Philipp Schleich
 * \date 05.07.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Start(MPI_Request* request, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Start called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Start(request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_REQUEST};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Start", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief same as MPI_Startall
 * \author Philipp Schleich
 * \date 05.07.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Startall(int count, MPI_Request array_of_requests[], const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Startall called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Startall(count, array_of_requests);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_REQUEST, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Startall", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Get_count
 * \author Philipp Schleich
 * \date 26.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Get_count(const MPI_Status* status, MPI_Datatype datatype, int* count, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Get_count called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Get_count(status, datatype, count);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_TYPE};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Get_count", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief same as MPI_Get_address
 * \author Philipp Schleich
 * \date 26.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Get_address(const void* location, MPI_Aint* address, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Get_address called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Get_address(location, address);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Get_address", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief same as MPI_Abort
 * \author Philipp Schleich
 * \date 24.09.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Abort(MPI_Comm comm, int errorcode, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Abort called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Abort(comm, errorcode);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Abort", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief same as MPI_Request_free
 * \author Philipp Schleich
 * \date 24.09.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Request_free(MPI_Request* request, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Request_free called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Request_free(request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Request_free", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief same as MPI_cancel
 * \author Philipp Schleich
 * \date 24.09.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Cancel(MPI_Request* request, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Cancel called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Cancel(request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Cancel", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief probe MPI to get status without actually receiving the message
 * \author Tim Wegmann
 * \date 23.12.2021
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status* status, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Get_count called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Probe(source, tag, comm, status);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_TYPE};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Probe", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief Iprobe MPI to get status without actually receiving the message
 * \author Tim Wegmann
 * \date 23.12.2021
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Iprobe(int source, int tag, MPI_Comm comm, int* flag, MPI_Status* status, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Get_count called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Iprobe(source, tag, comm, flag, status);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_TYPE};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Iprobe", returncodes, 2, name, customString);
  }

  return result;
}

// ----------------------------------------------------------------------------
// MPI Info
// ----------------------------------------------------------------------------
/** \brief same as MPI_Info_create
 * \author Philipp Schleich
 * \date 26.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Info_create(MPI_Info* info, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Info_create called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Info_create(info);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Info_create", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief same as MPI_Info_free
 * \author Philipp Schleich
 * \date 26.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Info_free(MPI_Info* info, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Info_free called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Info_free(info);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER, MPI_ERR_INFO};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Info_free", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Info_get
 * \author Philipp Schleich
 * \date 26.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Info_get(MPI_Info info, const char* key, int valuelen, char* value, int* flag, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Info_get called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Info_get(info, key, valuelen, value, flag);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER, MPI_ERR_INFO_KEY, MPI_ERR_ARG, MPI_ERR_INFO_VALUE};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 5);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Info_get", returncodes, 5, name, customString);
  }

  return result;
}

/** \brief same as MPI_Info_get_nthkey
 * \author Philipp Schleich
 * \date 26.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Info_get_nthkey(MPI_Info info, int n, char* key, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Info_get_nthkey called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Info_get_nthkey(info, n, key);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Info_get_nthkey", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_Info_get_nkeys
 * \author Philipp Schleich
 * \date 26.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Info_get_nkeys(MPI_Info info, int* nkeys, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Info_get_nkeys called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Info_get_nkeys(info, nkeys);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 2);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Info_get_nkeys", returncodes, 2, name, customString);
  }

  return result;
}

/** \brief same as MPI_Info_get_valuelen
 * \author Philipp Schleich
 * \date 26.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_Info_get_valuelen(MPI_Info info, const char* key, int* valuelen, int* flag, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_Info_get_valuelen called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_Info_get_valuelen(info, key, valuelen, flag);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_OTHER, MPI_ERR_INFO_KEY};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_Info_get_valuelen", returncodes, 3, name, customString);
  }

  return result;
}

// ----------------------------------------------------------------------------
// MPI File
// ----------------------------------------------------------------------------
/** \brief same as MPI_File_open
 * \author Philipp Schleich
 * \date 27.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_File_open(MPI_Comm comm, const char* filename, int amode, MPI_Info info, MPI_File* mpi_fh,
                  const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_File_open called from " << name << endl;
  m_log << "    - communicator id:        " << getCommId(comm) << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_File_open(comm, filename, amode, info, mpi_fh);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_File_open", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_File_seek
 * \author Philipp Schleich
 * \date 27.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_File_seek(MPI_File mpi_fh, MPI_Offset offset, int whence, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_File_seek called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_File_seek(mpi_fh, offset, whence);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_File_seek", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_File_close
 * \author Philipp Schleich
 * \date 27.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_File_close(MPI_File* mpi_fh, const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_File_close called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_File_close(mpi_fh);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_File_close", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_File_write_shared
 * \author Philipp Schleich
 * \date 27.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_File_write_shared(MPI_File mpi_fh, const void* buf, int count, MPI_Datatype datatype, MPI_Status* status,
                          const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_File_write_shared called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_File_write_shared(mpi_fh, buf, count, datatype, status);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_File_write_shared", returncodes, 3, name, customString);
  }

  return result;
}

/** \brief same as MPI_File_iwrite_shared
 * \author Philipp Schleich
 * \date 27.06.2019
 *
 * \param[in] name the name of the calling function
 *
 * \return the return status of the MPI call
 *
 **/
int MPI_File_iwrite_shared(MPI_File mpi_fh, const void* buf, int count, MPI_Datatype datatype, MPI_Request* request,
                           const MString& name) {
#ifdef MAIA_MPI_DEBUG
  m_log << "  + MPI_File_iwrite_shared called from " << name << endl;
#endif

  startDlbIdleTimer(name);
  int result = MPI_File_iwrite_shared(mpi_fh, buf, count, datatype, request);
  stopDlbIdleTimer(name);

  const MInt returncodes[] = {MPI_SUCCESS, MPI_ERR_COMM, MPI_ERR_ARG};

#ifdef MAIA_MPI_DEBUG
  // Debug output
  debugResult(result, returncodes, 3);
#endif

  // Check return code
  if(result != MPI_SUCCESS) {
    const MString customString;
    raiseMPIerror(result, "MPI_File_iwrite_shared", returncodes, 3, name, customString);
  }

  return result;
}

// Enable compiler flag "-Wdeprecated-declarations" again
#pragma GCC diagnostic pop
