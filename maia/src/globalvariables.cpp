// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

////////////////////////////////////////////////////////////////////////////////
/// \file \brief This files declares all global variables of MAIA
///
/// The global variables can be initialized _only_ in this file, and have to be
/// defined in the globalvariables.h file preceeded by the extern keyword.
////////////////////////////////////////////////////////////////////////////////
#include <ostream>
#include <vector>
#include "INCLUDE/maiatypes.h"
#include "IO/infoout.h"
#include "MEMORY/genericobject.h"



#ifdef WITH_PHYDLL_DIRECT
#include "ML/mlCouplerPhyDLL.h"
#endif

////////////////////////////////////////////////////////////////////////////////
/// Variables to initialize:
MInt GenericObject::objectCounter = 0;
////////////////////////////////////////////////////////////////////////////////
// Memory allocation
namespace maia {
namespace alloc {
std::vector<GenericObject*> g_allocatedObjects;
MLong g_allocatedBytes = 0;
MLong g_maxAllocatedBytes = 0;
} // namespace alloc
} // namespace maia
////////////////////////////////////////////////////////////////////////////////
MInt g_timeSteps;
MInt globalTimeStep;
MInt g_restartTimeStep;
MBool g_dynamicLoadBalancing;
MBool g_splitMpiComm;
MBool g_multiSolverGrid = false;


namespace maia {
namespace dlb {
DlbTimerController g_dlbTimerController;
} // namespace dlb
} // namespace maia

////////////////////////////////////////////////////////////////////////////////
// other timers
MInt g_t_readGeomFile = -1;

////////////////////////////////////////////////////////////////////////////////
// timer collections
std::vector<std::pair<MString, MInt>> g_tc_geometry;

////////////////////////////////////////////////////////////////////////////////
// Logging
#ifndef PVPLUGIN
InfoOutFile m_log;
InfoOutFile maia_res;
#else
std::ostream& m_log = std::cerr;
std::ostream& maia_res = std::cerr;
#endif
std::ostream cerr0(nullptr);
////////////////////////////////////////////////////////////////////////////////
