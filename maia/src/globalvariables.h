// Copyright (C) 2024 The m-AIA AUTHORS
//
// This file is part of m-AIA (https://git.rwth-aachen.de/aia/m-AIA/m-AIA)
//
// SPDX-License-Identifier: LGPL-3.0-only

#ifndef MAIA_GLOBAL_VARIABLES_
#define MAIA_GLOBAL_VARIABLES_
////////////////////////////////////////////////////////////////////////////////
/// \file \brief This file contains all global variables of MAIA.
///
/// The global variables are defined once in the .cpp file (where they can be
/// initialized), and once in the .h file preceeded by the extern keyword.
////////////////////////////////////////////////////////////////////////////////
#include <ostream>
#include <vector>
#include "COMM/globalmpiinfo.h"
#include "COMM/mpioverride.h"
#include "INCLUDE/maiatypes.h"
#include "UTIL/dlbtimer.h"



////////////////////////////////////////////////////////////////////////////////
class InfoOutFile;
class GenericObject;
class DlbTimerController;
////////////////////////////////////////////////////////////////////////////////
// Memory allocation
namespace maia {
namespace alloc {
extern std::vector<GenericObject*> g_allocatedObjects;
extern MLong g_allocatedBytes;
extern MLong g_maxAllocatedBytes;
} // namespace alloc
} // namespace maia
////////////////////////////////////////////////////////////////////////////////
extern MInt g_timeSteps;
extern MInt globalTimeStep;
extern MInt g_restartTimeStep;
extern MBool g_dynamicLoadBalancing;
extern MBool g_splitMpiComm;
// Temporary variable for use with true multi-solver simulations
// Warning: Will be removed in the near future once all components of MAIA can handle multi-solver simulations
extern MBool g_multiSolverGrid;


namespace maia {
namespace dlb {
extern DlbTimerController g_dlbTimerController;
} // namespace dlb
} // namespace maia
////////////////////////////////////////////////////////////////////////////////
// other timers
extern MInt g_t_readGeomFile;

////////////////////////////////////////////////////////////////////////////////
// timer collections
extern std::vector<std::pair<MString, MInt>> g_tc_geometry;

////////////////////////////////////////////////////////////////////////////////
// Logging
#ifndef PVPLUGIN
extern InfoOutFile m_log;
extern InfoOutFile maia_res;
#else
extern std::ostream& m_log;
extern std::ostream& maia_res;
#endif
extern std::ostream cerr0;
////////////////////////////////////////////////////////////////////////////////

#endif
