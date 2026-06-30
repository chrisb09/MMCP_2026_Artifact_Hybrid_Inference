
cmake_minimum_required(VERSION 2.8.1)

set(CMAKE_VERBOSE_MAKEFILE OFF)
################################################################################
# Set source files for MAIA library
# Note: if you add solver/component-specific source files for a component that
# can be disabled using configure.py, please also add the corresponding file to
# the corresponding _to_remove list below

file(GLOB src_lib CONFIGURE_DEPENDS
  ${SRC_DIR_ABS}/*.cpp
  ${SRC_DIR_ABS}/*/*.cpp
  )
set(_to_remove
  ${SRC_DIR_ABS}/maia.cpp
  ${SRC_DIR_ABS}/nvhpcbugfix.cpp
  ${SRC_DIR_ABS}/COUPLER/lslb2d.cpp
  ${SRC_DIR_ABS}/COUPLER/lslb3d.cpp
  ${SRC_DIR_ABS}/LB/lbsolver.cpp
  ${SRC_DIR_ABS}/LB/lbsolverdxqy.cpp
  ${SRC_DIR_ABS}/FV/fvcartesiansolverxd.cpp
  ${SRC_DIR_ABS}/FV/fvcartesianbndrycndxd.cpp
  )
list(REMOVE_ITEM src_lib ${_to_remove})
unset(_to_remove)

################################################################################
# NO NEED TO CHANGE ANYTHING BEYOND THIS LINE
################################################################################

################################################################################
# Disable certain components of MAIA
# If a component is to be disabled, the corresponding macro is defined and the
# source files that should not be compiled are removed from the complete list
# above. If you add a new file to any of the components that may be disabled to
# src_lib, make sure you also remove them again here.

# Disable DG solver
if(MAIA_DISABLE_DG)
  add_definitions("-DMAIA_DISABLE_DG")
  set(_to_remove
      ${SRC_DIR_ABS}/cartesiangrid_inst_dg.cpp
      ${SRC_DIR_ABS}/DG/dgcartesiansolver_inst_2d_linearscalaradv.cpp
      ${SRC_DIR_ABS}/DG/dgcartesiansolver_inst_3d_linearscalaradv.cpp
      ${SRC_DIR_ABS}/DG/dgcartesiansolver_inst_2d_acousticperturb.cpp
      ${SRC_DIR_ABS}/DG/dgcartesiansolver_inst_3d_acousticperturb.cpp
      ${SRC_DIR_ABS}/DG/dgcartesianinterpolation.cpp
      ${SRC_DIR_ABS}/DG/dgcartesiansyseqnacousticperturb.cpp
      ${SRC_DIR_ABS}/DG/dgcartesianbcfactoryacousticperturb.cpp
      ${SRC_DIR_ABS}/DG/dgcartesianbcfactorylinearscalaradv.cpp
      ${SRC_DIR_ABS}/COUPLER/dgccacousticperturbsourcefiles.cpp
      ${SRC_DIR_ABS}/DG/dgcartesiangalerkinprojection.cpp
  )
  list(REMOVE_ITEM src_lib ${_to_remove})
  unset(_to_remove)
endif()

# Disable LB solver
if(MAIA_DISABLE_LB)
  add_definitions("-DMAIA_DISABLE_LB")
  set(_to_remove
    ${SRC_DIR_ABS}/cartesiangrid_inst_lb.cpp
    ${SRC_DIR_ABS}/LB/lbsolver_inst.cpp
    ${SRC_DIR_ABS}/LB/lbsolverdxqy_inst.cpp
    ${SRC_DIR_ABS}/LB/lbbndcnd.cpp
    ${SRC_DIR_ABS}/LB/lbbndcnddxqy_inst.cpp
    ${SRC_DIR_ABS}/LB/lbinterface.cpp
    ${SRC_DIR_ABS}/LB/lbinterfacedxqy_inst.cpp
    ${SRC_DIR_ABS}/LB/lbinterfacecell.cpp
    ${SRC_DIR_ABS}/LB/lbparentcell.cpp
    ${SRC_DIR_ABS}/LB/lbsrctermcontroller.cpp
    ${SRC_DIR_ABS}/LB/lbsrcterm.cpp
    ${SRC_DIR_ABS}/LB/lbsrctermsgs.cpp
  )
  list(REMOVE_ITEM src_lib ${_to_remove})
  unset(_to_remove)
endif()

# Disable structured solver
if(MAIA_DISABLE_STRUCTURED)
  add_definitions("-DMAIA_DISABLE_STRUCTURED")
  set(_to_remove
    ${SRC_DIR_ABS}/FV/fvstructuredbndrycnd3d.cpp
    ${SRC_DIR_ABS}/FV/fvstructuredbndrycnd2d.cpp    
    ${SRC_DIR_ABS}/FV/fvstructuredbndrycnd.cpp    
    ${SRC_DIR_ABS}/FV/fvstructuredsolver3d.cpp
    ${SRC_DIR_ABS}/FV/fvstructuredsolver2d.cpp    
    ${SRC_DIR_ABS}/FV/fvstructuredsolver.cpp
    ${SRC_DIR_ABS}/FV/fvstructuredsolver3drans.cpp
    ${SRC_DIR_ABS}/FV/fvstructuredsolver2drans.cpp    
    ${SRC_DIR_ABS}/FV/fvstructuredzonalbc.cpp    
    ${SRC_DIR_ABS}/FV/fvstructuredsolverwindowinfo.cpp
    ${SRC_DIR_ABS}/FV/fvstructuredwindowmapping.cpp
    ${SRC_DIR_ABS}/FV/fvstructuredsolvercommunicationhandle.cpp
    ${SRC_DIR_ABS}/FV/fvstructuredcomm.cpp
    ${SRC_DIR_ABS}/FV/fvstructuredpostprocessing.cpp    
    ${SRC_DIR_ABS}/GRID/structuredpartition.cpp
    ${SRC_DIR_ABS}/GRID/structuredgrid.cpp    
  )
  list(REMOVE_ITEM src_lib ${_to_remove})
  unset(_to_remove)
endif()


################################################################################
# Add include directories for the compiler
# MAIA source directory itself
include_directories(${CMAKE_SOURCE_DIR}/${INCLUDE_DIR})
# All external libraries as system headers to avoid compiler warnings
include_directories(SYSTEM ${INCLUDE_DIRS})
include_directories(${SRC_DIR_ABS})
################################################################################
# Add meta information to be displayed as command line options
execute_process(COMMAND
    "git" "describe" "--abbrev=8" "--dirty" "--always" "--tags"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE maia_version_string
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
if("${maia_version_string}" STREQUAL "")
    set(maia_version_string "n/a")
endif()
set(maia_compiler_string "${MAIA_COMPILER}")
set(maia_compiler_version_string "${CMAKE_CXX_COMPILER_VERSION}")
set(maia_build_type_string "${CMAKE_BUILD_TYPE}")
set(maia_host_string "${MAIA_HOST}")
set(info_properties
    "-DMAIA_VERSION_STRING='${maia_version_string}'"
    "-DMAIA_COMPILER_STRING='${maia_compiler_string}'"
    "-DMAIA_COMPILER_VERSION_STRING='${maia_compiler_version_string}'"
    "-DMAIA_BUILD_TYPE_STRING='${maia_build_type_string}'"
    "-DMAIA_HOST_STRING='${maia_host_string}'")
join(_info_properties_string " " ${info_properties})
set_source_files_properties(${SRC_DIR_ABS}/environment.cpp
                            ${SRC_DIR_ABS}/IO/parallelio_pnetcdf.cpp
                            ${SRC_DIR_ABS}/IO/parallelio_hdf5.cpp                            
                            ${SRC_DIR_ABS}/IO/infoout.cpp
                            PROPERTIES COMPILE_FLAGS ${_info_properties_string})

################################################################################
# Allow users to figure out host as a macro
if(DEFINED ${MAIA_HOST})
  string(TOUPPER ${MAIA_HOST} _host_upper)
  add_definitions("-DMAIA_HOST_${_host_upper}")
  unset(_host_upper)
endif()

################################################################################
# Selective support for certain libraries and features

# Enable special libraries
if(WITH_ZLIB)
  add_definitions("-DWITH_ZLIB")
endif()
if(WITH_HDF5)
  add_definitions("-DWITH_HDF5")
endif()
if(WITH_LIKWID)
  add_definitions("-DWITH_LIKWID")
  add_definitions("-DLIKWID_PERFMON")
endif()
add_definitions("-DUSE_CPP_ML_INTERFACE")
add_subdirectory("${CMAKE_SOURCE_DIR}/CPP-ML-Interface" "${CMAKE_BINARY_DIR}/CPP-ML-Interface")
include_directories("${CMAKE_SOURCE_DIR}/CPP-ML-Interface/include")
include_directories("${CMAKE_SOURCE_DIR}/CPP-ML-Interface-old/include") # snapshot_writer.hpp (kept until Phase 8)
include_directories("${CMAKE_BINARY_DIR}/CPP-ML-Interface") # generated_registry.hpp
list(APPEND LIBRARIES cpp_ml_interface_library tomlplusplus::tomlplusplus)

add_definitions("-DWITH_SCOREP")

# Enable debug functions via command line
if(ENABLE_DEBUG_FUNCTION)
  add_definitions("-DMAIA_DEBUG_FUNCTION")
endif()

# Enable backtracing
if(ENABLE_BACKTRACE)
  add_definitions("-DENABLE_BACKTRACE")
  # The following definitions are necessary for the LLVM includes
  add_definitions("-D__STDC_LIMIT_MACROS")
  add_definitions("-D__STDC_CONSTANT_MACROS")
endif()
if(USE_IO_BACKEND AND NOT USE_IO_BACKEND STREQUAL "_default")
  add_definitions("-DPARALLELIO_DEFAULT_BACKEND=${USE_IO_BACKEND}")
endif()

# Disable Output
if(DISABLE_OUTPUT)
  add_definitions("-DDISABLE_OUTPUT")
endif()
################################################################################
# TODO: WAR: bugfix for nvhpc parallel stl
if(MAIA_COMPILER STREQUAL "NVHPC" AND ENABLE_PSTL AND MAIA_HOST STREQUAL "AIA")
  # Add stdpar to linker options
  set(LIBRARIES "-stdpar=gpu" ${LIBRARIES})
  # << WAR: bugfix for nvhpc parallel stl
  message(STATUS "WARNING: NVHPC WAR: overriding free() function")
  add_library(NVHPCBUGFIX SHARED ${SRC_DIR_ABS}/nvhpcbugfix.cpp)
  set(LIBRARIES NVHPCBUGFIX ${LIBRARIES})
  # WAR >>
endif()
################################################################################

# Create object library, which serves as a container for all .o files
add_library(${OBJ_LIB_TARGET} OBJECT ${src_lib})
set_target_properties(${OBJ_LIB_TARGET} PROPERTIES OUTPUT_NAME ${LIB_NAME})

# Create executable target
add_executable(${BIN_TARGET} $<TARGET_OBJECTS:${OBJ_LIB_TARGET}> ${SRC_DIR_ABS}/maia.cpp)
set_target_properties(${BIN_TARGET} PROPERTIES OUTPUT_NAME ${BIN_NAME})
target_link_libraries(${BIN_TARGET} ${LIBRARIES})

# If desired, create library target and link executable to it
if(CREATE_LIB)
  # Create library target
  add_library(${LIB_TARGET} $<TARGET_OBJECTS:${OBJ_LIB_TARGET}>)
  set_target_properties(${LIB_TARGET} PROPERTIES OUTPUT_NAME ${LIB_NAME})
  # Create additional static library if desired
  if(BUILD_SHARED_LIBS AND ALWAYS_BUILD_STATIC)
    add_library(${LIB_TARGET}_static STATIC $<TARGET_OBJECTS:${OBJ_LIB_TARGET}>)
    set_target_properties(${LIB_TARGET}_static PROPERTIES OUTPUT_NAME ${LIB_NAME})
  endif()
endif()

################################################################################
# Add LD_LIBRARY_PATH to rpath if desired
if(LD_LIBRARY_PATH_TO_RPATH AND NOT "${LD_LIBRARY_PATH}" STREQUAL "")
  set_target_properties(${BIN_TARGET} PROPERTIES
                        LINK_FLAGS "-Wl,-rpath,${LD_LIBRARY_PATH}")
endif()
