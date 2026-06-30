################################################################################
# Set supported compilers
set(HOST_SUPPORTED_COMPILERS "GNU" "Intel")

################################################################################
# Set default compiler
set(HOST_DEFAULT_COMPILER "GNU")

################################################################################
# Set executable for each supported compiler (as 'HOST_COMPILER_<compiler>'))
set(HOST_COMPILER_EXE_Intel "mpiicpc")
set(HOST_COMPILER_EXE_GNU "mpicxx")

################################################################################
# Set default settings
# Include directories

set(INCLUDE_DIRS
)

# Library directories

set(LIBRARY_DIRS
)

################################################################################
# Set additional include/library directories or libraries for certain compilers

set(INCLUDE_DIRS_Intel
	    "~/libraries/fftw_intel/include"
)

set(LIBRARY_DIRS_Intel
	    "~/libraries/fftw_intel/lib"
)

# Library names
set(LIBRARY_NAMES
    "fftw3_mpi"
    "fftw3"
    "pnetcdf"
)

set(HOST_CONFIGURE_GNU
    #"module load GCC/11.3.0 OpenMPI"
    #"module load PnetCDF"
    #"module load FFTW.MPI"
    #"module load CMake"
)

if(WITH_HDF5)
    set(HOST_CONFIGURE_GNU
        #"module load GCC/11.3.0 OpenMPI"
        #"module load PnetCDF"
        #"module load FFTW.MPI"
        #"module load CMake"
        #"module load HDF5"
        #"module load Szip"
    )
endif()

set(HOST_CONFIGURE_Intel
    #"module load intel/2022a impi"
    #"module load PnetCDF"
    #"module load CMake"
)

if(WITH_HDF5)
   set(HOST_CONFIGURE_INTEL
       #"module load LIBRARIES"
       #"module load hdf5/1.10.4"
       #"module load intelmkl"
   )
endif()

# HDF5
if(WITH_HDF5)
    if(WITH_GNU)
#  # note: if HDF5 module is loaded on CLAIX, it will define HDF5_DIR
        #set(INCLUDE_DIRS ${INCLUDE_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/HDF5/1.14.5-gompi-2024a/include")
        #set(INCLUDE_DIRS ${INCLUDE_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Szip/2.1.1-GCCcore-13.3.0/include")
        #set(LIBRARY_DIRS ${LIBRARY_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/HDF5/1.14.5-gompi-2024a/lib")
        #set(LIBRARY_DIRS ${LIBRARY_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Szip/2.1.1-GCCcore-13.3.0/lib")
        
        set(INCLUDE_DIRS ${INCLUDE_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/HDF5/1.14.3-gompi-2023b/include")
        set(INCLUDE_DIRS ${INCLUDE_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Szip/2.1.1-GCCcore-13.2.0/include")
        set(LIBRARY_DIRS ${LIBRARY_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/HDF5/1.14.3-gompi-2023b/lib")
        set(LIBRARY_DIRS ${LIBRARY_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Szip/2.1.1-GCCcore-13.2.0/lib")
    endif()
    if(WITH_INTEL)
        set(LIBRARY_DIRS ${LIBRARY_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/sapphirerapids/software/HDF5/1.14.0-iimpi-2022a/include")
        set(LIBRARY_DIRS ${LIBRARY_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/sapphirerapids/software/HDF5/1.14.0-iimpi-2022a/lib")
        set(LIBRARY_DIRS ${LIBRARY_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/sapphirerapids/software/Szip/2.1.1-GCCcore-11.3.0/include")
        set(LIBRARY_DIRS ${LIBRARY_DIRS} "/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/sapphirerapids/software/Szip/2.1.1-GCCcore-11.3.0/lib")
    endif()
    set(LIBRARY_NAMES ${LIBRARY_NAMES} "hdf5_hl")
    set(LIBRARY_NAMES ${LIBRARY_NAMES} "hdf5")
    set(LIBRARY_NAMES ${LIBRARY_NAMES} "z")
    set(LIBRARY_NAMES ${LIBRARY_NAMES} "dl")
    set(LIBRARY_NAMES ${LIBRARY_NAMES} "sz")
endif()

# CANTERA
if(WITH_CANTERA)
  find_package(Threads REQUIRED)
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${SRC_DIR_ABS}/../include/canteraSoftLink/include)
  set(LIBRARY_DIRS ${LIBRARY_DIRS} ${SRC_DIR_ABS}/../include/canteraSoftLink/build/lib  )
  set(LIBRARY_NAMES ${LIBRARY_NAMES} "cantera")
  set(CANTERA_DATA ${CANTERA_DATA} ${SRC_DIR_ABS}/../include/canteraSoftLink/build/data)
endif()

################################################################################
# Set additional include/library directories or libraries for certain compilers

################################################################################
# Set additional include/library directories or libraries for PhyDLL

#set(INCLUDE_DIRS ${INCLUDE_DIRS} ${SRC_DIR_ABS}/../../../phydll/phydll/BUILD/include)
#set(LIBRARY_DIRS ${LIBRARY_DIRS} ${SRC_DIR_ABS}/../../../phydll/phydll/BUILD/lib)
#set(LIBRARY_NAMES ${LIBRARY_NAMES} "phydll")

#####################
# Note: CPP-ML-Interface (CMI) libraries (libtorch, AIxeleratorService, phydll,
# cpp_ml_interface_library, HighFive) are now built in-tree via add_subdirectory
# and linked transitively through the cpp_ml_interface_library CMake target.
# No find_library entries needed here.


################################################################################
# Set (if necessary) host-specific default options for the configuration process
# Note: To find out about supported options, check configure.py
set(HOST_DEFAULT_OPTIONS "WITH_HDF5")
set(HOST_DEFAULT_OPTIONS ${HOST_DEFAULT_OPTIONS})

################################################################################
# Set (if necessary) configuration commands that should be executed when
# configuring MAIA.
set(HOST_PRESERVE_ENV "=")

