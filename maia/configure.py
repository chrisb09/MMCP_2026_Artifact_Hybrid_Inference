#!/usr/bin/env python3

# Import needed libraries
import sys
import subprocess
import shutil
import shlex
import re
import os
import errno
import hashlib

# Directory with cmake files (relative to MAIA_ROOT)
CMAKE_DIR = 'cmake'

# Name of cmake precompile helper (relative to MAIA_ROOT/CMAKE_DIR)
CMAKE_PRECOMPILE = 'Configure.cmake'

# Directory with compiler files (relative to MAIA_ROOT)
COMPILERS_DIR = 'auxiliary/compilers'

# Directory with host files (relative to MAIA_ROOT)
HOSTS_DIR = 'auxiliary/hosts'

# Configuration file suffix
CONFIG_FILE_SUFFIX = '.cmake'

# CMake minimum version
CMAKE_MINIMUM_VERSION = '2.8.1'

# Source code directory
SRC_DIR = 'src'

# Documentation directory
DOC_DIR = 'doc'

INCLUDE_DIR = 'include'

# Submodule directory for cantera git files
CANTERA_SUBMODULE_DIR = 'include/cantera'

CANTERA_INSTALL_DIR = 'include/canteraInstall'

# Script to add environment variables to Makefile
ADDENV_SCRIPT = 'auxiliary/addenv.py'

# Script to delete build files before running configure.py
RESET_SCRIPT = 'auxiliary/reset.sh'

# Supported build systems (must be a dict with parameter names as keys, CMake
# build system name as value)
SUPPORTED_BUILD_SYSTEMS = {'make': "Unix Makefiles", 'ninja': "Ninja"}

# Build system commands
BUILD_SYSTEM_COMMANDS = {'make': 'make', 'ninja': 'ninja'}

# Build system distclean target
DISTCLEAN_TARGET = 'distclean'

# Default build system to use
DEFAULT_BUILD_SYSTEM = 'make'

# Build system environment variable
BUILD_SYSTEM_ENV = 'MAIA_BUILD_SYSTEM'

# Color usage environment variable
COLORS_ENV = 'MAIA_CONFIGUREPY_COLORS'

# supported parallel STL versions for NVHPC
SUPPORTED_PSTL = {'multicore', 'volta', 'turing', 'ampere'}
DEFAULT_PSTL = 'multicore'

# Supported instrumentation systems (must be a dict with parameter names as
# keys, instrumentation command as value)
SUPPORTED_INSTRUMENTATIONS = {'scorep': "scorep"}

# Default instrumentation to use
DEFAULT_INSTRUMENTATION = 'scorep'

# Supported instrumentation types (note that not all instrumentation systems
# may support all of the types, see next configuration value)
SUPPORTED_INSTRUMENTATION_TYPES = ['mpi', 'compiler', 'user']

# Set list of supported types by instrumentation system (must be a dict with
# instrumentation systems as names, list of instrumentation types as values)
INSTRUMENTATION_BY_SYSTEM = {'scorep': ['mpi', 'compiler', 'user']}

# Default instrumentation type
DEFAULT_INSTRUMENTATION_TYPE = 'compiler'

# Supported BigData I/O backend types
SUPPORTED_IO_BACKENDS = ['pnetcdf', 'hdf5']

# Translate backend name to actual class
IO_BACKEND_CLASSES = {
    'pnetcdf': 'BigDataPNetcdf', 'hdf5': 'BigDataHdf5'}

# Set default backend to avoid overriding MAIA if not specifically requested by
# user
DEFAULT_IO_BACKEND = '_default'

# Set list of components that can be disabled
DISABLE_COMPONENTS = ['avg', 'dg', 'lbm', 'particle', 'structured']


################################################################################
# NO NEED TO EDIT ANYTHING BEYOND THIS POINT
################################################################################

# Import legacy file if running on host with Python < 2.7
if sys.version_info < (2, 7):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "postprocessing"))
    import LEGACY_PYTHON.argparse as argparse
else:
    import argparse


class CMakeConfigure:
    def __init__(self):
        # Get current working directory + full path to MAIA
        self.maia_root = os.path.dirname(os.path.realpath(__file__))
        self.cwd = os.getcwd()
        self.doc_dir = os.path.join(self.maia_root, DOC_DIR)
        self.configure_file_name = os.path.basename(__file__)
        self.reset = False

        # Reset member variables
        self.logfile = None
        self.compiler = None
        self.build_type = None
        self.build_system = None
        self.build_dir = None

        # Reset configure variables
        self.conf_host = None
        self.conf_compilers = []
        self.conf_default_compiler = None
        self.conf_cuda_version = "11.5"
        self.conf_build_types = dict()
        self.conf_csa_dir = None
        self.conf_support_csa = dict()
        self.conf_have_openmp = dict()
        self.conf_have_pstl = dict()
        self.conf_openmp_flags = dict()
        self.conf_pstl_flags = dict()
        self.conf_compiler_exe = dict()
        self.conf_default_build_type = dict()
        self.conf_configure_host = []
        self.conf_configure = dict()
        self.conf_preserve_env_host = []
        self.conf_preserve_env = dict()
        self.conf_default_options = []

    def path_to_cantera_library(self):

        if self.conf_host == 'AIA':
            self.cantera_path = '/aia/opt/cantera/'
        #elif self.conf_host == 'AIA' and self.build_type == 'sanitize_adress':
            #self.cantera_path = '/aia/opt/canteraSanitizeAdress/'
        else:
            self.log(' The path to the',
                     newline=False)
            self.log(' Cantera', newline=False,
                     highlight='error', raw=True)
            self.log(' libraries has been defaulted to the cantera submodule directory (include/cantera). Remember to install it manually or automatically with',
                     newline=False, raw=True)
            self.log(' --install-cantera.\n',
                     newline=False, highlight='error', raw=True)
            self.cantera_path = os.path.join(
                self.maia_root, CANTERA_SUBMODULE_DIR)

    def cantera_setup(self):

        # Updates Cantera submodule, cleans build files and uninstalls Cantera to avoid conflicts
        cmds_pre = 'git submodule update --init --recursive; scons clean -C include/cantera; scons build -c -C include/cantera'
        # Only necessary on the HAWK, loads python3
        cmds_hawk = 'module load python/3'
        # Builds and installs Cantera
        cmds = 'scons build -j 12 -C include/cantera python_package=none prefix=' + \
            os.path.join(self.maia_root, CANTERA_INSTALL_DIR) + \
            '; scons install -C include/cantera'
        cmds_sanitize = 'scons build -j 12 -C include/cantera debug_linker_flags="-fsanitize=address" cc_flags="-fsanitize=address" python_package=none prefix=' + \
            os.path.join(self.maia_root, CANTERA_INSTALL_DIR) + \
            '; scons install -j 12 -C include/cantera'

        if self.cantera_path == 'default':
            self.path_to_cantera_library()

        # if CANTERA is installed, change path to cantera install directory
        if self.install_cantera is True:
            self.cantera_path = os.path.join(
                self.maia_root, CANTERA_SUBMODULE_DIR)

        # Create a symbolic link inside 'src/CANTERA' that points to the CANTERA library to be used by the compiler
        if self.cantera_path is not None:
            try:
                os.symlink(self.cantera_path,
                           os.path.join(self.maia_root, INCLUDE_DIR, 'canteraSoftLink'))
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(os.path.join(self.maia_root,
                                           INCLUDE_DIR, 'canteraSoftLink'))
                    os.symlink(self.cantera_path,
                               os.path.join(self.maia_root, INCLUDE_DIR, 'canteraSoftLink'))
                else:
                    raise e

        # If Cantera is used, but no link to the libraries exists, and no Path is given, a link is created
        if self.with_cantera is True and self.cantera_path is None:
            self.path_to_cantera_library()
            if not os.path.exists(os.path.join(self.maia_root,
                                               INCLUDE_DIR, 'canteraSoftLink')):
                os.symlink(self.cantera_path,
                           os.path.join(self.maia_root, INCLUDE_DIR, 'canteraSoftLink'))

        # Compiles and installs cantera from the git submodule
        if self.install_cantera is True:

            try:
                os.remove("./include/cantera/cantera.conf")
            except FileNotFoundError:
                 self.log('Cantera.conf not found.', highlight='info')

            self.log('Fetching and updating cantera submodules...',
                     highlight='info')
            p1 = subprocess.Popen(cmds_pre, shell=True, stdout=subprocess.PIPE)

            sStdout1, sStdErr1 = p1.communicate()

            if self.conf_host == 'HAWK':
                self.log('Preparing HAWK-specific commands...',
                         highlight='info')
                subprocess.Popen(cmds_hawk, shell=True, stdout=subprocess.PIPE)

            self.log('Building and installing Cantera...', highlight='info')
            if self.build_type == 'sanitize_address': 
                self.log('-fsanitize=address flag passed to Cantera compiler and linker...', highlight='info')
                p2 = subprocess.Popen(cmds_sanitize, shell=True)
                # stdout=subprocess.PIPE
            else:
                p2 = subprocess.Popen(cmds, shell=True)

            sStdout2, sStdErr2 = p2.communicate()

    def run(self):
        with open('config.log', 'w') as self.logfile:
            # Ensure that we have a good cmake executable on the PATH
            self.check_cmake_version()
            # Get all necessary information on the host and its available
            # compilers, build types etc.
            self.get_configure_information()
            # Parse the command line arguments from the user
            self.parse_command_line()
            # Set git submodules correctly
            self.update_submodules(self.reset)
            # Prepare cantera setup, installation if necessary
            self.cantera_setup()
            # Clean up directory if compiler executable has changed
            self.pre_cleanup()
            # Configure MAIA
            self.run_cmake()

            # Symlink compile_commands.json to root dir
            if self.output_compile_commands:
                os.symlink(os.path.join(self.build_dir, 'compile_commands.json'),
                           os.path.join(self.maia_root, '_compile_commands.json'))
                os.rename(os.path.join(self.maia_root, '_compile_commands.json'),
                          os.path.join(self.maia_root, 'compile_commands.json'))

            # Inform user that we are ready to rumble
            if self.enable_csa:
                # TODO Find better way to create this output - since it is
                # specific to a build mode, it should be located within CMake
                self.log("Run the clang static analyzer by executing "
                         "'./analyze' (all additional arguments are "
                         "forwarded).")
                self.log("MAIA is ready for './analyze'!", highlight='success')
            else:
                self.log("MAIA is ready for {bs}!".format(bs=self.build_system),
                         highlight='success')

    def parse_command_line(self):
        # Get default build system - if the environment variable is set, it
        # overrides the configuration variable above
        if BUILD_SYSTEM_ENV in os.environ:
            default_bs = os.environ[BUILD_SYSTEM_ENV]
        else:
            default_bs = DEFAULT_BUILD_SYSTEM

        # Check if clang's static analyzer is available
        if self.conf_csa_dir:
            csa_supported = ''
        else:
            csa_supported = (" This feature is NOT supported on the current "
                             "host.")

        # Check if HDF5 should be enabled by default
        if 'WITH_HDF5' in self.conf_default_options:
            with_hdf5_by_default = True
        else:
            with_hdf5_by_default = False
            
        # Set command line help epilog message that is displayed at the end
        epilog = ("By default, `configure.py` uses colors to highlight "
                  "important information when used interactively. If this "
                  "is not desired, you can disable colors by setting the "
                  "environment variable `{env}` to 'off'."
                  .format(env=COLORS_ENV))

        # Create the argparse instance and parse the user arguments
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            epilog=epilog)
        parser.add_argument(
            '--reset',
            action='store_true',
            help=("Delete all build-related files before doing anything "
                  "else."))
        parser.add_argument(
            'compiler',
            nargs='?',
            default=None,
            metavar='<COMPILER>',
            help="Specify compiler or '?' for default compiler.")
        parser.add_argument(
            'build_type',
            nargs='?',
            default=None,
            metavar='<BUILD_TYPE>',
            help="Specify build type or '?' for default type.")
        parser.add_argument(
            '--with-zlib',
            action='store_true',
            help="Build MAIA with zlib support.")
        hdf_group = parser.add_mutually_exclusive_group()
        hdf_group.add_argument(
            '--with-hdf5',
            dest='with_hdf5',
            action='store_true', default=(True if with_hdf5_by_default else False),
            help="Link against HDF5 library and enable HDF5-related code.")
        hdf_group.add_argument(
            '--without-hdf5',
            dest='with_hdf5',
            action='store_false',
            help= "Do not link against HDF5 library and disable HDF5-related code.")
        parser.add_argument(
                '--disable-updateGitSubmodules',
                action='store_true',
                help="Disable Git from updating the submodules.")
        parser.add_argument(
            '--with-cantera',
            action='store_true',
            help="Build MAIA with CANTERA support.")
        parser.add_argument(
            '--with-likwid',
            action='store_true',
            help="Build MAIA with likwid support.")
        parser.add_argument(
            '--disable', '-d',
            action='append',
            choices=DISABLE_COMPONENTS,
            help=("Disable certain components of MAIA during compilation."))
        parser.add_argument(
            '--allow-unused',
            action='store_true',
            help="Do not issue errors for 'unused' warnings.")
        parser.add_argument(
            '--build-system',
            choices=SUPPORTED_BUILD_SYSTEMS,
            default=default_bs,
            help="Set build system to use.")
        parser.add_argument(
            '--disable-openmp',
            action='store_true',
            help="Disable OpenMP extensions (enabled by default).")
        parser.add_argument(
            '--enable-pstl',
            choices=SUPPORTED_PSTL,
            help="Enable Parallel STL extensions.")
        parser.add_argument(
            '--with-knlBooster',
            action='store_true',
            help="Enable computations on the KNL Booster of the Jureca (WARNING: Please load the module Architecture/KNL before compiling the code).")
        parser.add_argument(
            '--clang-static-analyzer', '--csa',
            action='store_true',
            help="Compile with clang's static analyzer.{v}".format(
                v=csa_supported))
        parser.add_argument(
            '--backtrace',
            '--bt',
            action='store_true',
            dest="backtrace",
            help="Enable internal backtrace capabilities. "
            "Enabled by default when debug is enabled.")
        parser.add_argument(
            '--no-backtrace',
            '--no-bt',
            action='store_false',
            dest="backtrace",
            help="Disable internal backtrace capabilities. "
            "Useful to get a debug build without backtracing.")
        parser.add_argument(
            '--compile-commands', '--cc',
            action='store_true',
            help="Output compile command database.")
        parser.add_argument(
            '--enable-instrumentation',
            choices=SUPPORTED_INSTRUMENTATIONS,
            nargs='?',
            const=DEFAULT_INSTRUMENTATION,
            help="Instrument MAIA for performance measurements.")
        parser.add_argument(
            '--instrument',
            action='append',
            choices=SUPPORTED_INSTRUMENTATION_TYPES,
            default=[DEFAULT_INSTRUMENTATION_TYPE],
            help=("Specify the kinds of instrumentations that should be " +
                  "employed. You may specify this argument multiple" +
                  " times. This option implies '--enable-instrumentation'."
                  ))
        parser.add_argument(
            '--enable-debug',
            action='store_true',
            help=("Enable the MAIA debugging facilities, "
                  "which might have a negative impact on the performance. "
                  "Please note that it will not tell the compiler to add "
                  "debugging symbols to the binary, it will only enable "
                  "the MAIA-internal debug functions."
                  ))
        parser.add_argument(
            '--io',
            choices=SUPPORTED_IO_BACKENDS,
            default=DEFAULT_IO_BACKEND,
            help=("Specify the default I/O backend that should be used for "
                  "BigData. If you choose '--io hdf5', it implies "
                  "'--with-hdf5'."))
        parser.add_argument(
            '--disable-output',
            action='store_true',
            help=("Disable any kind of output at compile time for" 
                  "meaningful performance measurements on HPC systems"))
        parser.add_argument(
            '--cantera-path',
            nargs='?',
            const='default',
            help=("Specify the path to the CANTERA library."))
        parser.add_argument(
            '--install-cantera',
            action='store_true',
            help=("Installs Cantera in submodule directory."))
        parser.add_argument(
            '--build-dir-name',
            action='store',
            type = str,
            help=("Specify a name for the build directory"))
        parser.add_argument(
            '--build-in-tmp',
            action='store_true',
            help=("Specify that the build directory should be created in /tmp to speed up linking"))

        args = parser.parse_args()

        # If '--reset' was supplied, clean up first
        if args.reset:
            self.execute(os.path.join(self.maia_root, RESET_SCRIPT))
            self.reset = True

        # Save whether to enable the MAIA debug function
        self.enable_debug = args.enable_debug

        # Save whether to enable zlib
        self.with_zlib = args.with_zlib

        # Save whether to enable hdf5
        self.with_hdf5 = args.with_hdf5

        # Git submodules
        self.disable_updateGitSubmodules = args.disable_updateGitSubmodules

        # Save whether to enable CANTERA
        self.with_cantera = args.with_cantera

        # Save the path to CANTERA library
        self.cantera_path = args.cantera_path

        # Save the path to CANTERA library
        self.install_cantera = args.install_cantera

        # Save whether to enable likwid
        self.with_likwid = args.with_likwid

        # Save components that should be disabled
        self.disable_components = args.disable

        # Save if compile commands should be stored
        self.output_compile_commands = args.compile_commands

        # Save which I/O backend should be used by default
        self.io = args.io
        if self.io == 'hdf5' and not self.with_hdf5:
            # If default I/O backend is HDF5, also link against HDF5
            self.with_hdf5 = True
        
        # Save wether output is going to be suppressed 
        self.disable_output = args.disable_output

        # Save whether to demote unused errors to warnings
        self.allow_unused = args.allow_unused

        # Save build type
        self.build_system = args.build_system

        # Check if instrumentation type is supported by system and save
        # instrumentation information
        self.enable_instrumentation = args.enable_instrumentation
        if self.enable_instrumentation is None and len(args.instrument) > 1:
            self.enable_instrumentation = DEFAULT_INSTRUMENTATION
        if self.enable_instrumentation:
            for option in args.instrument:
                if option not in (
                        INSTRUMENTATION_BY_SYSTEM[self.enable_instrumentation]):
                    sys.stderr.write("configure.py: error: instrumentation "
                                     "system '{s}' does not support "
                                     "instrumentations of type '{t}'.\n".format(
                                         s=self.enable_instrumentation,
                                         t=option))
                    sys.exit(2)
            if len(args.instrument) > 1:
                self.instrument = list(set(args.instrument[1:]))
            else:
                self.instrument = list(set(args.instrument))

        # Inform the user about what is going on
        self.log("current host: ", newline=False)
        self.log(" {h}\n".format(h=self.conf_host), raw=True,
                 highlight='info')
        self.slog("parsing command line arguments")

        # Convert compiler from integer (if any) to strings
        try:
            args.compiler = self.conf_compilers[int(args.compiler) - 1]
        except:
            pass

        # Parse the compiler argument
        if args.compiler is None:
            # If compiler is None, show available compilers and quit
            self.log("available compilers and build types:")
            message = '\n'
            for nc, c in enumerate(self.conf_compilers):
                for nb, b in enumerate(self.conf_build_types[c]):
                    message += "  {nc} {nb} -> {c}, {b}\n".format(
                        nb=nb+1, b=b, nc=nc+1, c=c)
                message += '\n'
            self.log(message, raw=True, newline=False, highlight='info')
            self.log("defaults on current host:", newline=False)
            dc = self.conf_default_compiler
            db = self.conf_default_build_type[dc]
            self.log(" ? ? -> {dc}, {db}\n".format(dc=dc, db=db), raw=True,
                     highlight='info')
            self.log("use id or literal name (case-insensitive) to select "
                     "compiler and build type")
            self.log("use '?' to select the default(s)")
            sys.exit(0)
        elif args.compiler == '?':
            # If compiler is '?', use the default compiler for this host
            self.compiler = self.conf_default_compiler
        else:
            # Otherwise check if the selected compiler is available, and if yes,
            # set it to the correct spelling
            real_compiler = None
            for compiler in self.conf_compilers:
                if compiler.lower() == args.compiler.lower():
                    real_compiler = compiler
            if real_compiler is None:
                self.abort(
                    "compiler not available: {c}".format(c=args.compiler))
            else:
                self.compiler = real_compiler
        # Inform user on selected compiler
        self.log("selected compiler: ", newline=False)
        self.log(" {c}\n".format(c=self.compiler), raw=True, highlight='info')

        # Convert build type from integer (if any) to strings
        try:
            args.build_type = self.conf_build_types[self.compiler][
                int(args.build_type) - 1]
        except:
            pass

        # Parse the build type argument
        if args.build_type is None:
            # If build type is None, show available build types and quit
            self.log("available build types:")
            message = '\n'
            for nb, b in enumerate(self.conf_build_types[self.compiler]):
                message += "  {nb} -> {b}\n".format(nb=nb+1, b=b)
            message += '\n'
            self.log(message, raw=True, newline=False, highlight='info')
            self.log("default build type for selected compiler:", newline=False)
            self.log(" ? -> {d}\n".format(
                d=self.conf_default_build_type[self.compiler]), raw=True,
                highlight='info')
            self.log("use id or literal name (case-insensitive) to select "
                     "build type")
            self.log("use '?' to select the default(s)")
            sys.exit(0)
        elif args.build_type == '?':
            # If build type is '?', use the default build type for this compiler
            self.build_type = self.conf_default_build_type[self.compiler]
        else:
            # Otherwise check if the selected build type is available, and if
            # yes, set it to the correct spelling
            real_build_type = None
            for build_type in self.conf_build_types[self.compiler]:
                if build_type.lower() == args.build_type.lower():
                    real_build_type = build_type
            if real_build_type is None:
                self.abort("build type not available: {b}".format(
                    b=args.build_type))
            else:
                self.build_type = real_build_type
        # Inform user on selected build type
        self.log("selected build type:", newline=False)
        self.log(" {b}\n".format(b=self.build_type),
                 raw=True, highlight='info')

        # Automatically enable backtracing with debug builds
        # and save wether to enable backtracing
        self.enable_backtrace = args.backtrace
        if (self.build_type == 'debug'
            and not ("--no-bt" in sys.argv or "--no-backtrace" in sys.argv)
            and "Clang" in self.conf_compilers
                and not "Clang" in self.compiler):  # disable for Clang since this is bugged
            self.enable_backtrace = True

        # Check if backtracing is possible on the current host
        if self.enable_backtrace and "Clang" not in self.conf_compilers:
            self.log("Backtracing is not supported on this host (No Clang installed).",
                     error=True, highlight="error")
            sys.exit(2)

        # Check if OpenMP is supported by the chosen compiler
        self.disable_openmp = args.disable_openmp
        if not self.disable_openmp:
            if not self.conf_have_openmp.get(self.compiler, False):
                self.log("OpenMP is not supported by the selected compiler. And has been disabled!",
                         error=True, highlight='error')
                self.disable_openmp = True
                # sys.exit(2)

        # Check if parallel STL is supported by the chosen compiler
        self.pstl_options = ''
        self.enable_pstl = args.enable_pstl
        if self.enable_pstl is None:
            self.disable_pstl = True
        else:
            self.disable_pstl = False
            if not self.conf_have_pstl.get(self.compiler, False):
                self.log("Parallel STL is not supported by the selected compiler. And has been disabled!",
                         error=True, highlight='error')
                self.disable_pstl = True
            else: # Parallel STL is enabled
                self.pstl_options = '-DMAIA_PSTL '
                supportedMethods='"' + " ".join(SUPPORTED_PSTL) + '"'
                if args.enable_pstl and 'NVHPC' == self.compiler:
                    # Map backend and gpu architecture to NVHPC options:
                    if args.enable_pstl == 'multicore':
                        self.pstl_options += '-stdpar=multicore -DMAIA_NVHPC_PSTL_MULTICORE'
                    elif args.enable_pstl == 'volta':
                        self.pstl_options += '-stdpar=gpu -gpu=cc70,managed,cuda{}'.format(self.conf_cuda_version)
                    elif args.enable_pstl == 'turing':
                        self.pstl_options += '-stdpar=gpu -gpu=cc75,managed,cuda{}'.format(self.conf_cuda_version)
                    elif args.enable_pstl == 'ampere':
                        self.pstl_options += '-stdpar=gpu -gpu=cc80,managed,cuda{}'.format(self.conf_cuda_version)
                    else:
                        if len(args.enable_pstl) == 0:
                          self.abort("parallel STL enabled but --enable-pstl not passed. Use one of the following ones: {}".format(supportedMethods))
                        else:
                          self.abort("parallel STL enabled but mode '{}' is unkown in configure. Use on of the following ones: {}".format(args.enable_pstl, supportedMethods))
                        self.pstl_options += args.enable_pstl
                elif args.enable_pstl and 'GNU' == self.compiler: # GCC parallel STL needs exception enabled
                    if args.enable_pstl == 'multicore':
                      self.pstl_options += ' -fexceptions '
                    else:
                      self.abort("Passing custom PSTL options with an unsupported compiler: {}".format(args.enable_pstl))
                elif not 'GNU' == self.compiler and not args.enable_pstl:
                    self.abort("parallel STL enabled but --enable-pstl not passed")
                self.pstl_options_fmt = '[{}]'.format(self.pstl_options.split('-DMAIA_PSTL ')[1])

        # Check if the current host is Jureca
        self.with_knlBooster = args.with_knlBooster
        if self.with_knlBooster:
            if not self.conf_host == "Jureca":
                self.log("The current host is not Jureca.",
                         error=True, highlight='error')
                sys.exit(2)

       # Inform user of parallelism type
        self.log("selected parallelism type:", newline=False)
        if not self.disable_openmp and not self.disable_pstl:
            self.log(" MPI+OpenMP+PSTL{}\n".format(self.pstl_options_fmt), raw=True, highlight='info')
        elif not self.disable_openmp:
            self.log(" MPI+OpenMP\n", raw=True, highlight='info')
        elif not self.disable_pstl:
            self.log(" MPI+PSTL{}\n".format(self.pstl_options_fmt), raw=True, highlight='info')
        else:
            self.log(" MPI\n", raw=True, highlight='info')

        # Inform user of parallelism type
        if self.with_knlBooster:
            self.log("compile on KNL Booster of Jureca (WARNING: Please load the module Architecture/KNL before compiling the code):", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

       # Inform user iff a different build system is used
        if self.build_system != DEFAULT_BUILD_SYSTEM:
            self.log("selected build system:", newline=False)
            self.log(" {bs}\n".format(bs=self.build_system), raw=True,
                     highlight='info')

        # Inform user iff ZLIB is enabled
        if self.with_zlib:
            self.log("compile with zlib:", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Inform user iff HDF5 is enabled
        if self.with_hdf5:
            self.log("compile with HDF5 support in BigData:",
                     newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Inform user if CANTERA is enabled
        if self.with_cantera:
            self.log("compile with CANTERA:", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Inform user if likwid is enabled
        if self.with_likwid:
            self.log("compile with likwid:", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Inform user iff certain components are disabled
        if self.disable_components:
            self.log("disabled components:",
                     newline=False)
            self.log(" {}\n".format(', '.join(self.disable_components)),
                     raw=True, highlight='info')

        # Inform user iff -Wunused warnings are not counted as errors
        if self.allow_unused:
            self.log("'-Wunused-xxx' is only a warning:", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Inform user iff debug functions are activated
        if self.enable_debug:
            self.log("debug functions:", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Check if clang static analyzer is supported by the current host and
        # the current compiler
        self.enable_csa = args.clang_static_analyzer
        if self.enable_csa:
            if not self.conf_csa_dir:
                self.log("Using the clang static analyzer is not supported on this "
                         "host.",
                         error=True, highlight='error')
                sys.exit(2)
            if not self.conf_support_csa.get(self.compiler, False):
                self.log("Using the clang static analyzer is not supported by the "
                         "selected compiler.",
                         error=True, highlight='error')
                sys.exit(2)
            self.log("clang static analyzer:", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Inform user iff backtracing is enabled
        if self.enable_backtrace:
            self.log("backtracing:", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Inform user iff output of compiler commands is enabled
        if self.output_compile_commands:
            self.log("compile command output:", newline=False)
            self.log(" enabled\n", raw=True, highlight='info')

        # Inform user iff instrumentation is used and which
        if self.enable_instrumentation:
            if self.enable_csa:
                self.log("Cannot use instrumentation while using the clang "
                         "static analyzer.",
                         error=True, highlight='error')
                sys.exit(2)
            self.log("instrument source code:", newline=False)
            self.log(" {i}\n".format(i=self.enable_instrumentation), raw=True,
                     highlight='info')
            self.log("instrumentation type(s):", newline=False)
            self.log(" {t}\n".format(t=', '.join(self.instrument)), raw=True,
                     highlight='info')

        # Inform user iff non-standard I/O backend is used
        if self.io != DEFAULT_IO_BACKEND:
            self.log("I/O backend:", newline=False)
            self.log(" {b}\n".format(b=self.io), raw=True, highlight='info')

        # Inform user that output is suppressed
        if self.disable_output:
            self.log("output: ", newline=False)
            self.log(" disabled\n", raw=True, highlight='error')

        # Set build directory
        forbidden_dir_names =  {'.', './', '/'}
        localBuildDir = '_'.join(("build", self.compiler.lower(), self.build_type))
        if args.build_in_tmp:
            if args.build_dir_name is None:
                # Get first 8 characters of hash of pwd to create unique build directory
                self.build_dir = '_'.join(("/tmp/" + hashlib.sha256(os.getcwd().encode('utf-8')).hexdigest()[:8], self.compiler.lower(), self.build_type))
            else:
                self.abort("--build-in-tmp and --build-dir-name cannot be used together!")
            # check if link already exists or a real directory in its place and remove it
            if os.path.exists(localBuildDir):
                if os.path.islink(localBuildDir):
                    os.remove(localBuildDir)
                else:
                    shutil.rmtree(localBuildDir)
            # check if build dir already exists and remove it
            if os.path.exists(self.build_dir):
                shutil.rmtree(self.build_dir)
            # create link
            os.symlink(self.build_dir, localBuildDir)
        else:
            if args.build_dir_name is None or args.build_dir_name in forbidden_dir_names:
                self.build_dir = localBuildDir
                if os.path.exists(localBuildDir) and os.path.islink(localBuildDir):
                    os.remove(localBuildDir)
            else:
                self.build_dir = args.build_dir_name
                os.symlink(self.build_dir, localBuildDir)

    def update_submodules(self, force):
      """ This function update all git submodules.
      """
      # Change into maia root dir
      os.chdir(self.maia_root)

      if self.disable_updateGitSubmodules:
        self.log("Skip update git submodules")
        return
      else:
        self.log("update git submodules (skip this step with --disable-updateGitSubmodules)")
      cmds = ['git', 'submodule', 'update', '--init']
      if force:
        cmds.append('--force')
      try:
        code = subprocess.call(cmds)
      except Exception:
        code = -1
      if code:
        cmdsUser='"'+(" ".join(cmds))+'"'
        self.abort("ERROR: Git submodules could not be initialized!\n"
            "Try manually checking out the submodules with:\n"
            + cmdsUser)
      # Change back into current working directory
      os.chdir(self.cwd)

    def check_cmake_version(self):
        # Inform the user on what we are doing
        self.slog("checking cmake version")

        # Get CMake version string
        _, o, _ = self.configure('cmake --version', bash=True, critical=True)

        # Parse version
        cmake_version = tuple(map(int, re.findall(r'\d+', o.split(' ')[2])))
        self.slog("cmake version is {v}".format(v='.'.join(
            [str(c) for c in cmake_version])))
        minimum_version = tuple(
            [int(c) for c in CMAKE_MINIMUM_VERSION.split('.')])

        # Abort if version does not meet minimum required
        if minimum_version > cmake_version:
            self.abort("cmake version >= {v} required".format(
                v=CMAKE_MINIMUM_VERSION))

    def get_configure_information(self):
        # Inform the user on what we are doing
        self.slog("getting configure information")

        # Construct command that will give us all needed information
        cmds = ['cmake',
                '-DMAIA_ROOT={d}'.format(d=self.maia_root),
                '-DCMAKE_DIR={d}'.format(d=CMAKE_DIR),
                '-DCOMPILERS_DIR={d}'.format(d=COMPILERS_DIR),
                '-DHOSTS_DIR={d}'.format(d=HOSTS_DIR),
                '-DCONFIG_FILE_SUFFIX={d}'.format(d=CONFIG_FILE_SUFFIX),
                '-P', os.path.join(self.maia_root, CMAKE_DIR, CMAKE_PRECOMPILE)]

        # Run command
        _, _, output = self.configure(cmds, critical=True)
        self.slog("received {n} lines of information".format(
            n=len(output.splitlines())))
        self.slog(output, raw=True)

        # Parse output for information on host, compiler, build types etc.
        self.slog("parsing information...", newline=False)
        for line in output.splitlines():
            tokens = shlex.split(line)
            if len(tokens) == 0:
                continue
            cmd = tokens[0]
            args = tokens[1:]
            if cmd == 'HOST':
                self.conf_host = args[0]
            elif cmd == 'COMPILERS':
                self.conf_compilers = args
            elif cmd == 'DEFAULT_COMPILER':
                self.conf_default_compiler = args[0]
            elif cmd == 'CUDA_VERSION':
                self.conf_cuda_version = args[0]
            elif cmd == 'BUILD_TYPES':
                self.conf_build_types[args[0]] = args[1:]
            elif cmd == 'COMPILER_EXE':
                self.conf_compiler_exe[args[0]] = args[1]
            elif cmd == 'DEFAULT_BUILD_TYPE':
                self.conf_default_build_type[args[0]] = args[1]
            elif cmd == 'HAVE_PSTL':
                self.conf_have_pstl[args[0]] = True
            elif cmd == 'HAVE_OPENMP':
                self.conf_have_openmp[args[0]] = True
            elif cmd == 'SUPPORT_CLANG_STATIC_ANALYZER':
                self.conf_support_csa[args[0]] = True
            elif cmd == 'CLANG_STATIC_ANALYZER_DIR':
                self.conf_csa_dir = args[0]
            elif cmd == 'PSTL_FLAGS':
                self.conf_pstl_flags[args[0]] = args[1:]
            elif cmd == 'OPENMP_FLAGS':
                self.conf_openmp_flags[args[0]] = args[1:]
            elif cmd == 'CONFIGURE_HOST':
                self.conf_configure_host.append(' '.join(args))
            elif cmd == 'CONFIGURE':
                if args[0] not in self.conf_configure:
                    self.conf_configure[args[0]] = []
                self.conf_configure[args[0]].append(' '.join(args[1:]))
            elif cmd == 'PRESERVE_ENV_HOST':
                self.conf_preserve_env_host.append(args[0])
            elif cmd == 'PRESERVE_ENV':
                if args[0] not in self.conf_preserve_env:
                    self.conf_preserve_env[args[0]] = []
                self.conf_preserve_env[args[0]].append(args[1])
            elif cmd == 'DEFAULT_OPTIONS':
                self.conf_default_options = args
            else:
                self.slog("fail\n", raw=True)
                self.abort("information not recognized: {i}".format(i=cmd))
        self.slog("done\n", raw=True)

    def pre_cleanup(self):
        # Get old executable
        self.slog("checking if executable has changed in build system")
        old_compiler_exe = None
        if os.path.isfile('CMakeCache.txt'):
            with open('CMakeCache.txt', 'r') as f:
                for line in f.readlines():
                    if line.startswith('CMAKE_CXX_COMPILER:FILEPATH='):
                        old_compiler_exe = line.split('=')[1].strip()
                        break
        # Check if compiler was found
        if old_compiler_exe is None:
            self.slog("no old compiler found, nothing to do")
        else:
            self.slog("old compiler executable: {e}".format(
                e=old_compiler_exe))
            # Get new executable
            new_compiler_exe = self.conf_compiler_exe[self.compiler]
            self.slog("new compiler executable: {e}".format(
                e=new_compiler_exe))
            # Compare executables
            if old_compiler_exe == new_compiler_exe:
                # Executables match, do nothing
                self.slog("compiler executables match, nothing to do")
            else:
                # Executables do not match, call {make_command} distclean
                cmds = [BUILD_SYSTEM_COMMANDS[self.build_system],
                        DISTCLEAN_TARGET]
                self.abort("compiler executable changed since last "
                           "configuration step - please run '{c}' first"
                           .format(c=' '.join(cmds)))

    def run_cmake(self):
        # Build command list
        cmds = []
        envvars = []
        # ... with host-specific pre-configuration commands
        cmds.extend(self.conf_configure_host)
        # ... with compiler-specific pre-configuration commands
        if self.compiler in self.conf_configure:
            cmds.extend(self.conf_configure[self.compiler])
        # ... with the host-specific preserved environment variables
        envvars.extend(self.conf_preserve_env_host)
        # ... with the compiler-specific preserved environment variables
        if self.compiler in self.conf_preserve_env:
            envvars.extend(self.conf_preserve_env[self.compiler])
        # ... with the actual configuration command
        if len(envvars) > 0 and self.build_system == 'make':
            pe = (' && ' + os.path.join(self.maia_root, ADDENV_SCRIPT)
                  + ' ' + os.path.join(self.cwd, self.build_dir, 'Makefile') + ' '
                  + ' '.join(envvars) +
                  ' && ' + os.path.join(self.maia_root, ADDENV_SCRIPT)
                  + ' ' + os.path.join(self.cwd, self.build_dir,  SRC_DIR, 'Makefile') + ' '
                  + ' '.join(envvars))
        else:
            pe = ''

        # Generate executable
        if self.enable_csa and self.conf_csa_dir:
            exe = os.path.join(self.conf_csa_dir, 'libexec', 'c++-analyzer')
        else:
            exe = self.conf_compiler_exe[self.compiler]

        # Generate zlib argument
        if self.with_zlib:
            with_zlib = 'ON'
        else:
            with_zlib = 'OFF'

         # Generate zlib argument
        if self.with_hdf5:
            with_hdf5 = 'ON'
        else:
            with_hdf5 = 'OFF'
            # deactive structured fv solver
            if self.disable_components:
               if not "structured" in self.disable_components:
                   self.log("adding 'structured' to disable_components since WITH-HDF5=OFF")
                   self.disable_components.append("structured")
            else:
                self.log("adding 'structured' to disable_components since WITH-HDF5=OFF")
                self.disable_components = ["structured"]

        # Generate CANTERA argument
        if self.with_cantera:
            with_cantera = 'ON'
        else:
            with_cantera = 'OFF'

        # Generate likwid argument
        if self.with_likwid:
            with_likwid = 'ON'
        else:
            with_likwid = 'OFF'

        # Generate argument to disable components
        if self.disable_components:
            disable_components = []
            for c in self.disable_components:
                disable_components.append(
                    '-DMAIA_DISABLE_' + c.upper() + '=ON')
            disable_components = ' '.join(disable_components)
        else:
            disable_components = ''

        # Generate backtrace argument
        if self.enable_backtrace:
            enable_backtrace = 'ON'
        else:
            enable_backtrace = 'OFF'

        # Generate compiler commands argument
        if self.output_compile_commands:
            output_compile_commands = 'ON'
        else:
            output_compile_commands = 'OFF'

        # Generate instrumentation argument
        if self.enable_instrumentation:
            if self.enable_instrumentation == 'scorep':
                flags = {'mpi': '--mpp=mpi', 'user': '--user',
                         'compiler': '--compiler'}
                instrumentation = 'scorep '
                if 'compiler' not in self.instrument:
                    instrumentation += "--nocompiler "
                instrumentation += "--nocuda " #Otherwise doesnt work on nodes without GPUs like claix cpu
                instrumentation += "--thread=none "
                for f in self.instrument:
                    instrumentation += "{flag} ".format(flag=flags[f])
                    instrumentation_define = ''
            elif self.enable_instrumentation == 'nvtx':
                instrumentation = ''
                instrumentation_define = '-DWITH_NVTX=ON '
        else:
            instrumentation = ''
            instrumentation_define = ''
        
        # Generate parallel STL argument
        if self.disable_pstl:
            pstl = 'OFF'
        else:
            pstl = 'ON'

        pstl_options = ''
        if len(self.pstl_options) > 0:
            pstl_options = '-DPSTL_FLAGS="{}"'.format(self.pstl_options)

        # Generate OpenMP argument
        if self.disable_openmp:
            openmp = 'OFF'
        else:
            openmp = 'ON'

         # Generate KNL_BOOSTER argument
        if self.with_knlBooster:
            with_knlBooster = 'ON'
        else:
            with_knlBooster = 'OFF'

       # Generate clang static analyzer argument
        if self.enable_csa and self.conf_csa_dir:
            csa = 'ON'
        else:
            csa = 'OFF'

        # Generate debug function argument
        if self.enable_debug:
            enable_debug = 'ON'
        else:
            enable_debug = 'OFF'

        # Generate allow unused argument
        if self.allow_unused:
            allow_unused = 'ON'
        else:
            allow_unused = 'OFF'

        # Generate I/O backend argument
        if self.io == DEFAULT_IO_BACKEND:
            use_io_backend = DEFAULT_IO_BACKEND
        else:
            use_io_backend = IO_BACKEND_CLASSES[self.io]

        # Generate disable output argument
        if self.disable_output:
            disable_output = 'ON'
        else:
            disable_output = 'OFF'

        # Build cmake command
        cmake_cmd = ["CXX='{i}{exe}'", "cmake {path}", "-B{bdir}", "-G '{bs}'",
                     "-DMAIA_COMPILER={c}", "-DCMAKE_BUILD_TYPE={bt}",
                     "-DENABLE_PSTL={pstl} ", "{pstl_options}",
                     "-DENABLE_OPENMP={o}", "-DENABLE_DEBUG_FUNCTION={ed}",
                     "-DCLANG_STATIC_ANALYZER={csa}",
                     "-DALLOW_UNUSED={au}", "-DENABLE_BACKTRACE={back}",
                     "-DCMAKE_EXPORT_COMPILE_COMMANDS={cc}",
                     "-DWITH_ZLIB={z}", "-DUSE_IO_BACKEND={io}",
                     "-DWITH_HDF5={h5}", "-DWITH_KNLBOOSTER={knl}",
                     "-DDISABLE_OUTPUT={dOut}",
                     "-DWITH_LIKWID={l}", "-DWITH_CANTERA={can}",
                     disable_components,
                     "{pe}",
                     "-DWITH_INSTRUMENTATION={ibool}"
                     ]
        # Build cmake documentation command
        doc_cmd = ["CXX='{i}{exe}'", "cmake {path}", "-B{ddir}"]
        cmds.append(
            ' '.join(cmake_cmd).format(
                exe=exe,
                path=os.path.relpath(self.maia_root),
                bdir=self.build_dir,
                bs=SUPPORTED_BUILD_SYSTEMS[self.build_system],
                c=self.compiler,
                bt=self.build_type,
                i=instrumentation,
                ibool=True if self.enable_instrumentation else False,
                o=openmp,
                pstl=pstl,
                pstl_options=pstl_options,
                ed=enable_debug,
                csa=csa,
                au=allow_unused,
                back=enable_backtrace,
                cc=output_compile_commands,
                z=with_zlib,
                io=use_io_backend,
                dOut=disable_output,
                h5=with_hdf5,
                can=with_cantera,
                l=with_likwid,
                pe=pe,
                knl=with_knlBooster))
        cmds.append(
            ' '.join(doc_cmd).format(
                i=instrumentation,
                exe=exe,
                path=os.path.relpath(self.doc_dir),
                ddir=os.path.relpath(self.doc_dir)))
        cmd = '; '.join(cmds)

        # Run cmake
        self.log("configuring MAIA")
        self.slog(cmd)
        _, o, e = self.configure(cmd, bash=True, critical=True)
        self.slog(o, raw=True)

    def execute(self, cmds, bash=False):
        # If bash is True, evalue `cmds` inside bash shell
        if bash:
            cmds = ['/bin/bash', '-c', cmds]

        # Run command
        p = subprocess.Popen(cmds, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = p.communicate()

        # Return exit value and output
        return p.returncode, stdout, stderr

    def configure(self, cmds, critical=False, bash=False):
        # Run commands
        returncode, stdout, stderr = self.execute(cmds, bash=bash)

        # If returncode indicates a problem and critical is True, abort script
        if (returncode != 0 or "CMake Error" in stderr) and critical:
            cmd = cmds if bash else ' '.join(cmds)
            self.slog(cmd)
            self.slog("$? = {rc}".format(rc=returncode))
            self.slog("output of failed command:", error=True)
            self.slog(stderr, raw=True)
            self.abort("critical command failed")

        # Return exit value and output of commands
        return returncode, stdout, stderr

    def log(self, msg, raw=False, error=False, silent=False, newline=True,
            highlight=None):
        # If this is an error, write to stderr instead of stdout
        if error:
            fd = sys.stderr
        else:
            fd = sys.stdout

        # If raw, do not apply any formatting
        if raw:
            formatted = str(msg)
        else:
            formatted = "{c}: {e}{m}{n}".format(e="error: " if error else '',
                                                c=self.configure_file_name,
                                                m=str(msg).strip(),
                                                n='\n' if newline else '')

        # Write message to log file
        self.logfile.write(formatted)

        # If this was not a silenced command, also write to stdout/stderr
        if not silent:
            # Only apply coloring if stdout/stderr is a terminal, not a file,
            # and only if it is not disabled by the user
            if fd.isatty() and (COLORS_ENV not in os.environ or
                                os.environ[COLORS_ENV] not in ['off']):
                if highlight == 'error':
                    formatted = '\033[31m' + formatted + '\033[0m'
                elif highlight == 'success':
                    formatted = '\033[32m' + formatted + '\033[0m'
                elif highlight == 'info':
                    formatted = '\033[34m' + formatted + '\033[0m'
            fd.write(formatted)

            # Flush to make sure that formatting codes do not get mixed up
            fd.flush()

    def slog(self, msg, raw=False, error=False, newline=True, highlight=None):
        self.log(msg, raw=raw, error=error, newline=newline,
                 highlight=highlight, silent=True)

    def abort(self, msg):
        # Write error message to log
        self.log(str(msg).strip(), error=True, highlight='error')

        # Write info message to log
        self.log("See `config.log' for more details.\n", raw=True, error=True,
                 highlight='error')

        # Abort with bad exit code
        sys.exit(1)


def main():
    # Create class instance and call run method
    cmc = CMakeConfigure()
    cmc.run()


# Run this script only if it is called directly
if __name__ == '__main__':
    main()
