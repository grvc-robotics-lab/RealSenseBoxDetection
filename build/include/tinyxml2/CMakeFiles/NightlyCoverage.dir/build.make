# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection/build

# Utility rule file for NightlyCoverage.

# Include any custom commands dependencies for this target.
include include/tinyxml2/CMakeFiles/NightlyCoverage.dir/compiler_depend.make

# Include the progress variables for this target.
include include/tinyxml2/CMakeFiles/NightlyCoverage.dir/progress.make

include/tinyxml2/CMakeFiles/NightlyCoverage:
	cd /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection/build/include/tinyxml2 && /usr/local/bin/ctest -D NightlyCoverage

NightlyCoverage: include/tinyxml2/CMakeFiles/NightlyCoverage
NightlyCoverage: include/tinyxml2/CMakeFiles/NightlyCoverage.dir/build.make
.PHONY : NightlyCoverage

# Rule to build all files generated by this target.
include/tinyxml2/CMakeFiles/NightlyCoverage.dir/build: NightlyCoverage
.PHONY : include/tinyxml2/CMakeFiles/NightlyCoverage.dir/build

include/tinyxml2/CMakeFiles/NightlyCoverage.dir/clean:
	cd /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection/build/include/tinyxml2 && $(CMAKE_COMMAND) -P CMakeFiles/NightlyCoverage.dir/cmake_clean.cmake
.PHONY : include/tinyxml2/CMakeFiles/NightlyCoverage.dir/clean

include/tinyxml2/CMakeFiles/NightlyCoverage.dir/depend:
	cd /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection/include/tinyxml2 /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection/build /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection/build/include/tinyxml2 /home/jorge/Documents/GitHub/GRVC/RealSenseBoxDetection/build/include/tinyxml2/CMakeFiles/NightlyCoverage.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : include/tinyxml2/CMakeFiles/NightlyCoverage.dir/depend
