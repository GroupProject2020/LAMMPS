# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/wall-e/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/193.6015.37/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/wall-e/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/193.6015.37/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake/cmake-build-debug

# Utility rule file for gitversion.

# Include the progress variables for this target.
include CMakeFiles/gitversion.dir/progress.make

CMakeFiles/gitversion:
	/home/wall-e/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/193.6015.37/bin/cmake/linux/bin/cmake -DCMAKE_CURRENT_SOURCE_DIR="/home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake" -DGIT_EXECUTABLE="/usr/bin/git" -DGIT_FOUND="TRUE" -DLAMMPS_STYLE_HEADERS_DIR="/home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake/cmake-build-debug/styles" -P /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake/Modules/generate_lmpgitversion.cmake

gitversion: CMakeFiles/gitversion
gitversion: CMakeFiles/gitversion.dir/build.make

.PHONY : gitversion

# Rule to build all files generated by this target.
CMakeFiles/gitversion.dir/build: gitversion

.PHONY : CMakeFiles/gitversion.dir/build

CMakeFiles/gitversion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gitversion.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gitversion.dir/clean

CMakeFiles/gitversion.dir/depend:
	cd /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake/cmake-build-debug /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake/cmake-build-debug /home/wall-e/Cranfield/GroupProject/mylammps/lammps/cmake/cmake-build-debug/CMakeFiles/gitversion.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gitversion.dir/depend

