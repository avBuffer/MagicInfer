# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jalymo/work/ai/MagicInfer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jalymo/work/ai/MagicInfer/build

# Include any dependencies generated for this target.
include demo/CMakeFiles/yolo_infer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include demo/CMakeFiles/yolo_infer.dir/compiler_depend.make

# Include the progress variables for this target.
include demo/CMakeFiles/yolo_infer.dir/progress.make

# Include the compile flags for this target's objects.
include demo/CMakeFiles/yolo_infer.dir/flags.make

demo/CMakeFiles/yolo_infer.dir/image_util.cpp.o: demo/CMakeFiles/yolo_infer.dir/flags.make
demo/CMakeFiles/yolo_infer.dir/image_util.cpp.o: ../demo/image_util.cpp
demo/CMakeFiles/yolo_infer.dir/image_util.cpp.o: demo/CMakeFiles/yolo_infer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jalymo/work/ai/MagicInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demo/CMakeFiles/yolo_infer.dir/image_util.cpp.o"
	cd /home/jalymo/work/ai/MagicInfer/build/demo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT demo/CMakeFiles/yolo_infer.dir/image_util.cpp.o -MF CMakeFiles/yolo_infer.dir/image_util.cpp.o.d -o CMakeFiles/yolo_infer.dir/image_util.cpp.o -c /home/jalymo/work/ai/MagicInfer/demo/image_util.cpp

demo/CMakeFiles/yolo_infer.dir/image_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo_infer.dir/image_util.cpp.i"
	cd /home/jalymo/work/ai/MagicInfer/build/demo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jalymo/work/ai/MagicInfer/demo/image_util.cpp > CMakeFiles/yolo_infer.dir/image_util.cpp.i

demo/CMakeFiles/yolo_infer.dir/image_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo_infer.dir/image_util.cpp.s"
	cd /home/jalymo/work/ai/MagicInfer/build/demo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jalymo/work/ai/MagicInfer/demo/image_util.cpp -o CMakeFiles/yolo_infer.dir/image_util.cpp.s

demo/CMakeFiles/yolo_infer.dir/main.cpp.o: demo/CMakeFiles/yolo_infer.dir/flags.make
demo/CMakeFiles/yolo_infer.dir/main.cpp.o: ../demo/main.cpp
demo/CMakeFiles/yolo_infer.dir/main.cpp.o: demo/CMakeFiles/yolo_infer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jalymo/work/ai/MagicInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object demo/CMakeFiles/yolo_infer.dir/main.cpp.o"
	cd /home/jalymo/work/ai/MagicInfer/build/demo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT demo/CMakeFiles/yolo_infer.dir/main.cpp.o -MF CMakeFiles/yolo_infer.dir/main.cpp.o.d -o CMakeFiles/yolo_infer.dir/main.cpp.o -c /home/jalymo/work/ai/MagicInfer/demo/main.cpp

demo/CMakeFiles/yolo_infer.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo_infer.dir/main.cpp.i"
	cd /home/jalymo/work/ai/MagicInfer/build/demo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jalymo/work/ai/MagicInfer/demo/main.cpp > CMakeFiles/yolo_infer.dir/main.cpp.i

demo/CMakeFiles/yolo_infer.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo_infer.dir/main.cpp.s"
	cd /home/jalymo/work/ai/MagicInfer/build/demo && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jalymo/work/ai/MagicInfer/demo/main.cpp -o CMakeFiles/yolo_infer.dir/main.cpp.s

# Object files for target yolo_infer
yolo_infer_OBJECTS = \
"CMakeFiles/yolo_infer.dir/image_util.cpp.o" \
"CMakeFiles/yolo_infer.dir/main.cpp.o"

# External object files for target yolo_infer
yolo_infer_EXTERNAL_OBJECTS =

demo/yolo_infer: demo/CMakeFiles/yolo_infer.dir/image_util.cpp.o
demo/yolo_infer: demo/CMakeFiles/yolo_infer.dir/main.cpp.o
demo/yolo_infer: demo/CMakeFiles/yolo_infer.dir/build.make
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
demo/yolo_infer: ../lib/libmagic.so
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
demo/yolo_infer: /usr/local/lib/libglog.so.0.7.0
demo/yolo_infer: /usr/lib/libarmadillo.so
demo/yolo_infer: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
demo/yolo_infer: /usr/lib/x86_64-linux-gnu/libpthread.a
demo/yolo_infer: demo/CMakeFiles/yolo_infer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jalymo/work/ai/MagicInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable yolo_infer"
	cd /home/jalymo/work/ai/MagicInfer/build/demo && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo_infer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demo/CMakeFiles/yolo_infer.dir/build: demo/yolo_infer
.PHONY : demo/CMakeFiles/yolo_infer.dir/build

demo/CMakeFiles/yolo_infer.dir/clean:
	cd /home/jalymo/work/ai/MagicInfer/build/demo && $(CMAKE_COMMAND) -P CMakeFiles/yolo_infer.dir/cmake_clean.cmake
.PHONY : demo/CMakeFiles/yolo_infer.dir/clean

demo/CMakeFiles/yolo_infer.dir/depend:
	cd /home/jalymo/work/ai/MagicInfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jalymo/work/ai/MagicInfer /home/jalymo/work/ai/MagicInfer/demo /home/jalymo/work/ai/MagicInfer/build /home/jalymo/work/ai/MagicInfer/build/demo /home/jalymo/work/ai/MagicInfer/build/demo/CMakeFiles/yolo_infer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demo/CMakeFiles/yolo_infer.dir/depend
