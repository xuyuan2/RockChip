cmake_minimum_required (VERSION 2.8)
# Flags overwriting the default nes
#if (NOT CMAKE_TOOLCHAIN_FILE)
#set(CMAKE_USER_MAKE_RULES_OVERRIDE
#       ${CMAKE_CURRENT_SOURCE_DIR}/c_flags_overrides.cmake)
#set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX
#       ${CMAKE_CURRENT_SOURCE_DIR}/cxx_flags_overrides.cmake)
#endif()

project(tablet)

FIND_PACKAGE(Threads REQUIRED)
###########################################
#       Environment definitions
###########################################

# Init env variables
set(WIN_ENV false)
set(LX_ENV false)

# Update environment
if(WIN32 OR WIN64)
    set(WIN_ENV true)
    message("Windows Environment Detected")
    add_definitions(-DWIN_ENV)

elseif(UNIX)
    set(LX_ENV true)
    message("Linux Environment Detected")
    add_definitions(-DLX_ENV)

endif()

###########################################
#       Features per OS
###########################################


# Update features per OS
if(WIN_ENV)
    set(OS_NAME "win32")

elseif(LX_ENV)
    set(OS_NAME "lx")

endif()


###########################################
#       External dependencies
###########################################

    include_directories(dep/opencv/include/calib3d)
    include_directories(dep/opencv/include/ccalib)
    include_directories(dep/opencv/include/core)
    include_directories(dep/opencv/include/face)
    include_directories(dep/opencv/include/features2d)
    include_directories(dep/opencv/include/flann)
    include_directories(dep/opencv/include/fuzzy)
    include_directories(dep/opencv/include/hal)
    include_directories(dep/opencv/include/highgui)
    include_directories(dep/opencv/include/imgcodecs)
    include_directories(dep/opencv/include/imgproc)
    include_directories(dep/opencv/include/line_descriptor)
    include_directories(dep/opencv/include/ml)
    include_directories(dep/opencv/include/objdetect)
    include_directories(dep/opencv/include/optflow)
    include_directories(dep/opencv/include/photo)
    include_directories(dep/opencv/include/shape)
    include_directories(dep/opencv/include/stereo)
    include_directories(dep/opencv/include/surface_matching)
    include_directories(dep/opencv/include/text)
    include_directories(dep/opencv/include/video)
    include_directories(dep/opencv/include/videoio)
    include_directories(dep/opencv/include/videostab)
    include_directories(dep/opencv/include/ximgproc)

    # List OpencvLibs
if(WIN_ENV)

    set (OPENCV_LIBS
        optimized
        opencv_bgsegm300
        optimized
        opencv_calib3d300
        optimized
        opencv_ccalib300
        optimized
        opencv_core300
        optimized
        opencv_face300
        optimized
        opencv_features2d300
        optimized
        opencv_flann300
        optimized
        opencv_fuzzy300
        optimized
        opencv_highgui300
        optimized
        opencv_imgcodecs300
        optimized
        opencv_imgproc300
        optimized
        opencv_line_descriptor300
        optimized
        opencv_ml300
        optimized
        opencv_objdetect300
        optimized
        opencv_optflow300
        optimized
        opencv_photo300
        optimized
        opencv_plot300
        optimized
        opencv_shape300
        optimized
        opencv_stereo300
        optimized
        opencv_surface_matching300
        optimized
        opencv_text300
        optimized
        opencv_video300
        optimized
        opencv_videoio300
        optimized
        opencv_videostab300
        optimized
        opencv_ximgproc300
        optimized
        opencv_xobjdetect300

        debug
        opencv_bgsegm300d
        debug
        opencv_calib3d300d
        debug
        opencv_ccalib300d
        debug
        opencv_core300d
        debug
        opencv_face300d
        debug
        opencv_features2d300d
        debug
        opencv_flann300d
        debug
        opencv_fuzzy300d
        debug
        opencv_highgui300d
        debug
        opencv_imgcodecs300d
        debug
        opencv_imgproc300d
        debug
        opencv_line_descriptor300d
        debug
        opencv_ml300d
        debug
        opencv_objdetect300d
        debug
        opencv_optflow300d
        debug
        opencv_photo300d
        debug
        opencv_plot300d
        debug
        opencv_shape300d
        debug
        opencv_stereo300d
        debug
        opencv_surface_matching300d
        debug
        opencv_text300d
        debug
        opencv_video300d
        debug
        opencv_videoio300d
        debug
        opencv_videostab300d
        debug
        opencv_ximgproc300d
        debug
        opencv_xobjdetect300d
    )
elseif(LX_ENV)
    set( OPENCV_LIBS	
        #opencv_bgsegm
        opencv_core
        #opencv_calib3d
        #opencv_ccalib	
        #opencv_face
        opencv_features2d
        opencv_flann
        #opencv_fuzzy
        opencv_highgui
        opencv_imgcodecs
        opencv_imgproc
        #opencv_line_descriptor
        opencv_ml
        opencv_objdetect
        #opencv_optflow
        opencv_photo
        #opencv_plot
        opencv_shape
        #opencv_stereo
        #opencv_surface_matching
        #opencv_text
        opencv_video
        opencv_videoio
        opencv_videostab
        #opencv_ximgproc
        #opencv_xobjdetect
       )


endif()


###########################################
#       Compilation flags
###########################################
if(NOT WIN_ENV)
    add_definitions(-std=c++11)
    add_definitions(-zlib)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-O2)
endif()


###########################################
#       Link to external libs
###########################################

set(OPENCV_PATH ${CMAKE_SOURCE_DIR}/dep/opencv/lib/${OS_NAME})
link_directories(${OPENCV_PATH})

###########################################
#       Configure destination folders
###########################################

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_INSTALL_PREFIX "bin")



add_subdirectory(main)
