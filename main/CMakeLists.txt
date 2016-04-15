cmake_minimum_required (VERSION 2.8)

project(main)

set (SRC_FILES main.cpp)

add_executable(main ${SRC_FILES})

target_link_libraries(main  ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(main  ${OPENCV_LIBS})
