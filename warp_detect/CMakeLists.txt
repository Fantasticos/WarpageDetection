cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(warp_detect)

find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(headers
    src/testicp.h
    src/lineseg2d.h
    src/visualize.h
    src/registration.hpp
    src/warpage.hpp
    src/gldetect.hpp
    )

set(sources
    src/main.cpp
    src/registration.cpp
    src/warpage.cpp
    src/gldetect.cpp
    )
list(SORT headers)
list(SORT sources)
source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})
add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME} ${PCL_LIBRARIES})
target_link_libraries( ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} )
