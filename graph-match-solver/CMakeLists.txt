project(graph-match-solver)
cmake_minimum_required(VERSION 2.8)

include_directories(src)
aux_source_directory(src SRC_LIST)
include_directories(src/MinCost)
aux_source_directory(src/MinCost SRC_LIST)
include_directories(src/QPBO)
aux_source_directory(src/QPBO SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgproc opencv_highgui
opencv_flann)

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb ")
#set(CMAKE_CXX_FLAGS_RELEASE "${ENV{CXXFLAGS} -O3 -Wall")
