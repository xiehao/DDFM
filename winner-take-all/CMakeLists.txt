project(winner-take-all)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

include_directories(src)
aux_source_directory(src SRC_LIST)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb ")
#set(CMAKE_CXX_FLAGS_RELEASE "${ENV{CXXFLAGS} -O3 -Wall")
