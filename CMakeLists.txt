cmake_minimum_required(VERSION 2.8.12)
project(MLB)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS} include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB SOURCES "src/*.cpp")
 
add_executable(pitching_detect ${SOURCES})
TARGET_LINK_LIBRARIES(pitching_detect ${OpenCV_LIBS})
