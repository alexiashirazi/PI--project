cmake_minimum_required(VERSION 3.14)
project(Lab8)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Lab8
        main.cpp
        src/proiect.cpp
        src/proiect.h
)

target_link_libraries(Lab8 ${OpenCV_LIBS})
