cmake_minimum_required(VERSION 3.9.6)

project (perceptron)

set (CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        main.cxx
        Perceptron.cxx
        Perceptron.h
        )

add_executable (perceptron
        ${SOURCE_FILES}
        )
include_directories(
        .
        pcg/.
        )

add_subdirectory(tests)
