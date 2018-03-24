cmake_minimum_required(VERSION 3.9.6)

project (net)

set (CMAKE_CXX_STANDARD 11)

add_executable (net
        main.cxx
        Perceptron.cxx
        Perceptron.h
        )

include_directories(
        .
        pcg/.
        )

add_subdirectory(tests)
