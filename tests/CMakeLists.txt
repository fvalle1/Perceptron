project(perceptron_test)
add_subdirectory(lib/googletest-master)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}/)

add_executable(perceptron_test
        PerceptronTest.cxx)

target_link_libraries(perceptron_test gtest gtest_main)