project(net_test)
add_subdirectory(lib/googletest-master)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}/)

add_executable(net_test
        PerceptronTest.cxx)

target_link_libraries(net_test gtest gtest_main)