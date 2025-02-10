cmake_minimum_required(VERSION 3.10)
project(BCI_Simulation)

set(CMAKE_CXX_STANDARD 17)

add_executable(BCI_Simulation src/cpp/main.cpp src/cpp/eeg_filter.cpp src/cpp/ffi_bridge.cpp)

add_custom_command(TARGET BCI_Simulation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/src/rust/eeg_ml/target/release/eeg_ml.dll"
        $<TARGET_FILE_DIR:BCI_Simulation>)
