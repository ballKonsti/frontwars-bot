cmake_minimum_required(VERSION 3.20)
project(fw_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(json)

add_library(fw_core
  src/state.cpp
  src/config.cpp
  src/strategy/planner.cpp
  src/strategy/expand.cpp
  src/strategy/military.cpp
  src/strategy/economy.cpp
  src/strategy/naval.cpp
  src/strategy/diplomacy.cpp
  src/strategy/strategic.cpp
  src/learn/mlp.cpp
  src/learn/learner.cpp)
target_include_directories(fw_core PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(fw_core PUBLIC nlohmann_json::nlohmann_json Threads::Threads)
# The nets are plain loops; -O3 -march=native -ffast-math lets the compiler vectorise them for this machine.
target_compile_options(fw_core PRIVATE -Wall -Wextra -O3 -march=native -ffast-math)

add_executable(fw_engine src/main.cpp)
target_link_libraries(fw_engine PRIVATE fw_core)

find_package(GTest QUIET)
if(GTest_FOUND)
  enable_testing()
  add_executable(fw_tests tests/strategy_test.cpp tests/learn_test.cpp)
  target_link_libraries(fw_tests PRIVATE fw_core GTest::gtest_main)
  add_test(NAME fw_tests COMMAND fw_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
