# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(unit_tests)

# Set the source files
set(SRC_FILES
    main.cpp
    # List the test headers (they are included in main.cpp)
    #test_function1.h
    #test_function2.h
    # Add more test headers here
)

# Include directories from your main codebase
include_directories(
    ${PROJECT_SOURCE_DIR}/../../src  # Adjust the path as needed
    # Include other directories if necessary
)

# Find and include any required packages
# find_package(PackageName REQUIRED)
# include_directories(${PackageName_INCLUDE_DIRS})

# Add the executable
add_executable(unit_test ${SRC_FILES})

# Link against your main codebase library (if available)
# add_subdirectory(${PROJECT_SOURCE_DIR}/../../src ${CMAKE_BINARY_DIR}/src)
# target_link_libraries(unit_test artemis_library_name)

# If your codebase is not a library, you might need to include source files directly
# target_sources(unit_test PRIVATE ${PROJECT_SOURCE_DIR}/../../src/your_source.cpp)
