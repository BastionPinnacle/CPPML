add_library(cppml Batch.cpp Function.cpp Loss.cpp Module.cpp Sequential.cpp)

target_link_libraries(cppml PRIVATE compiler_flags)

target_include_directories(cppml PRIVATE "${CMAKE_SOURCE_DIR}/include")
