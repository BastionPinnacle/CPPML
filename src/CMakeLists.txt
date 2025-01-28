add_library(cppml batch.cpp function.cpp loss.cpp module.cpp sequential.cpp)

target_link_libraries(cppml PRIVATE compiler_flags)

target_include_directories(cppml PUBLIC ${CMAKE_SOURCE_DIR}/include)