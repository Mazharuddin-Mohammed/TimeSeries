find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(tsx_core STATIC
    src/core/ts_arima.cpp
    src/utils/matrix_ops.cpp
)

target_include_directories(tsx_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tsx_core PUBLIC ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

# Tests
add_executable(test_arima tests/test_arima.cpp)
target_link_libraries(test_arima PRIVATE tsx_core)