find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(tsx_backend tsx_bind.cpp)
target_link_libraries(tsx_backend PRIVATE tsx_core)
set_target_properties(tsx_backend PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)