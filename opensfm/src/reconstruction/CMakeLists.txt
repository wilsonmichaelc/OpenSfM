set(RECONSTRUCTION_FILES
  shot.h
  landmark.h
  manager.h
  defines.h
  src/landmark.cc
  src/manager.cc
  src/shot.cc
)
add_library(reconstruction ${RECONSTRUCTION_FILES})
target_link_libraries(reconstruction 
  PRIVATE
    foundation
    geometry
)
target_include_directories(reconstruction PUBLIC ${CMAKE_SOURCE_DIR})

# pybind11_add_module(pyrobust python/pybind.cc)
# target_link_libraries(pyrobust 
#   PRIVATE 
#     robust
#     foundation
#     pybind11
# )
set_target_properties(reconstruction PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${opensfm_SOURCE_DIR}/.."
)