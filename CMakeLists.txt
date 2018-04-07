cmake_minimum_required(VERSION 2.8)

project(faceali)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV 3.0 QUIET)

include_directories(
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/src
)

set(SDM_SRC
src/sdm_fit.cpp
)

set(SDM_LIB "sdmface")
add_library(${SDM_LIB} STATIC ${SDM_SRC})
target_link_libraries(${SDM_LIB} 
${OpenCV_LIBS}
)

set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# training sdm face model
add_executable(train_sdm_facealign examples/train_sdm_facealign.cpp)
target_link_libraries(train_sdm_facealign ${OpenCV_LIBS})

# fitting on images 
add_executable(test_image examples/test_image.cpp)
target_link_libraries(test_image ${SDM_LIB} ${OpenCV_LIBS})

add_executable(test_testset examples/test_testset.cpp)
target_link_libraries(test_testset ${SDM_LIB} ${OpenCV_LIBS})

add_executable(test_webcam examples/test_webcam.cpp)
target_link_libraries(test_webcam ${SDM_LIB} ${OpenCV_LIBS})


