cmake_minimum_required(VERSION 2.8)

project(faceali)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV 3.0 QUIET)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/src
)

#set(LIB_SRC
#src/test_sdm_facealign.cpp
#src/train_sdm_facealign.cpp
#)

#set(LIB_NAME "sdm")
#add_library(${LIB_NAME} STATIC ${LIB_SRC})

set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#target_link_libraries(${LIB_NAME} 
#${OpenCV_LIBS}
#)

add_executable(train_sdm_facealign src/train_sdm_facealign.cpp)
target_link_libraries(train_sdm_facealign ${OpenCV_LIBS})

add_executable(test_sdm_facealign src/test_sdm_facealign.cpp)
target_link_libraries(test_sdm_facealign  ${OpenCV_LIBS})


