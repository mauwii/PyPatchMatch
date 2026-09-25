cmake_minimum_required(VERSION 3.21)
project(patchmatch LANGUAGES CXX)

option(PATCHMATCH_BUILD_EXAMPLES "Build the C++ example (requires OpenCV imgcodecs)" OFF)
option(PATCHMATCH_WITH_HIGHGUI "Enable the debug visualization of Inpainting::run (requires OpenCV highgui)" OFF)
option(PATCHMATCH_FAST_MATH "Compile with fast-math optimizations" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(PATCHMATCH_OPENCV_COMPONENTS core)
if(PATCHMATCH_WITH_HIGHGUI)
  list(APPEND PATCHMATCH_OPENCV_COMPONENTS imgproc highgui)
endif()
if(PATCHMATCH_BUILD_EXAMPLES)
  list(APPEND PATCHMATCH_OPENCV_COMPONENTS imgcodecs)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${PATCHMATCH_OPENCV_COMPONENTS})

add_library(patchmatch SHARED
  patchmatch/csrc/inpaint.cpp
  patchmatch/csrc/masked_image.cpp
  patchmatch/csrc/nnf.cpp
  patchmatch/csrc/pyinterface.cpp
)
target_include_directories(patchmatch PUBLIC patchmatch/csrc)
target_link_libraries(patchmatch PUBLIC opencv_core)

if(MSVC AND NOT OpenCV_SHARED)
  # A static OpenCV (as built by scripts/build_opencv.py) uses the static MSVC runtime.
  set_property(TARGET patchmatch PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(PATCHMATCH_FAST_MATH)
  if(MSVC)
    target_compile_options(patchmatch PRIVATE /fp:fast)
  else()
    target_compile_options(patchmatch PRIVATE -ffast-math)
  endif()
endif()

if(PATCHMATCH_WITH_HIGHGUI)
  target_compile_definitions(patchmatch PRIVATE PATCHMATCH_WITH_HIGHGUI)
  target_link_libraries(patchmatch PRIVATE opencv_imgproc opencv_highgui)
endif()

if(PATCHMATCH_BUILD_EXAMPLES)
  add_executable(cpp_example examples/cpp_example.cpp)
  target_link_libraries(cpp_example PRIVATE patchmatch opencv_imgcodecs)
endif()

# The Python package loads the library via ctypes from its own directory.
install(TARGETS patchmatch
  LIBRARY DESTINATION patchmatch
  RUNTIME DESTINATION patchmatch
)

# A static OpenCV becomes part of the library, so a wheel has to ship its licenses,
# which scripts/build_opencv.py collects in <prefix>/licenses.
if(DEFINED SKBUILD_METADATA_DIR AND NOT OpenCV_SHARED)
  set(PATCHMATCH_OPENCV_LICENSES "${OpenCV_INSTALL_PATH}/licenses")
  if(EXISTS "${PATCHMATCH_OPENCV_LICENSES}/opencv-LICENSE")
    install(DIRECTORY "${PATCHMATCH_OPENCV_LICENSES}/"
      DESTINATION "${SKBUILD_METADATA_DIR}/licenses/opencv")
  else()
    message(WARNING "No OpenCV licenses in ${PATCHMATCH_OPENCV_LICENSES}, "
      "the wheel will not include them")
  endif()
endif()
