cmake_minimum_required(VERSION 3.21)
project(patchmatch LANGUAGES CXX)

option(PATCHMATCH_BUILD_EXAMPLES "Build the C++ example (requires OpenCV imgcodecs)" OFF)
option(PATCHMATCH_FAST_MATH "Compile with fast-math optimizations" ON)
option(PATCHMATCH_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer, without fast math (GCC, Clang)" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(PATCHMATCH_OPENCV_COMPONENTS core)
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

# CI adds CMAKE_COMPILE_WARNING_AS_ERROR; the OpenCV headers are system headers
if(MSVC)
  target_compile_options(patchmatch PRIVATE /W4)
else()
  target_compile_options(patchmatch PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
endif()

if(MSVC AND NOT OpenCV_SHARED)
  # A static OpenCV (as built by scripts/build_opencv.py) uses the static MSVC runtime.
  set_property(TARGET patchmatch PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(PATCHMATCH_FAST_MATH AND NOT PATCHMATCH_SANITIZE)
  if(MSVC)
    target_compile_options(patchmatch PRIVATE /fp:fast)
  else()
    target_compile_options(patchmatch PRIVATE -ffast-math)
  endif()
endif()

if(PATCHMATCH_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "PATCHMATCH_SANITIZE needs GCC or Clang")
  endif()
  # GCC's -fsanitize=undefined leaves out float-cast-overflow
  set(PATCHMATCH_SANITIZERS -fsanitize=address,undefined,float-cast-overflow)
  target_compile_options(patchmatch PRIVATE ${PATCHMATCH_SANITIZERS}
    -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
  target_link_options(patchmatch PRIVATE ${PATCHMATCH_SANITIZERS})
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
