cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(sherpa-onnx)

set(SHERPA_ONNX_VERSION "1.9.10-mulang")

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

option(SHERPA_ONNX_ENABLE_PYTHON "Whether to build Python" ON)
option(SHERPA_ONNX_ENABLE_TESTS "Whether to build tests" OFF)
option(SHERPA_ONNX_ENABLE_CHECK "Whether to build with assert" OFF)
option(BUILD_SHARED_LIBS "Whether to build shared libraries" ON)
option(SHERPA_ONNX_ENABLE_PORTAUDIO "Whether to build with portaudio" ON)
option(SHERPA_ONNX_ENABLE_JNI "Whether to build JNI internface" OFF)
option(SHERPA_ONNX_ENABLE_C_API "Whether to build C API" ON)
option(SHERPA_ONNX_ENABLE_WEBSOCKET "Whether to build webscoket server/client" OFF)
option(SHERPA_ONNX_ENABLE_GPU "Enable ONNX Runtime GPU support" OFF)
option(SHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY "True to link libstdc++ statically. Used only when BUILD_SHARED_LIBS is OFF on Linux" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(SHERPA_ONNX_RPATH_ORIGIN "$ORIGIN")
else()
  set(SHERPA_ONNX_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${SHERPA_ONNX_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${SHERPA_ONNX_RPATH_ORIGIN})

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(DEFINED ANDROID_ABI AND NOT SHERPA_ONNX_ENABLE_JNI)
  message(STATUS "Set SHERPA_ONNX_ENABLE_JNI to ON for Android")
  set(SHERPA_ONNX_ENABLE_JNI ON CACHE BOOL "" FORCE)
endif()

if(SHERPA_ONNX_ENABLE_PYTHON AND NOT BUILD_SHARED_LIBS)
  message(STATUS "Set BUILD_SHARED_LIBS to ON since SHERPA_ONNX_ENABLE_PYTHON is ON")
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

if(SHERPA_ONNX_ENABLE_JNI AND NOT BUILD_SHARED_LIBS)
  message(STATUS "Set BUILD_SHARED_LIBS to ON since SHERPA_ONNX_ENABLE_JNI is ON")
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()

if(SHERPA_ONNX_ENABLE_GPU)
  message(WARNING "\
Compiling for NVIDIA GPU is enabled. Please make sure cudatoolkit
is installed on your system. Otherwise, you will get errors at runtime.
Hint: You don't need sudo permission to install CUDA toolkit. Please refer to
  https://k2-fsa.github.io/k2/installation/cuda-cudnn.html
to install CUDA toolkit if you have not installed it.")
  if(NOT BUILD_SHARED_LIBS)
    message(STATUS "Set BUILD_SHARED_LIBS to ON since SHERPA_ONNX_ENABLE_GPU is ON")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  endif()
endif()

if(BUILD_SHARED_LIBS AND MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(NOT BUILD_SHARED_LIBS AND MSVC)
  # see https://cmake.org/cmake/help/latest/prop_tgt/MSVC_RUNTIME_LIBRARY.html
  # https://stackoverflow.com/questions/14172856/compile-with-mt-instead-of-md-using-cmake
  if(MSVC)
      add_compile_options(
          $<$<CONFIG:>:/MT> #---------|
          $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
          $<$<CONFIG:Release>:/MT> #--|
      )
  endif()
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}")
message(STATUS "SHERPA_ONNX_ENABLE_PYTHON ${SHERPA_ONNX_ENABLE_PYTHON}")
message(STATUS "SHERPA_ONNX_ENABLE_TESTS ${SHERPA_ONNX_ENABLE_TESTS}")
message(STATUS "SHERPA_ONNX_ENABLE_CHECK ${SHERPA_ONNX_ENABLE_CHECK}")
message(STATUS "SHERPA_ONNX_ENABLE_PORTAUDIO ${SHERPA_ONNX_ENABLE_PORTAUDIO}")
message(STATUS "SHERPA_ONNX_ENABLE_JNI ${SHERPA_ONNX_ENABLE_JNI}")
message(STATUS "SHERPA_ONNX_ENABLE_C_API ${SHERPA_ONNX_ENABLE_C_API}")
message(STATUS "SHERPA_ONNX_ENABLE_WEBSOCKET ${SHERPA_ONNX_ENABLE_WEBSOCKET}")
message(STATUS "SHERPA_ONNX_ENABLE_GPU ${SHERPA_ONNX_ENABLE_GPU}")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")


include(CheckIncludeFileCXX)

if(UNIX AND NOT APPLE)
  check_include_file_cxx(alsa/asoundlib.h SHERPA_ONNX_HAS_ALSA)
  if(SHERPA_ONNX_HAS_ALSA)
    add_definitions(-DSHERPA_ONNX_ENABLE_ALSA=1)
  else()
    message(WARNING "\
Could not find alsa/asoundlib.h !
We won't build sherpa-ncnn-alsa
To fix that, please do:
  (1) sudo apt-get install alsa-utils libasound2-dev
  (2) rm -rf build
  (3) re-try
  ")
  endif()
endif()

check_include_file_cxx(cxxabi.h SHERPA_ONNX_HAVE_CXXABI_H)
check_include_file_cxx(execinfo.h SHERPA_ONNX_HAVE_EXECINFO_H)

if(WIN32)
  add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
endif()

if(WIN32 AND MSVC)
  # disable various warnings for MSVC
  # 4244: 'return': conversion from 'unsigned __int64' to 'int', possible loss of data
  # 4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
  # 4305: 'argument': truncation from 'double' to 'const float'
  # 4334: '<<': result of 32-bit shift implicitly converted to 64 bits
  # 4800: 'int': forcing value to bool 'true' or 'false'
  # 4996: 'fopen': This function or variable may be unsafe
  set(disabled_warnings
      /wd4244
      /wd4267
      /wd4305
      /wd4334
      /wd4800
      /wd4996
  )
  message(STATUS "Disabled warnings: ${disabled_warnings}")
  foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
  endforeach()

  add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(NOT BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  if(SHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY)
    message(STATUS "Link libstdc++ statically")
    set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc ")
  else()
    message(STATUS "Link libstdc++ dynamically")
  endif()
endif()

include(kaldi-native-fbank)
include(kaldi-decoder)
include(onnxruntime)
set(ONNXRUNTIME_DIR ${onnxruntime_SOURCE_DIR})
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

if(SHERPA_ONNX_ENABLE_PORTAUDIO)
  include(portaudio)
endif()

if(SHERPA_ONNX_ENABLE_PYTHON)
  include(pybind11)
endif()

if(SHERPA_ONNX_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()

if(SHERPA_ONNX_ENABLE_WEBSOCKET)
  include(websocketpp)
  include(asio)
endif()

include(espeak-ng-for-piper)
set(ESPEAK_NG_DIR ${espeak_ng_SOURCE_DIR})
message(STATUS "ESPEAK_NG_DIR: ${ESPEAK_NG_DIR}")
include(piper-phonemize)

add_subdirectory(sherpa-onnx)

if(SHERPA_ONNX_ENABLE_C_API)
  add_subdirectory(c-api-examples)
endif()
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

if(NOT BUILD_SHARED_LIBS)
  if(APPLE)
    set(SHERPA_ONNX_PKG_CONFIG_EXTRA_LIBS "-lc++ -framework Foundation")
  endif()

  if(UNIX AND NOT APPLE)
    set(SHERPA_ONNX_PKG_CONFIG_EXTRA_LIBS "-lstdc++ -lm -pthread -ldl")
  endif()
endif()

# See https://people.freedesktop.org/~dbn/pkg-config-guide.html
configure_file(cmake/sherpa-onnx.pc.in ${PROJECT_BINARY_DIR}/sherpa-onnx.pc @ONLY)
install(
  FILES
    ${PROJECT_BINARY_DIR}/sherpa-onnx.pc
  DESTINATION
    .
)
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
