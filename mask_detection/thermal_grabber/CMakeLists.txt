cmake_minimum_required (VERSION 3.14)

project (lepton3_jetson)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
set(CMAKE_DEBUG_POSTFIX "_d")

#########################################################
# FLAGS
set (CMAKE_CXX_STANDARD 17)
#########################################################

# Letpon3 Library source code
add_subdirectory (grabber_lib)

# Thermal grabber
add_subdirectory (thermal_grabber)
