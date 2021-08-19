SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -pthread")

find_package(OpenCV REQUIRED)

add_executable (thermal_grabber thermal_grabber.cpp)
target_link_libraries (thermal_grabber LINK_PUBLIC 
                       lepton3_grabber 
                       ${OpenCV_LIBS})
