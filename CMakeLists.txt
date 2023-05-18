set(CMAKE_VERBOSE_MAKEFILE ON)

#======================#
# Name of this project #
#======================#
project ("SketchToPlantUML")

#======================#
# CMake min version    #
#======================#
cmake_minimum_required (VERSION 3.8)

#========================#
# Set C++ standard 17    #
#========================#
set(CMAKE_CXX_STANDARD 17)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

#==========================#
# Shortcuts for exec & lib #
#==========================#
# Name of executable target
set(exec ${PROJECT_NAME})
# Name of library
set(lib ${PROJECT_NAME}_lib)

#=========================#
# OpenCV4 library         #
#=========================#
find_package( OpenCV 4.0.0 REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
MESSAGE("OpenCV version : ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

#========================#
# Create project library #
#========================#
# Add the source files needed to compile the program here.
# (Except for the .cpp file that contains the main() function)
# The library will be linked to the executable later.

add_library(${lib}
    # Add any new source files here...
    src/segment/Segment.cpp
    src/text_detection/TextDetection.cpp
    src/classify/Classify.cpp
    src/transform/Transform.cpp
    src/util/Util.cpp
    src/util/Color.cpp
	
)

# Link each executable with the OpenCV lib
#if (CMAKE_VERSION VERSION_GREATER 3.12)
#  foreach(target ${lib})
#    TARGET_LINK_LIBRARIES( ${target} ${OpenCV_LIBS} )
#  endforeach()
#endif()

#========================#
# Build flags    #
#========================#
#add_compile_options(-v)

#if(MSVC)
#    add_compile_options(/W4 /Wall)
#else()
#    add_compile_options(-W4 -Wextra -Wpedantic -Werror)
#endif()

## Sanitizer libraries
# Can be linked to targets.
#set(asan
    #[=[Address sanitizer
        Detects out of bounds access, use after free (delete)
        double free (2x delete).
    #]=]
#    -fsanitize=address
#)
#set(leak
    # Memory leak sanitizer, detects dangling pointers i.e. "forgot to delete".
#    -fsanitize=leak
#)
#set(undef
    #[=[Undefined behavior sanitizer
        Detects use of null pointer, signed integer overflow division by zero,
        sign changes in integer conversion, integer conversion truncation
        etc.
    #]=]
#    -fsanitize=undefined
#)
#set(cov
    # Compile and link with coverage analysis (for analyzing test coverage.
#    --coverage
#)
#set(debug_flags
    # Flags to improve address sanitizer features.
#    -O1
#    -fno-omit-frame-pointer
#)

# GCC / Clang specific
#if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|GNU")
#    target_compile_options(${lib}
#        PRIVATE
#            # Compile options specific to this library.
#           -Wshadow
#       PUBLIC
#            # Compile options for this lib and targets that links to it.
#          "$<$<CONFIG:DEBUG>:${debug_flags}>"  # For debug only
#    )
#    target_link_libraries(${lib}
#        PUBLIC
#            # Add sanitizers for dynamic (runtime) analysis in debug config.
#            "$<$<CONFIG:DEBUG>:${asan} ${undef} ${leak} ${cov}>"  # For debug
#    )
#endif()

#==========================#
# Location of header files #
#==========================#
target_include_directories(${lib}
    # Add any new header folders here...
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/segment"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/utils"
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/text_detection"
    #PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/transform"
)

#=======================#
# Create the executable #
#=======================#
# file that contains the main() function here.
add_executable(${exec}
    "src/SketchToPlantUML.cpp"
)

# Link each executable with the OpenCV lib
if (CMAKE_VERSION VERSION_GREATER 3.12)
  foreach(target ${exec})
    TARGET_LINK_LIBRARIES( ${target} ${OpenCV_LIBS} )
    TARGET_LINK_LIBRARIES( ${target} ${lib} )    
  endforeach()
endif()


#==========================#
# Copy resources directory #
# to build directory       #
#==========================#  

# Set the path to the Resources directory
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# Get a list of all image files in the Resources directory
file(GLOB IMAGE_FILES "${RESOURCES_DIR}/*.jpg" "${RESOURCES_DIR}/*.png" "${RESOURCES_DIR}/*.bmp")
# Copy all image files to the build directory
foreach(IMAGE ${IMAGE_FILES})
    configure_file(${IMAGE} ${CMAKE_CURRENT_BINARY_DIR}/resources COPYONLY)
endforeach()

# Add build directory to the list of directories searched for resources
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})