cmake_minimum_required(VERSION 3.8)
project(ros2_gsam2)

find_package(ament_cmake REQUIRED)
find_package(ament_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Install directories (CMake handles directories correctly)
install(DIRECTORY launch resources
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
