cmake_minimum_required(VERSION 3.10)
project(ninibot_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/GraspCup.action"
  "action/MoveCup.action"
  "srv/SolveGrippertipIk.srv"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
