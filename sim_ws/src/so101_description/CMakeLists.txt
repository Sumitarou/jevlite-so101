cmake_minimum_required(VERSION 3.8)
project(so101_description)
find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf DESTINATION share/${PROJECT_NAME})
install(FILES assets.lock LICENSE NOTICE DESTINATION share/${PROJECT_NAME})
# The meshes are not in git: run scripts/sim/fetch_so101_assets.py before building.
# With --symlink-install the installed directory links back here, so fetching later also works.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/meshes)
  install(DIRECTORY meshes DESTINATION share/${PROJECT_NAME})
else()
  message(WARNING "so101_description: meshes/ missing - run scripts/sim/fetch_so101_assets.py")
endif()

ament_package()
