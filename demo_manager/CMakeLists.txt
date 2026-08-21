enable_language(CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSON_C REQUIRED json-c)

# demo-manager: top-level daemon that manages all demo applications
add_executable(demo-manager src/demo_manager_daemon.cpp)

target_compile_options(demo-manager PRIVATE -Wall -Wextra -O2)
target_include_directories(demo-manager PRIVATE ${JSON_C_INCLUDE_DIRS})
target_link_libraries(demo-manager PRIVATE ${JSON_C_LIBRARIES})

# demo-ctl: CLI client to send commands to the daemon
add_executable(demo-ctl src/demo_ctl.cpp)

target_compile_options(demo-ctl PRIVATE -Wall -Wextra -O2)
target_include_directories(demo-ctl PRIVATE ${JSON_C_INCLUDE_DIRS})
target_link_libraries(demo-ctl PRIVATE ${JSON_C_LIBRARIES})

install(TARGETS demo-manager demo-ctl RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES demo-manager.service DESTINATION ${SYSTEMD_SYSTEM_UNITDIR})
