#
# source
#

install(
    FILES
        __init__.py
    DESTINATION lib/cargo/tools
    )

#
# descend
#

add_subdirectory(labor)
add_subdirectory(sql)

