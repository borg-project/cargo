#
# source
#

install(
    FILES
        __init__.py
        errors.py
        flags.py
        io.py
        iterators.py
        json.py
        log.py
        plot.py
        random.py
        sugar.py
        temporal.py
        testing.py
		profile.py
    DESTINATION lib/cargo
    )

#
# descend
#

add_subdirectory(_files)
add_subdirectory(labor)
add_subdirectory(sql)
add_subdirectory(statistics)
add_subdirectory(test)
add_subdirectory(unix)

