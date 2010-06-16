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
    DESTINATION cargo
    )

#
# descend
#

add_subdirectory(labor)
add_subdirectory(sql)
add_subdirectory(statistics)
add_subdirectory(test)
add_subdirectory(unix)

