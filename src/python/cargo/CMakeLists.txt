#
# source
#

install(
    FILES
        __init__.py
        errors.py
        io.py
        iterators.py
        json.py
        log.py
        condor.py
        plot.py
        random.py
        sugar.py
        temporal.py
        testing.py
		profile.py
		labor.py
		labor2.py
		numpy.py
        defaults.py
    DESTINATION lib/cargo
    )

#
# descend
#

add_subdirectory(_files)
add_subdirectory(sql)
add_subdirectory(statistics)
add_subdirectory(test)
add_subdirectory(unix)
add_subdirectory(tools)
add_subdirectory(gsl)

