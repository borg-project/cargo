#
# source
#

install(
    FILES
        __init__.py
		copy.py
		describe.py
		prune.py
    DESTINATION lib/cargo/tools/sql
    )

#
# descend
#

add_subdirectory(test)

