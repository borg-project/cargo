install(
	FILES
    array_packing.h
	assert.h
	DESTINATION include/utexas
	)

# descend into subdirectories; order matters! they should be sorted
# topologically (by their dependency relations).
add_subdirectory(python)
add_subdirectory(statistics)
add_subdirectory(test)

