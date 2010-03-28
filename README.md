CARGO.KIT
=========

COMPILATION
-----------

To provide the location of libraries in nonstandard installation prefixes to
cmake, use the CMAKE_PREFIX_PATH environment variable; eg,

CMAKE_PREFIX_PATH=/opt/foo ./run_cmake

, where required libraries are installed under /opt/foo/include and
/opt/foo/lib. Note that cmake has known issues with paths that include colons.

LICENSE
-------

This software package is provided under the non-copyleft free "MIT" license.
The complete legal notice can be found in the included LICENSE file.

