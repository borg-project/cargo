add_library(
    py_coroutines SHARED
    _coroutines.cc
    Trampoline.h
    Trampoline.cc
    TrampolineThread.h
    TrampolineThread.cc
    CppCoroutine.h
    CppCoroutine.cc
    TrampolineEvent.h
    TrampolineEvent.cc
    FileCoroutines.h
    FileCoroutines.cc
    PySignalSource.h
    PySignalSource.cc
    details/TrampolineEventWaiter.h
    details/TrampolineEventWaiter.cc
    details/FileOperationCoroutine.h
    details/FileOperationCoroutine.cc
    details/FileReadCoroutine.h
    details/FileReadCoroutine.cc
    details/FileReadUntilCoroutine.h
    details/FileReadUntilCoroutine.cc
    details/FileReadAtMostCoroutine.h
    details/FileReadAtMostCoroutine.cc
    )

find_package(
    Boost REQUIRED COMPONENTS
    python
    )

include_directories(
    ${UuvUtils_INCLUDE_DIRS}
    )

link_directories(
    ${UuvUtils_LIBRARY_DIRS}
    )

target_link_libraries(
    py_coroutines
    UuvUtils
    ConversionsCommon
    ${Boost_LIBRARIES}
    )

install(
    TARGETS py_coroutines
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

