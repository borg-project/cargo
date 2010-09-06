"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

cdef extern from "gsl/gsl_errno.h":
    ctypedef struct gsl_error_handler_t

    int GSL_SUCCESS

    char*                gsl_strerror             (int gsl_errno)
    gsl_error_handler_t* gsl_set_error_handler_off()

