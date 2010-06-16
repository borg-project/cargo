"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import  numpy
cimport numpy

cdef extern from "gsl/gsl_errno.h":
    int GSL_SUCCESS

    char* gsl_strerror(int gsl_errno)

cdef extern from "gsl/gsl_sf_result.h":
    ctypedef struct gsl_sf_result:
        double val
        double err

cdef extern from "gsl/gsl_sf.h":
    int gsl_sf_lngamma_e(double v, gsl_sf_result* result)

cpdef double ln_gamma(double v):
    """
    Compute the natural log of the Pochhammer function.
    """

    cdef gsl_sf_result result
    cdef int           status = gsl_sf_lngamma_e(v, &result)

    if status != GSL_SUCCESS:
        raise RuntimeError("%s (v = %f)" % (gsl_strerror(status), v))

        return -1

    return result.val

def multinomial_log_probability(
    numpy.ndarray[double, ndim = 1]        log_beta_D,
    numpy.ndarray[unsigned long, ndim = 1] counts_D,
    ):
    """
    Calculate the log probability of the multinomial distribution.
    """

    # mise en place
    cdef size_t D = log_beta_D.shape[0]

    assert counts_D.shape[0] == D

    # calculate
    cdef unsigned long n = 0

    for d in xrange(D):
        n += counts_D[d]

    cdef double lp = ln_gamma(n + 1)

    for d in xrange(D):
        lp -= ln_gamma(counts_D[d] + 1)
        lp += log_beta_D[d] * counts_D[d]

    return lp

