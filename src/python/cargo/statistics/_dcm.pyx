"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import  numpy
cimport numpy

cdef extern from "math.h":
    double fabs(double)

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t

cdef extern from "gsl/gsl_errno.h":
    int GSL_SUCCESS

    char* gsl_strerror(int gsl_errno)

cdef extern from "gsl/gsl_sf_result.h":
    ctypedef struct gsl_sf_result:
        double val
        double err

cdef extern from "gsl/gsl_sf.h":
    int gsl_sf_lnpoch_e(double a, double x, gsl_sf_result* result)
    double gsl_sf_lnpoch(double a, double x)
    double gsl_sf_psi(double x)

cpdef double ln_poch(double a, double x):
    """
    Compute the natural log of the Pochhammer function.
    """

    cdef gsl_sf_result result
    cdef int           status = gsl_sf_lnpoch_e(a, x, &result)

    if status != GSL_SUCCESS:
        raise RuntimeError("%s (a = %f; x = %f)" % (gsl_strerror(status), a, x))

        return -1

    return result.val

def dcm_log_probability(
    double                                sum_alpha,
    numpy.ndarray[double, ndim = 1]       alpha_D,
    numpy.ndarray[unsigned int, ndim = 1] counts_D,
    ):
    """
    Calculate the log probability of the DCM distribution.
    """

    # mise en place
    cdef size_t D = alpha_D.shape[0]

    if counts_D.shape[0] != D:
        raise ValueError("shapes of alpha and counts arrays do not match")

    # calculate
    cdef size_t        d
    cdef unsigned long n = 0

    for d in xrange(D):
        n += counts_D[d]

    cdef double psigm = 0.0

    for d in xrange(D):
        psigm += ln_poch(alpha_D[d], counts_D[d])

    return psigm - ln_poch(sum_alpha, n)

def minka_fixed_update(
    numpy.ndarray[double, ndim = 1]       alpha_D,
    numpy.ndarray[unsigned int, ndim = 2] counts_ND,
    numpy.ndarray[unsigned int, ndim = 1] counts_sum1_N,
    numpy.ndarray[double, ndim = 1]       weights_N,
    double                                weights_sum,
    ):
    """
    Compute the next value in the fixed-point iteration.
    """

    # parameter sanity should be ensured by the caller
    cdef size_t N = counts_ND.shape[0]
    cdef size_t D = counts_ND.shape[1]

    # calculate the denominator
    cdef size_t d
    cdef double alpha_sum = 0.0

    for d in xrange(D):
        alpha_sum += alpha_D[d]

    cdef size_t n
    cdef double denominator = 0.0

    for n in xrange(N):
        denominator += gsl_sf_psi(counts_sum1_N[n] + alpha_sum) * weights_N[n]

    denominator -= weights_sum * gsl_sf_psi(alpha_sum)

    # calculate the numerator and update alpha
    cdef double difference = 0.0
    cdef double numerator
    cdef double ratio

    for d in xrange(D):
        numerator = 0.0

        for n in xrange(N):
            numerator += gsl_sf_psi(alpha_D[d] + counts_ND[n, d]) * weights_N[n]

        numerator -= weights_sum * gsl_sf_psi(alpha_D[d])

        # update alpha
        ratio = numerator / denominator

        alpha_D[d] *= ratio
        difference += fabs(ratio - 1.0)

    # done
    return difference

def estimate_dcm_minka_fixed(
    numpy.ndarray[unsigned int, ndim = 2] counts_ND,
    numpy.ndarray[double, ndim = 1]       weights_N,
    double                                threshold,
    unsigned int                          cutoff,
    ):
    """
    Estimate the maximum likelihood DCM distribution.

    Uses the Minka fixed-point estimation method.
    """

    # mise en place
    cdef size_t N = counts_ND.shape[0]
    cdef size_t D = counts_ND.shape[1]

    assert weights_N.shape[0] == N

    # precompute count norms
    cdef size_t                                n
    cdef size_t                                d
    cdef numpy.ndarray[unsigned int, ndim = 1] counts_sum1_N = numpy.empty(N, dtype = numpy.uint)

    for n in xrange(N):
        counts_sum1_N[n] = 0

        for d in xrange(D):
            counts_sum1_N[n] += counts_ND[n, d]

    # the fixed-point loop
    cdef numpy.ndarray[double, ndim = 1] alpha_D = numpy.empty(D)

    for d in xrange(D):
        alpha_D[d] = 1.0

    cdef double weights_sum = 0.0

    for n in xrange(N):
        weights_sum += weights_N[n]

    cdef double difference = threshold
    cdef size_t i          = 0

    while i < cutoff and difference >= threshold:
        i          += 1
        difference  = minka_fixed_update(alpha_D, counts_ND, counts_sum1_N, weights_N, weights_sum)

        # the digamma function is undefined at zero, so we take the approach
        # of flooring alpha at a small non-positive value; perhaps there is
        # a more principled approach to this issue
        for d in xrange(D):
            if alpha_D[d] < 1e-16:
                alpha_D[d] = 1e-16

    return alpha_D

