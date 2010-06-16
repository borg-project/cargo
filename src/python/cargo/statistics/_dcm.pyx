"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import  numpy
cimport numpy

cdef extern from "float.h":
    double DBL_MIN

cdef extern from "math.h":
    double fabs(double)
    int isfinite(double)

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t

    void* malloc(size_t size)
    void  free  (void* ptr)

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

cdef struct NormCount:
    unsigned int first
    double       second

cdef class PreWallachRecurrence:
    """
    Precomputed values used by the Wallach recurrence estimator.
    """

    cdef NormCount*  c_dot
    cdef size_t      c_dot_size
    cdef NormCount** c_k
    cdef size_t*     c_k_sizes

    def __cinit__(self):
        """
        Construct.
        """

    def __dealloc__(self):
        """
        Destruct.
        """

        # FIXME don't leak memory

def get_first(pair):
    return pair[0]

def pre_estimate_dcm_wallach_recurrence(
    numpy.ndarray[unsigned int, ndim = 2] counts_ND,
    numpy.ndarray[double, ndim = 1]       weights_N,
    ):
    """
    Precomputation for the Wallach DCM estimator.
    """

    # mise en place
    cdef size_t N = counts_ND.shape[0]
    cdef size_t D = counts_ND.shape[1]

    # precompute the unweighted norms
    c_dot_map = {}
    c_k_maps  = [{} for _ in xrange(D)]

    cdef double        previous
    cdef unsigned int  l1_norm
    cdef unsigned long count

    for n in xrange(N):
        l1_norm = 0

        for d in xrange(D):
            count    = counts_ND[n, d]
            l1_norm += count

            if count > 0:
                previous           = c_k_maps[d].get(count, 0)
                c_k_maps[d][count] = previous + weights_N[n]

        if l1_norm > 0:
            previous           = c_dot_map.get(l1_norm, 0)
            c_dot_map[l1_norm] = previous + weights_N[n]

    # arrange them for estimation
    cdef PreWallachRecurrence pre = PreWallachRecurrence()

    pre.c_dot_size  = len(c_dot_map)
    pre.c_dot       = <NormCount*>malloc(len(c_dot_map) * sizeof(NormCount))
    c_dot_map_items = sorted(c_dot_map.iteritems(), key = get_first, reverse = True)

    for (i, (first, second)) in enumerate(c_dot_map_items):
        pre.c_dot[i].first  = first
        pre.c_dot[i].second = second

    pre.c_k       = <NormCount**>malloc(D * sizeof(NormCount*))
    pre.c_k_sizes = <size_t*>malloc(D * sizeof(size_t))

    for d in xrange(D):
        pre.c_k_sizes[d] = len(c_k_maps[d])
        pre.c_k[d]       = <NormCount*>malloc(len(c_k_maps[d]) * sizeof(NormCount))

        c_k_maps_d = sorted(c_k_maps[d].iteritems(), key = get_first, reverse = True)

        for (i, (first, second)) in enumerate(c_k_maps_d):
            pre.c_k[d][i].first  = first
            pre.c_k[d][i].second = second

    return pre

def estimate_dcm_wallach_recurrence(
    numpy.ndarray[unsigned int, ndim = 2] counts_ND,
    numpy.ndarray[double, ndim = 1]       weights_N,
    double threshold,
    unsigned int cutoff,
    ):
    """
    Estimate the maximum likelihood DCM distribution.

    Uses the fixed-point estimator of Hanna Wallach that exploits digamma recurrence.
    """

    # mise en place
    cdef size_t N = counts_ND.shape[0]
    cdef size_t D = counts_ND.shape[1]

    assert weights_N.shape[0] == N

    cdef numpy.ndarray[double, ndim = 1] alpha_D = numpy.ones(D)

    # precompute the weighted norms
    cdef PreWallachRecurrence pre = pre_estimate_dcm_wallach_recurrence(counts_ND, weights_N)

    # run the fixed-point iteration to convergence
    cdef size_t    i
    cdef size_t    k
    cdef size_t    d
    cdef size_t    d_
    cdef double    ratio
    cdef double    sum_alpha
    cdef double    wallach_s
    cdef double    wallach_s_k
    cdef double    wallach_d
    cdef double    wallach_d_k
    cdef double    difference = threshold
    cdef NormCount c_k_item

    while cutoff > 0 and difference >= threshold:
        # countdown
        cutoff -= 1

        # compute sum_alpha
        sum_alpha = 0.0

        for d in xrange(D):
            sum_alpha += alpha_D[d]

        if sum_alpha > 1e6:
            # FIXME a bit of a hack
            break

        # compute the denominator; wallach_* are named as in (Wallach 2008)
        wallach_s = 0.0
        wallach_d = 0.0
        k         = 0

        for i from pre.c_dot_size >= i > 0:
            while k < pre.c_dot[i - 1].first:
                wallach_d += 1.0 / (k + sum_alpha)
                k         += 1

            wallach_s += pre.c_dot[i - 1].second * wallach_d

        # compute the numerator and update alpha
        difference = 0.0

        for d in xrange(D):
            # compute the numerator
            wallach_s_k = 0.0
            wallach_d_k = 0.0
            k           = 0

            for i from pre.c_k_sizes[d] >= i > 0:
                c_k_item = pre.c_k[d][i - 1]

                while k < c_k_item.first:
                    wallach_d_k += 1.0 / (k + alpha_D[d])
                    k           += 1

                wallach_s_k += c_k_item.second * wallach_d_k

            # update this dimension of alpha
            ratio = wallach_s_k / wallach_s

            alpha_D[d] *= ratio
            difference += fabs(ratio - 1.0)

            if alpha_D[d] < DBL_MIN:
                alpha_D[d] = DBL_MIN

    return alpha_D

