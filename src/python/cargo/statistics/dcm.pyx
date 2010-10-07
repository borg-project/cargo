#"""
#@author: Bryan Silverthorn <bcs@cargo-cult.org>
#"""

#import numpy

#from cargo.statistics.base import (
    #Estimator,
    #Distribution,
    #)

#cimport numpy

#from libc.float   cimport DBL_MIN
#from libc.stdlib  cimport (
    #free,
    #malloc,
    #)
#from cargo.gsl.sf cimport (
    #psi,
    #ln_poch,
    #)

#cdef extern from "math.h":
    #double fabs    (double)
    #int    isfinite(double)

#numpy.seterr(divide = "raise", invalid = "raise", over = "warn", under = "warn") # FIXME hack

#class DirichletCompoundMultinomial(Distribution):
    #"""
    #The Dirichlet compound multinomial (DCM) distribution.

    #Relevant types:
    #- sample: D-shaped uint ndarray
    #- sequence: ND-shaped uint ndarray
    #"""

    #def __init__(self, alpha, norm = 1):
        #"""
        #Instantiate the distribution.

        #@param alpha: The distribution parameter vector.
        #@param norm:  The L1 norm of samples from this distribution.
        #"""

        ## initialization
        #self._alpha = numpy.asarray(alpha)
        #self._norm  = norm

    #def random_variate(self, random = numpy.random):
        #"""
        #Return a sample from this distribution.
        #"""

        #return random.multinomial(self._norm, random.dirichlet(self._alpha))

    #def random_variates(self, size, random = numpy.random):
        #"""
        #Return an array of samples from this distribution.
        #"""

        #variates = numpy.empty((size, self._alpha.size))

        #for i in xrange(size):
            #variates[i] = self.random_variate(random)

        #return variates

    #def log_likelihood(self, sample):
        #"""
        #Return the log likelihood of C{sample} under this distribution.
        #"""

        ## mise en place
        #cdef numpy.ndarray[double]       alpha_D  = self._alpha
        #cdef numpy.ndarray[numpy.uint_t] counts_D = sample

        #if counts_D.shape[0] != sample.shape[0]:
            #raise ValueError("shapes of alpha and counts arrays do not match")

        ## calculate
        #cdef size_t        d
        #cdef unsigned long n = 0

        #for d in xrange(counts_D.shape[0]):
            #n += counts_D[d]

        #cdef double psigm     = 0.0
        #cdef double sum_alpha = 0.0

        #for d in xrange(counts_D.shape[0]):
            #psigm     += ln_poch(alpha_D[d], counts_D[d])
            #sum_alpha += alpha_D[d]

        #return psigm - ln_poch(sum_alpha, n)

    #def given(self, samples):
        #"""
        #Return the conditional distribution.
        #"""

        #samples = numpy.asarray(samples, numpy.uint)

        #return DirichletCompoundMultinomial(self._alpha + numpy.sum(samples, 0), self._norm)

    #@property
    #def alpha(self):
        #"""
        #Return the Dirichlet parameter vector.
        #"""

        #return self._alpha

#cdef struct NormCount:
    #unsigned int first
    #double       second

#cdef class PreWallachRecurrence:
    #"""
    #Precomputed values used by the Wallach recurrence estimator.
    #"""

    #cdef NormCount*  c_dot
    #cdef size_t      c_dot_size
    #cdef NormCount** c_k
    #cdef size_t*     c_k_sizes

    #def __cinit__(self):
        #"""
        #Construct.
        #"""

    #def __dealloc__(self):
        #"""
        #Destruct.
        #"""

        ## FIXME don't leak memory

#def get_first(pair):
    #return pair[0]

#def pre_estimate_dcm_wallach_recurrence(
    #numpy.ndarray[numpy.uint_t, ndim = 2] counts_ND,
    #numpy.ndarray[double]                 weights_N,
    #):
    #"""
    #Precomputation for the Wallach DCM estimator.
    #"""

    ## mise en place
    #cdef size_t N = counts_ND.shape[0]
    #cdef size_t D = counts_ND.shape[1]

    ## precompute the unweighted norms
    #c_dot_map = {}
    #c_k_maps  = [{} for _ in xrange(D)]

    #cdef double        previous
    #cdef unsigned int  l1_norm
    #cdef unsigned long count

    #for n in xrange(N):
        #l1_norm = 0

        #for d in xrange(D):
            #count    = counts_ND[n, d]
            #l1_norm += count

            #if count > 0:
                #previous           = c_k_maps[d].get(count, 0)
                #c_k_maps[d][count] = previous + weights_N[n]

        #if l1_norm > 0:
            #previous           = c_dot_map.get(l1_norm, 0)
            #c_dot_map[l1_norm] = previous + weights_N[n]

    ## arrange them for estimation
    #cdef PreWallachRecurrence pre = PreWallachRecurrence()

    #pre.c_dot_size  = len(c_dot_map)
    #pre.c_dot       = <NormCount*>malloc(len(c_dot_map) * sizeof(NormCount))
    #c_dot_map_items = sorted(c_dot_map.iteritems(), key = get_first, reverse = True)

    #for (i, (first, second)) in enumerate(c_dot_map_items):
        #pre.c_dot[i].first  = first
        #pre.c_dot[i].second = second

    #pre.c_k       = <NormCount**>malloc(D * sizeof(NormCount*))
    #pre.c_k_sizes = <size_t*>malloc(D * sizeof(size_t))

    #for d in xrange(D):
        #pre.c_k_sizes[d] = len(c_k_maps[d])
        #pre.c_k[d]       = <NormCount*>malloc(len(c_k_maps[d]) * sizeof(NormCount))

        #c_k_maps_d = sorted(c_k_maps[d].iteritems(), key = get_first, reverse = True)

        #for (i, (first, second)) in enumerate(c_k_maps_d):
            #pre.c_k[d][i].first  = first
            #pre.c_k[d][i].second = second

    #return pre

#def estimate_dcm_wallach_recurrence(
    #numpy.ndarray[numpy.uint_t, ndim = 2] counts_ND,
    #numpy.ndarray[double]                 weights_N,
    #double threshold,
    #unsigned int cutoff,
    #):
    #"""
    #Estimate the maximum likelihood DCM distribution.

    #Uses the fixed-point estimator of Hanna Wallach that exploits digamma recurrence.
    #"""

    ## mise en place
    #cdef size_t N = counts_ND.shape[0]
    #cdef size_t D = counts_ND.shape[1]

    #assert weights_N.shape[0] == N

    #cdef numpy.ndarray[double] alpha_D = numpy.ones(D)

    ## precompute the weighted norms
    #cdef PreWallachRecurrence pre = pre_estimate_dcm_wallach_recurrence(counts_ND, weights_N)

    ## run the fixed-point iteration to convergence
    #cdef size_t    i
    #cdef size_t    k
    #cdef size_t    d
    #cdef size_t    d_
    #cdef double    ratio
    #cdef double    sum_alpha
    #cdef double    wallach_s
    #cdef double    wallach_s_k
    #cdef double    wallach_d
    #cdef double    wallach_d_k
    #cdef double    difference = threshold
    #cdef NormCount c_k_item

    #while cutoff > 0 and difference >= threshold:
        ## countdown
        #cutoff -= 1

        ## compute sum_alpha
        #sum_alpha = 0.0

        #for d in xrange(D):
            #sum_alpha += alpha_D[d]

        #if sum_alpha > 1e6:
            ## FIXME a bit of a hack
            #break

        ## compute the denominator; wallach_* are named as in (Wallach 2008)
        #wallach_s = 0.0
        #wallach_d = 0.0
        #k         = 0

        #for i from pre.c_dot_size >= i > 0:
            #while k < pre.c_dot[i - 1].first:
                #wallach_d += 1.0 / (k + sum_alpha)
                #k         += 1

            #wallach_s += pre.c_dot[i - 1].second * wallach_d

        ## compute the numerator and update alpha
        #difference = 0.0

        #for d in xrange(D):
            ## compute the numerator
            #wallach_s_k = 0.0
            #wallach_d_k = 0.0
            #k           = 0

            #for i from pre.c_k_sizes[d] >= i > 0:
                #c_k_item = pre.c_k[d][i - 1]

                #while k < c_k_item.first:
                    #wallach_d_k += 1.0 / (k + alpha_D[d])
                    #k           += 1

                #wallach_s_k += c_k_item.second * wallach_d_k

            ## update this dimension of alpha
            #ratio = wallach_s_k / wallach_s

            #alpha_D[d] *= ratio
            #difference += fabs(ratio - 1.0)

            #if alpha_D[d] < DBL_MIN:
                #alpha_D[d] = DBL_MIN

    #return alpha_D

#class WallachRecurrenceEstimator(Estimator):
    #"""
    #Estimate the parameters of a DCM distribution using Wallach's digamma
    #recurrence iteration.

    #Extended to allow sample weighting for expectation maximization in mixture
    #models.
    #"""

    #def __init__(self, norm = 1, threshold = 1e-5, cutoff = 1e3, epsilon = 1e-6):
        #"""
        #Initialize.
        #"""

        #self._norm      = norm
        #self._threshold = threshold
        #self._cutoff    = int(cutoff)
        #self._epsilon   = epsilon

    #def estimate(self, samples, random = numpy.random, weights = None):
        #"""
        #Return the estimated maximum likelihood distribution.
        #"""

        ## parameters
        #samples = numpy.asarray(samples, numpy.uint)

        #if weights is None:
            #weights = numpy.ones(samples.shape[0])
        #else:
            #weights = numpy.asarray(weights)

        ## counts are available; estimate
        #alpha = \
            #estimate_dcm_wallach_recurrence(
                #samples,
                #weights,
                #self._threshold,
                #self._cutoff,
                #)

        ## smooth, if requested
        #if self._epsilon is not None:
            #alpha += max(numpy.min(alpha), self._epsilon) * 1e-2

        ## done
        #return DirichletCompoundMultinomial(alpha, self._norm)

## select the "best" estimator
#DCM_Estimator = WallachRecurrenceEstimator

