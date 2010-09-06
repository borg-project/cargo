"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.gsl.errno cimport (
    GSL_SUCCESS,
    gsl_strerror,
    )

cdef extern from "gsl/gsl_sf_result.h":
    ctypedef struct gsl_sf_result:
        double val
        double err

cdef extern from "gsl/gsl_sf.h":
    int gsl_sf_psi_e    (double x,           gsl_sf_result* result)
    int gsl_sf_log_e    (double v,           gsl_sf_result* result)
    int gsl_sf_lnpoch_e (double a, double x, gsl_sf_result* result)
    int gsl_sf_lngamma_e(double v,           gsl_sf_result* result)

cpdef double psi(double x) except? -1:
    """
    Compute the psi function.
    """

    cdef gsl_sf_result result
    cdef int           status = gsl_sf_psi_e(x, &result)

    if status == GSL_SUCCESS:
        return result.val
    else:
        raise RuntimeError("%s (x = %f)" % (gsl_strerror(status), x))

        return -1

cpdef double log(double v) except? -1:
    """
    Compute the natural log of the Pochhammer function.
    """

    cdef gsl_sf_result result
    cdef int           status = gsl_sf_log_e(v, &result)

    if status == GSL_SUCCESS:
        return result.val
    else:
        raise RuntimeError("%s (v = %f)" % (gsl_strerror(status), v))

        return -1

cpdef double ln_gamma(double v) except? -1:
    """
    Compute the natural log of the Pochhammer function.
    """

    cdef gsl_sf_result result
    cdef int           status = gsl_sf_lngamma_e(v, &result)

    if status == GSL_SUCCESS:
        return result.val
    else:
        raise RuntimeError("%s (v = %f)" % (gsl_strerror(status), v))

        return -1

cpdef double ln_poch(double a, double x) except? -1:
    """
    Compute the natural log of the Pochhammer function.
    """

    cdef gsl_sf_result result
    cdef int           status = gsl_sf_lnpoch_e(a, x, &result)

    if status == GSL_SUCCESS:
        return result.val
    else:
        raise ValueError("%s (a = %f; x = %f)" % (gsl_strerror(status), a, x))

        return -1

