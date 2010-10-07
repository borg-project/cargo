"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

from cargo.gsl.errno cimport (
    GSL_SUCCESS,
    gsl_strerror,
    )

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_binomial_pdf(unsigned int k, double p, unsigned int n)

cpdef double binomial_pdf(unsigned int k, double p, unsigned int n) except? -1.0:
    """
    Compute the PDF of the binomial distribution.
    """

    return gsl_ran_binomial_pdf(k, p, n)

