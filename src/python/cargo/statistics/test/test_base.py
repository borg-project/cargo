"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

import numpy

from nose.tools import assert_almost_equal

def test_engine_ll_complex_model():
    """
    Test log-likelihood computation under a complex model.
    """

    from cargo.log import get_logger

    get_logger("cargo.llvm.constructs", level = "DEBUG")

    # test the model
    from cargo.statistics import (
        Tuple,
        ModelEngine,
        MixedBinomial,
        FiniteMixture,
        )

    model  = FiniteMixture(Tuple([(MixedBinomial(), 128)]), 8)
    engine = ModelEngine(model)

    # generate some fake parameters
    sample    = numpy.empty((), model.sample_dtype   )
    parameter = numpy.empty((), model.parameter_dtype)

    sample["d0"]["n"] = 4
    sample["d0"]["k"] = 1

    parameter["p"]       = 1.0 / 8.0
    parameter["c"]["d0"] = 0.5

    assert_almost_equal(engine.ll(parameter, sample), -177.445678223)

