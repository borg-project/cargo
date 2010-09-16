"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def test_ln_poch():
	"""
	Test computation of the natural log of the Pochhammer function.
	"""

	from nose.tools   import assert_equal
	from cargo.gsl.sf import ln_poch

	assert_equal(ln_poch(1, 2), 0.69314718055994529)
	assert_equal(ln_poch(2, 3), 3.1780538303479458)

