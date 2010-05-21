"""
@author: Bryan Silverthorn <bcs@cargo-cult.org>
"""

def get_color_list(n, saturation = 1.0, value = 0.75):
    """
    Get an optimally-spaced list of (RGB) color values.
    """

    import numpy

    from matplotlib.colors import hsv_to_rgb

    hsv_colors = numpy.empty((1, n, 3))

    hsv_colors[:, :, 0] = numpy.r_[0.0:1.0 - 1.0 / n:complex(0, n)]
    hsv_colors[:, :, 1] = 1.0
    hsv_colors[:, :, 2] = 0.75

    (rgb_colors,) = hsv_to_rgb(hsv_colors)

    return rgb_colors

