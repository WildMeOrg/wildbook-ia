import colorsys
import numpy as np
from plottool_ibeis import color_funcs

def TEST_COLORSYS():
    rgb = [1.0, .2, .1]
    hsv = colorsys.rgb_to_hsv(*rgb)
    rgb2 = colorsys.hsv_to_rgb(*hsv)

    new_rgb = color_funcs.brighten_rgb(rgb, 0.0)

    rgba = np.array([ 0.,  1.,  0.,  1.])
    rbg = rgba
    new_rgba1 = color_funcs.brighten_rgb(rgba, 0.0)
    new_rgba2 = color_funcs.brighten_rgb(rgba, 0.5)


if __name__ == '__main__':
    TEST_COLORSYS()
