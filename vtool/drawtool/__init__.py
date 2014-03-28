from __future__ import division, print_function

import mpl_keypoint  # NOQA
import mpl_keypoint as mpl_kp
import mpl_sift as mpl_sift
# The other module shouldn't exist. Functions in it need to be organized
from other import *  # NOQA

draw_keypoints = mpl_kp.draw_keypoints
draw_sifts = mpl_sift.draw_sifts
