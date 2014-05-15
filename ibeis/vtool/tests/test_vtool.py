from __future__ import absolute_import, division, print_function
import vtool.keypoint as ktool
import vtool.linalg as ltool
import numpy as np
import utool
from vtool.tests import dummy


def test_get_invR_mats_orientation():
    theta1 = np.tau / 8
    theta2 = -np.tau / 8
    theta3 = 0
    theta4 = 7 * np.tau / 8

    invV_mats = dummy.get_dummy_invV_mats()

    def R_mats(theta):
        return np.array([ltool.rotation_mat2x2(theta) for _ in xrange(len(invV_mats))])

    def test_rots(theta):
        invVR_mats = ltool.matrix_multiply(invV_mats, R_mats(theta))
        _oris = ktool.get_invVR_mats_oris(invVR_mats)
        print('________')
        print('theta = %r' % (theta % np.tau,))
        print('b / a = %r' % (_oris,))
        passed, error = utool.almost_eq(_oris, theta % np.tau, ret_error=True)
        try:
            assert np.all(passed)
        except AssertionError as ex:
            utool.printex(ex, 'rotation unequal', key_list=['passed',
                                                            'error'])

    test_rots(theta1)
    test_rots(theta2)
    test_rots(theta3)
    test_rots(theta4)


if __name__ == '__main__':
    test_get_invR_mats_orientation()
