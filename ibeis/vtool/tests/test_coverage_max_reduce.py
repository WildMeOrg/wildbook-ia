#Is it possible to use numpy.ufunc.reduce over an iterator of ndarrays?

#I have a generator function that yields ndarrays (all of the same shape and dtype) and I would like to find the maximum value at each index.

#Currently I have code that looks like this:


def main():
    import numpy as np
    import cv2

    shape = (250, 300)
    dsize = shape[::-1]

    affmat_list = np.array([
        [[  1.57351554e+00,   0.00000000e+00,   1.09061039e+02],
         [ -3.61827926e-01,   7.46059970e-01,   2.50669551e+01]],
        [[  3.05754491e+00,   0.00000000e+00,   8.28024922e+01],
         [ -2.13866309e-01,   1.72124200e+00,   1.72744669e+02]],
        [[  2.58008254e+00,   0.00000000e+00,   1.52155447e+02],
         [ -2.08041241e+00,   2.46195663e+00,   1.09493821e+02]],
        [[  2.01791864e+00,   0.00000000e+00,   2.45704669e+02],
         [ -1.07590956e+00,   3.33499949e+00,   1.66233498e+02]],
        [[  3.32012638e+00,   0.00000000e+00,   1.03847866e+02],
         [ -2.36557589e+00,   3.02063109e+00,   1.59907802e+02]],
        [[  4.94371474e+00,   0.00000000e+00,   7.92717193e+01],
         [ -2.67846198e+00,   3.66854256e+00,   1.47888210e+02]]])

    fx2_score = np.ones(len(affmat_list))

    patch = np.array([
        [ 0.0014,  0.0016,  0.0017,  0.0019,  0.0020, 0.0021,  0.0022,  0.0023,  0.0023,  0.0023, 0.0023,  0.0023,  0.0022,  0.0021,  0.0020, 0.0019,  0.0017,  0.0016,  0.0014],
        [ 0.0016,  0.0017,  0.0019,  0.0021,  0.0022, 0.0023,  0.0024,  0.0025,  0.0026,  0.0026, 0.0026,  0.0025,  0.0024,  0.0023,  0.0022, 0.0021,  0.0019,  0.0017,  0.0016],
        [ 0.0017,  0.0019,  0.0021,  0.0023,  0.0024, 0.0026,  0.0027,  0.0028,  0.0028,  0.0028, 0.0028,  0.0028,  0.0027,  0.0026,  0.0024, 0.0023,  0.0021,  0.0019,  0.0017],
        [ 0.0019,  0.0021,  0.0023,  0.0025,  0.0026, 0.0028,  0.0029,  0.0030,  0.0031,  0.0031, 0.0031,  0.0030,  0.0029,  0.0028,  0.0026, 0.0025,  0.0023,  0.0021,  0.0019],
        [ 0.0020,  0.0022,  0.0024,  0.0026,  0.0028, 0.0030,  0.0031,  0.0032,  0.0033,  0.0033, 0.0033,  0.0032,  0.0031,  0.0030,  0.0028, 0.0026,  0.0024,  0.0022,  0.0020],
        [ 0.0021,  0.0023,  0.0026,  0.0028,  0.0030, 0.0032,  0.0033,  0.0034,  0.0035,  0.0035, 0.0035,  0.0034,  0.0033,  0.0032,  0.0030, 0.0028,  0.0026,  0.0023,  0.0021],
        [ 0.0022,  0.0024,  0.0027,  0.0029,  0.0031, 0.0033,  0.0034,  0.0036,  0.0036,  0.0036, 0.0036,  0.0036,  0.0034,  0.0033,  0.0031, 0.0029,  0.0027,  0.0024,  0.0022],
        [ 0.0023,  0.0025,  0.0028,  0.0030,  0.0032, 0.0034,  0.0036,  0.0037,  0.0037,  0.0038, 0.0037,  0.0037,  0.0036,  0.0034,  0.0032, 0.0030,  0.0028,  0.0025,  0.0023],
        [ 0.0023,  0.0026,  0.0028,  0.0031,  0.0033, 0.0035,  0.0036,  0.0037,  0.0038,  0.0038, 0.0038,  0.0037,  0.0036,  0.0035,  0.0033, 0.0031,  0.0028,  0.0026,  0.0023],
        [ 0.0023,  0.0026,  0.0028,  0.0031,  0.0033, 0.0035,  0.0036,  0.0038,  0.0038,  0.0039, 0.0038,  0.0038,  0.0036,  0.0035,  0.0033, 0.0031,  0.0028,  0.0026,  0.0023],
        [ 0.0023,  0.0026,  0.0028,  0.0031,  0.0033, 0.0035,  0.0036,  0.0037,  0.0038,  0.0038, 0.0038,  0.0037,  0.0036,  0.0035,  0.0033, 0.0031,  0.0028,  0.0026,  0.0023],
        [ 0.0023,  0.0025,  0.0028,  0.0030,  0.0032, 0.0034,  0.0036,  0.0037,  0.0037,  0.0038, 0.0037,  0.0037,  0.0036,  0.0034,  0.0032, 0.0030,  0.0028,  0.0025,  0.0023],
        [ 0.0022,  0.0024,  0.0027,  0.0029,  0.0031, 0.0033,  0.0034,  0.0036,  0.0036,  0.0036, 0.0036,  0.0036,  0.0034,  0.0033,  0.0031, 0.0029,  0.0027,  0.0024,  0.0022],
        [ 0.0021,  0.0023,  0.0026,  0.0028,  0.0030, 0.0032,  0.0033,  0.0034,  0.0035,  0.0035, 0.0035,  0.0034,  0.0033,  0.0032,  0.0030, 0.0028,  0.0026,  0.0023,  0.0021],
        [ 0.0020,  0.0022,  0.0024,  0.0026,  0.0028, 0.0030,  0.0031,  0.0032,  0.0033,  0.0033, 0.0033,  0.0032,  0.0031,  0.0030,  0.0028, 0.0026,  0.0024,  0.0022,  0.0020],
        [ 0.0019,  0.0021,  0.0023,  0.0025,  0.0026, 0.0028,  0.0029,  0.0030,  0.0031,  0.0031, 0.0031,  0.0030,  0.0029,  0.0028,  0.0026, 0.0025,  0.0023,  0.0021,  0.0019],
        [ 0.0017,  0.0019,  0.0021,  0.0023,  0.0024, 0.0026,  0.0027,  0.0028,  0.0028,  0.0028, 0.0028,  0.0028,  0.0027,  0.0026,  0.0024, 0.0023,  0.0021,  0.0019,  0.0017],
        [ 0.0016,  0.0017,  0.0019,  0.0021,  0.0022, 0.0023,  0.0024,  0.0025,  0.0026,  0.0026, 0.0026,  0.0025,  0.0024,  0.0023,  0.0022, 0.0021,  0.0019,  0.0017,  0.0016],
        [ 0.0014,  0.0016,  0.0017,  0.0019,  0.0020, 0.0021,  0.0022,  0.0023,  0.0023,  0.0023, 0.0023,  0.0023,  0.0022,  0.0021,  0.0020, 0.0019,  0.0017,  0.0016,  0.0014]
    ])

    def warped_patch_generator():
        padded_patch = np.zeros(shape, dtype=np.float32)
        patch_h, patch_w = patch.shape
        warped = np.zeros(shape, dtype=np.float32)
        for count, (M, score) in enumerate(zip(affmat_list, fx2_score)):
            print(count)
            np.multiply(patch, score, out=padded_patch[:patch.shape[0], :patch.shape[1]] )
            cv2.warpAffine(padded_patch, M, dsize, dst=warped,
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
            yield warped
            #yield warped

    print("THREE")
    from six.moves import reduce
    import functools
    dstimg3 = np.zeros(shape, dtype=np.float32)
    maximum_partial = functools.partial(np.maximum, out=dstimg3)
    dstimg3 = reduce(maximum_partial, warped_patch_generator())

    print("ONE")
    dstimg1 = np.zeros(shape, dtype=np.float32)
    print("ONE")
    for warped in warped_patch_generator():
        #dstimg1 = np.maximum(dstimg1, warped)
        np.maximum(dstimg1, warped, out=dstimg1)

    print("FOUR")
    input_copy_ = np.array([w.copy() for w in warped_patch_generator()])
    dstimg4 = input_copy_.max(0)

    print("TWO")
    dstimg2 = np.zeros(shape, dtype=np.float32)
    input_iter_ = list((w for w in warped_patch_generator()))
    np.maximum.reduce(input_iter_, axis=0, dtype=np.float32, out=dstimg2)

    x = np.where(dstimg1.ravel() != dstimg2.ravel())[0]
    print(dstimg2.take(x))
    print(dstimg1.take(x))
    np.allclose(dstimg1, dstimg2)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(dstimg1)
    plt.subplot(222)
    plt.imshow(dstimg2)
    plt.subplot(223)
    plt.imshow(dstimg3)
    plt.subplot(224)
    plt.imshow(dstimg4)

    plt.show()


if __name__ == '__main__':
    main()

#I would have thought that I would be allowed to write something like this:
#    dstimg = np.maximum.reduce(warped_patch_generator())
