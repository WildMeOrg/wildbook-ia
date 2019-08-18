
"""
References:
    % Single-image noise level estimation for blind denoising.
    % http://www.ok.ctrl.titech.ac.jp/res/NLE/TIP2013-noise-level-estimation06607209.pdfhttp://www.ok.ctrl.titech.ac.jp/res/NLE/TIP2013-noise-level-estimation06607209.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import ubelt as ub
import numpy as np
import cv2


def compute_average_contrast(img):
    """
    CommandLine:
        python -m vtool.quality_classifier --exec-compute_average_contrast --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from vtool.quality_classifier import *  # NOQA
        >>> import vtool as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath, grayscale=True)
        >>> average_contrast, gradmag_sqrd = compute_average_contrast(img)
        >>> # xdoctest: +REQUIRES(module:plottool)
        >>> import plottool as pt
        >>> # xdoctest: +REQUIRES(--show)
        >>> pt.figure(fnum=1)
        >>> pt.plt.imshow(gradmag_sqrd)
        >>> ut.show_if_requested()
    """
    ksize = 5
    assert img.dtype == np.uint8
    img_ = img.astype(np.float64) / 255.0
    gradx = cv2.Sobel(img_, cv2.CV_64F, 1, 0, ksize=ksize)
    grady = cv2.Sobel(img_, cv2.CV_64F, 0, 1, ksize=ksize)
    gradmag_sqrd = (gradx ** 2) + (grady ** 2)
    total_contrast = gradmag_sqrd.sum()

    average_contrast = total_contrast / np.prod(gradmag_sqrd.shape)
    return average_contrast, gradmag_sqrd


def contrast_measures(img):
    ksize = 5
    assert img.dtype == np.uint8
    img_ = img.astype(np.float64) / 255.0
    gradx = cv2.Sobel(img_, cv2.CV_64F, 1, 0, ksize=ksize)
    grady = cv2.Sobel(img_, cv2.CV_64F, 0, 1, ksize=ksize)
    gradmag_sqrd = (gradx ** 2) + (grady ** 2)
    total_contrast = gradmag_sqrd.sum()

    average_contrast = total_contrast / np.prod(gradmag_sqrd.shape)

    std_dev = img.std()

    bin_count, bin_lbls = np.histogram(img)
    entropy = -np.sum(bin_count * np.log2(bin_count))

    import skimage.morphology
    import skimage.filters.rank
    entropyimg = skimage.filters.rank.entropy(img, skimage.morphology.disk(2))
    ave_local_entropy = entropyimg.sum() / np.prod(entropyimg.shape)

    contrast_measures = [average_contrast, std_dev, entropy, ave_local_entropy]
    return contrast_measures


def test_average_contrast():
    import vtool as vt
    ut.get_valid_test_imgkeys()
    img_fpath_list = [ut.grab_test_imgpath(key) for key in ut.get_valid_test_imgkeys()]
    img_list = [vt.imread(img, grayscale=True) for img in img_fpath_list]
    avecontrast_list = np.array([compute_average_contrast(img) for img in img_list])
    import plottool as pt
    nCols = len(img_list)
    fnum = None
    if fnum is None:
        fnum = pt.next_fnum()
    pt.figure(fnum=fnum, pnum=(2, 1, 1))
    sortx = avecontrast_list.argsort()
    y_list = avecontrast_list[sortx]
    x_list = np.arange(0, nCols) + .5
    pt.plot(x_list, y_list, 'bo-')
    sorted_imgs = list(ub.take(img_list, sortx))

    for px, img in ub.ProgIter(enumerate(sorted_imgs, start=1)):
        pt.imshow(img, fnum=fnum, pnum=(2, nCols, nCols + px))


def fourier_devtest(img):
    r"""
    Args:
        img (ndarray[uint8_t, ndim=2]):  image data

    CommandLine:
        python -m vtool.quality_classifier --test-fourier_devtest --show

    References:
        http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html
        http://cns-alumni.bu.edu/~slehar/fourier/fourier.html

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.quality_classifier import *  # NOQA
        >>> import vtool as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> img = vt.imread(img_fpath, grayscale=True)
        >>> magnitude_spectrum = fourier_devtest(img)
    """
    import plottool as pt
    def pad_img(img):
        rows, cols = img.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        right = ncols - cols
        bottom = nrows - rows
        bordertype = cv2.BORDER_CONSTANT
        nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, bordertype, value=0)
        return nimg

    def convert_to_fdomain(img):
        dft = cv2.dft(img.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        #dft_shift = np.fft.fftshift(dft)
        return dft

    def convert_from_fdomain(dft):
        img = cv2.idft(dft)
        img = cv2.magnitude(img[:, :, 0], img[:, :, 1])
        img /= img.max()
        return img * 255.0

    def get_fdomain_mag(dft_shift):
        magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        return magnitude_spectrum

    def imgstats(img):
        print('stats:')
        print('    dtype = %r ' % (img.dtype,))
        print('    ' + ut.get_stats_str(img, axis=None))

    nimg = pad_img(img)
    dft = convert_to_fdomain(nimg)
    #freq_domain = np.fft.fft2(img)
    #freq_domain_shift = np.fft.fftshift(freq_domain)

    rows, cols = nimg.shape
    crow, ccol = rows / 2 , cols / 2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    dft_mask = np.fft.ifftshift(np.fft.fftshift(dft) * mask)
    img_back = convert_from_fdomain(dft_mask)

    imgstats(dft)
    imgstats(mask)
    imgstats(nimg)
    imgstats(nimg)

    print('nimg.shape = %r' % (nimg.shape,))
    print('dft_shift.shape = %r' % (dft.shape,))

    if ut.show_was_requested():
        #import plottool as pt
        next_pnum = pt.make_pnum_nextgen(nRows=3, nCols=2)
        pt.imshow(nimg, pnum=next_pnum(), title='nimg')
        pt.imshow(20 * get_fdomain_mag(dft), pnum=next_pnum(), title='mag(f)')
        pt.imshow(20 * get_fdomain_mag(dft_mask), pnum=next_pnum(), title='dft_mask')
        pt.imshow(img_back, pnum=next_pnum(), title='img_back')
        pt.show_if_requested()


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool.quality_classifier
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
