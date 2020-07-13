# -*- coding: utf-8 -*-
import utool as ut


def impaint_mask(img, label_colors=None, init_mask=None, init_label=None):
    r"""
    CommandLine:
        python -m wbia.plottool.interact_impaint --test-impaint_mask

    References:
        http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html

    TODO: Slider for transparency
    TODO: Label selector

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.interact_impaint import *  # NOQA
        >>> import utool as ut
        >>> import vtool as vt
        >>> img_fpath = ut.grab_test_imgpath('lena.png')
        >>> img = vt.imread(img_fpath)
        >>> label_colors = [255, 200, 100, 0]
        >>> result = impaint_mask(img, label_colors)
        >>> # verify results
        >>> print(result)
    """
    import cv2
    import numpy as np

    print('begining impaint mask. c=circle, r=rect')

    globals_ = dict(
        drawing=False,  # true if mouse is pressed
        mode='rect',  # if True, draw rectangle. Press 'm' to toggle to curve
        color=255,
        fgcolor=255,
        bgcolor=0,
        label_index=0,
        radius=25,
        transparency=0.25,
        ix=-1,
        iy=-1,
    )

    # mouse callback function
    def draw_shape(x, y):
        keys = ['mode', 'ix', 'iy', 'color', 'radius']
        mode, ix, iy, color, radius = ut.dict_take(globals_, keys)
        if mode == 'rect':
            cv2.rectangle(mask, (ix, iy), (x, y), color, -1)
        elif mode == 'circ':
            cv2.circle(mask, (x, y), radius, color, -1)

    def mouse_callback(event, x, y, flags, param):
        # keys =  ['drawing', 'mode', 'ix', 'iy', 'color']
        # drawing, mode, ix, iy, color = ut.dict_take(globals_, keys)

        if event in [cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDOWN]:
            globals_['drawing'] = True
            globals_['ix'], globals_['iy'] = x, y
            if event == cv2.EVENT_RBUTTONDOWN:
                globals_['color'] = globals_['bgcolor']
            elif event == cv2.EVENT_LBUTTONDOWN:
                globals_['color'] = globals_['fgcolor']
        elif event == cv2.EVENT_MOUSEMOVE:
            if globals_['drawing'] is True:
                draw_shape(x, y)
        elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
            globals_['drawing'] = False
            draw_shape(x, y)
            if event == cv2.EVENT_RBUTTONUP:
                globals_['color'] = globals_['fgcolor']
            elif event == cv2.EVENT_LBUTTONUP:
                pass
                # globals_['color'] = 255

    if label_colors is None:
        color_list = [255, 0]
    else:
        color_list = label_colors[:]

    # Choose colors/labels to start with
    if init_label is None:
        init_color = 0
    else:
        init_color = color_list[init_label]

    print('color_list = %r' % (color_list,))
    print('init_color=%r' % (init_color,))

    title = 'masking image'
    if init_mask is not None:
        try:
            mask = init_mask[:, :, 0].copy()
        except Exception:
            mask = init_mask.copy()
    else:
        mask = np.zeros(img.shape[0:2], np.uint8) + init_color
    transparent_mask = np.zeros(img.shape[0:2], np.float32)
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_callback)

    print('Valid Keys: r,c,t,l,q')
    while 1:
        # Blend images
        transparency = globals_['transparency']
        # Move from 0 to 1
        np.divide(mask, 255.0, out=transparent_mask)
        # Unmask room for a bit of transparency
        np.multiply(transparent_mask, (1.0 - transparency), out=transparent_mask)
        # Add a bit of transparency
        np.add(transparent_mask, transparency, out=transparent_mask)
        # Multiply the image by the transparency mask
        masked_image = (img * transparent_mask[:, :, None]).astype(np.uint8)
        cv2.imshow(title, masked_image)
        keycode = cv2.waitKey(1) & 0xFF
        if keycode == ord('r'):
            globals_['mode'] = 'rect'
        if keycode == ord('c'):
            globals_['mode'] = 'circ'
        if keycode == ord('t'):
            globals_['transparency'] = (globals_['transparency'] + 0.25) % 1.0
        if keycode == ord('l'):
            globals_['label_index'] = (globals_['label_index'] + 1) % len(color_list)
            globals_['fgcolor'] = color_list[globals_['label_index']]
            print('fgcolor = %r' % (globals_['fgcolor'],))
        if keycode == ord('q') or keycode == 27:
            break

    cv2.destroyAllWindows()
    return mask


def cached_impaint(
    bgr_img,
    cached_mask_fpath=None,
    label_colors=None,
    init_mask=None,
    aug=False,
    refine=False,
):
    import vtool as vt

    if cached_mask_fpath is None:
        cached_mask_fpath = 'image_' + ut.hashstr_arr(bgr_img) + '.png'
    if aug:
        cached_mask_fpath += '.' + ut.hashstr_arr(bgr_img)
        if label_colors is not None:
            cached_mask_fpath += ut.hashstr_arr(label_colors)
        cached_mask_fpath += '.png'
    # cached_mask_fpath = 'tmp_mask.png'
    if refine or not ut.checkpath(cached_mask_fpath):
        if refine and ut.checkpath(cached_mask_fpath):
            if init_mask is None:
                init_mask = vt.imread(cached_mask_fpath, grayscale=True)
        custom_mask = impaint_mask(
            bgr_img, label_colors=label_colors, init_mask=init_mask
        )
        vt.imwrite(cached_mask_fpath, custom_mask)
    else:
        custom_mask = vt.imread(cached_mask_fpath, grayscale=True)
    return custom_mask


def demo():
    r"""
    CommandLine:
        python -m wbia.plottool.interact_impaint --test-demo

    References:
        http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_mouse_handling/py_mouse_handling.html

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.interact_impaint import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = demo()
        >>> # verify results
        >>> print(result)
    """
    import cv2
    import numpy as np

    globals_ = dict(
        drawing=False,  # true if mouse is pressed
        mode=False,  # if True, draw rectangle. Press 'm' to toggle to curve
        ix=-1,
        iy=-1,
    )

    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            globals_['drawing'] = True
            globals_['ix'], globals_['iy'] = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if globals_['drawing'] is True:
                if globals_['mode'] is True:
                    cv2.rectangle(
                        img, (globals_['ix'], globals_['iy']), (x, y), (0, 255, 0), -1
                    )
                else:
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            globals_['drawing'] = False
            if globals_['mode'] is True:
                cv2.rectangle(
                    img, (globals_['ix'], globals_['iy']), (x, y), (0, 255, 0), -1
                )
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while 1:
        cv2.imshow('image', img)
        keycode = cv2.waitKey(1) & 0xFF
        if keycode == ord('m'):
            globals_['mode'] = not globals_['mode']
        elif keycode == 27:
            break

    cv2.destroyAllWindows()
