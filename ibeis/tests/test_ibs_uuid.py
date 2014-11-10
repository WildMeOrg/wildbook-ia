#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
from PIL import Image
from ibeis.model.preproc import preproc_image
from uuid import UUID
from vtool.tests import grabdata
import hashlib
import multiprocessing
import numpy as np
import utool
import uuid
import PIL
import cv2
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_UUID]')

MACHINE_NAME = utool.get_computer_name()

TEST_TARGETS = utool.ddict(lambda: utool.ddict(dict))

# ---------------
# DEPENDS['C:\\Python27\\Lib\\site-packages\\PIL\\_imaging.pyd']['Ooo']
#       DLL Name: KERNEL32.dll
#       DLL Name: USER32.dll
#       DLL Name: GDI32.dll
#       DLL Name: python27.dll
#       DLL Name: MSVCR90.dll
# ---------------
# DEPENDS['C:\\Python27\\Lib\\site-packages\\PIL\\_imagingcms.pyd']['Ooo']
#       DLL Name: USER32.dll
#       DLL Name: GDI32.dll
#       DLL Name: python27.dll
#       DLL Name: MSVCR90.dll
#       DLL Name: KERNEL32.dll
# ---------------
# DEPENDS['C:\\Python27\\Lib\\site-packages\\PIL\\_imagingft.pyd']['Ooo']
#       DLL Name: python27.dll
#       DLL Name: MSVCR90.dll
#       DLL Name: KERNEL32.dll
# ---------------
# DEPENDS['C:\\Python27\\Lib\\site-packages\\PIL\\_imagingmath.pyd']['Ooo']
#       DLL Name: python27.dll
#       DLL Name: MSVCR90.dll
#       DLL Name: KERNEL32.dll
# ---------------
# DEPENDS['C:\\Python27\\Lib\\site-packages\\PIL\\_imagingtk.pyd']['Ooo']
#       DLL Name: tcl85.dll
#       DLL Name: tk85.dll
#       DLL Name: python27.dll
#       DLL Name: MSVCR90.dll
#       DLL Name: KERNEL32.dll
# ---------------
# DEPENDS['C:\\Python27\\Lib\\site-packages\\PIL\\_webp.pyd']['Ooo']
#       DLL Name: python27.dll
#       DLL Name: MSVCR90.dll
#       DLL Name: KERNEL32.dll
# ---------------
SPECS = {}
SPECS['Ooo'] = {
    'cv2_version': r'2.4.8',
    'pillow_version': r'2.4.0',
    'pil_version': r'1.1.7',
}

TEST_TARGETS['lena']['Ooo'] = {
    'gpath': r'C:/Users/joncrall/AppData/Roaming/utool/testdata\\lena.jpg',
    'npshape': r'(220, 220, 3)',
    'npsum': r'18615625',
    'uuid1': r'f2c4658f-a722-793b-8578-c392fd550888',
    'size': r'(220, 220)',
    'sum_': r'18615820',
    'img_bytes_': r'\xe2\x99W\xccib\xc8\x7fm\xbfXn\xb1cb ~~~TRUNCATED~~~ \xc8V\xb1',
    'hashbytes_16': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88',
    'hashbytes_20': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88\xac.sk',
    'bytes_sha1': r'<sha1 HASH object @ 0AF0BE80>',
    'uuid_': r'f2c4658f-a722-793b-8578-c392fd550888',
}
#----
TEST_TARGETS['jeff']['Ooo'] = {
    'gpath': r'C:/Users/joncrall/AppData/Roaming/utool/testdata\\jeff.png',
    'npshape': r'(400, 400, 3)',
    'npsum': r'111817836',
    'uuid1': r'aed981a2-4116-9936-6311-e46bd17e25de',
    'size': r'(400, 400)',
    'sum_': r'152617836',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde',
    'hashbytes_20': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde<\xb3\xd6\xe7',
    'bytes_sha1': r'<sha1 HASH object @ 0AF0BF60>',
    'uuid_': r'aed981a2-4116-9936-6311-e46bd17e25de',
}
#----
TEST_TARGETS['easy1']['Ooo'] = {
    'gpath': r'C:/Users/joncrall/AppData/Roaming/utool/testdata\\easy1.JPG',
    'npshape': r'(715, 1047, 3)',
    'npsum': r'354513878',
    'uuid1': r'383dda50-f26b-200a-8baf-548e3ef88f9c',
    'size': r'(1047, 715)',
    'sum_': r'354510703',
    'img_bytes_': r'\xde\xe9\xfc\xdf\xea\xfe\xdf\xea\xfe ~~~TRUNCATED~~~ 9\x81\x85',
    'hashbytes_16': r'8=\xdaP\xf2k \n\x8b\xafT\x8e>\xf8\x8f\x9c',
    'hashbytes_20': r'8=\xdaP\xf2k \n\x8b\xafT\x8e>\xf8\x8f\x9c"\xdf\xcb\x08',
    'bytes_sha1': r'<sha1 HASH object @ 0AF123A0>',
    'uuid_': r'383dda50-f26b-200a-8baf-548e3ef88f9c',
}
#----
TEST_TARGETS['http://i.imgur.com/QqSkNZe.png']['Ooo'] = {
    'gpath': r'C:\\Users\\joncrall\\AppData\\Roaming\\utool\\QqSkNZe.png',
    'npshape': r'(386, 564, 3)',
    'npsum': r'107691325',
    'uuid1': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
    'size': r'(564, 386)',
    'sum_': r'107691325',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e',
    'hashbytes_20': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e\xc5_H$',
    'bytes_sha1': r'<sha1 HASH object @ 0AF0BFA0>',
    'uuid_': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
}


TEST_TARGETS['lena']['Ooo'] = {
    'gpath': 'C:/Users/joncrall/AppData/Roaming/utool/testdata\\lena.jpg',
    'npshape': (220, 220, 3),
    'npsum': 18615625,
    'uuid1': UUID('f2c4658f-a722-793b-8578-c392fd550888'),
    'size': (220, 220),
    'sum_': 18615820,
    'img_bytes_': r'\xe2\x99W\xccib\xc8\x7fm\xbfXn\xb1cb ~~~TRUNCATED~~~ \xc8V\xb1',
    'hashbytes_16': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88',
    'hashbytes_20': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88\xac.sk',
    'bytes_sha1': '<sha1 HASH object @ 0AEF5240>',
    'uuid_': UUID('f2c4658f-a722-793b-8578-c392fd550888'),
}


TEST_TARGETS['lena']['hyrule'] = {
    'gpath':         '/home/joncrall/.config/utool/testdata/lena.jpg',
    'uuid': UUID('d3daf98d-3035-65b7-2ff4-ca4076ab0cf1'),
}

TEST_TARGETS['lena']['hendrik-vostro'] = {
    'gpath':         '/home/joncrall/.config/utool/testdata/lena.jpg',
    'uuid': UUID('d3daf98d-3035-65b7-2ff4-ca4076ab0cf1'),
}
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imaging.so']['Hyrule']
# 	linux-vdso.so.1 =>  (0x00007fffdc4bb000)
# 	libjpeg.so.8 => /usr/lib/x86_64-linux-gnu/libjpeg.so.8 (0x00007f13c40d1000)
# 	libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f13c3eba000)
# 	libtiff.so.4 => /usr/lib/x86_64-linux-gnu/libtiff.so.4 (0x00007f13c3c55000)
# 	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f13c3a38000)
# 	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f13c3678000)
# 	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f13c337b000)
# 	/lib64/ld-linux-x86-64.so.2 (0x00007f13c459d000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imagingft.so']['Hyrule']
# 	linux-vdso.so.1 =>  (0x00007fffde5ff000)
# 	libfreetype.so.6 => /usr/lib/x86_64-linux-gnu/libfreetype.so.6 (0x00007fc800ca1000)
# 	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fc800a84000)
# 	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fc8006c3000)
# 	libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fc8004ac000)
# 	/lib64/ld-linux-x86-64.so.2 (0x00007fc801175000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imagingmath.so']['Hyrule']
# 	linux-vdso.so.1 =>  (0x00007fff1b7ff000)
# 	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fc4be8ab000)
# 	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fc4be4eb000)
# 	/lib64/ld-linux-x86-64.so.2 (0x00007fc4bed00000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imagingtk.so']['Hyrule']
# 	linux-vdso.so.1 =>  (0x00007ffffe7ff000)
# 	libtcl8.5.so.0 => /usr/lib/libtcl8.5.so.0 (0x00007f090b4a5000)
# 	libtk8.5.so.0 => /usr/lib/libtk8.5.so.0 (0x00007f090b160000)
# 	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f090af42000)
# 	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f090ab82000)
# 	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f090a97e000)
# 	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f090a681000)
# 	libX11.so.6 => /usr/lib/x86_64-linux-gnu/libX11.so.6 (0x00007f090a34c000)
# 	libXss.so.1 => /usr/lib/x86_64-linux-gnu/libXss.so.1 (0x00007f090a148000)
# 	libXft.so.2 => /usr/lib/x86_64-linux-gnu/libXft.so.2 (0x00007f0909f32000)
# 	libfontconfig.so.1 => /usr/lib/x86_64-linux-gnu/libfontconfig.so.1 (0x00007f0909cfc000)
# 	/lib64/ld-linux-x86-64.so.2 (0x00007f090b9f4000)
# 	libxcb.so.1 => /usr/lib/x86_64-linux-gnu/libxcb.so.1 (0x00007f0909ade000)
# 	libXext.so.6 => /usr/lib/x86_64-linux-gnu/libXext.so.6 (0x00007f09098cc000)
# 	libfreetype.so.6 => /usr/lib/x86_64-linux-gnu/libfreetype.so.6 (0x00007f0909630000)
# 	libXrender.so.1 => /usr/lib/x86_64-linux-gnu/libXrender.so.1 (0x00007f0909426000)
# 	libexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007f09091fb000)
# 	libXau.so.6 => /usr/lib/x86_64-linux-gnu/libXau.so.6 (0x00007f0908ff8000)
# 	libXdmcp.so.6 => /usr/lib/x86_64-linux-gnu/libXdmcp.so.6 (0x00007f0908df2000)
# 	libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f0908bda000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/cv2.so']['Hyrule']
# 	linux-vdso.so.1 =>  (0x00007fff7d7f6000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007f0480083000)
# 	libopencv_core.so.2.4 => /usr/local/lib/libopencv_core.so.2.4 (0x00007f047fc40000)
# 	libopencv_flann.so.2.4 => /usr/local/lib/libopencv_flann.so.2.4 (0x00007f047f9d0000)
# 	libopencv_imgproc.so.2.4 => /usr/local/lib/libopencv_imgproc.so.2.4 (0x00007f047f50f000)
# 	libopencv_highgui.so.2.4 => /usr/local/lib/libopencv_highgui.so.2.4 (0x00007f047f1c1000)
# 	libopencv_features2d.so.2.4 => /usr/local/lib/libopencv_features2d.so.2.4 (0x00007f047ef15000)
# 	libopencv_calib3d.so.2.4 => /usr/local/lib/libopencv_calib3d.so.2.4 (0x00007f047ec76000)
# 	libopencv_ml.so.2.4 => /usr/local/lib/libopencv_ml.so.2.4 (0x00007f047e9f7000)
# 	libopencv_video.so.2.4 => /usr/local/lib/libopencv_video.so.2.4 (0x00007f047e79d000)
# 	libopencv_legacy.so.2.4 => /usr/local/lib/libopencv_legacy.so.2.4 (0x00007f047e47e000)
# 	libopencv_objdetect.so.2.4 => /usr/local/lib/libopencv_objdetect.so.2.4 (0x00007f047e1fe000)
# 	libopencv_photo.so.2.4 => /usr/local/lib/libopencv_photo.so.2.4 (0x00007f047dfde000)
# 	libopencv_nonfree.so.2.4 => /usr/local/lib/libopencv_nonfree.so.2.4 (0x00007f047dda8000)
# 	libopencv_contrib.so.2.4 => /usr/local/lib/libopencv_contrib.so.2.4 (0x00007f047dab6000)
# 	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f047d7b5000)
# 	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f047d4b9000)
# 	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f047d2a3000)
# 	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f047d085000)
# 	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f047ccc5000)
# 	libssl.so.1.0.0 => /lib/x86_64-linux-gnu/libssl.so.1.0.0 (0x00007f047ca67000)
# 	libcrypto.so.1.0.0 => /lib/x86_64-linux-gnu/libcrypto.so.1.0.0 (0x00007f047c68b000)
# 	libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f047c474000)
# 	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f047c270000)
# 	libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007f047c06c000)
# 	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f047be64000)
# 	libjpeg.so.8 => /usr/lib/x86_64-linux-gnu/libjpeg.so.8 (0x00007f047bc13000)
# 	libpng12.so.0 => /lib/x86_64-linux-gnu/libpng12.so.0 (0x00007f047b9eb000)
# 	libtiff.so.4 => /usr/lib/x86_64-linux-gnu/libtiff.so.4 (0x00007f047b787000)
# 	libjasper.so.1 => /usr/lib/x86_64-linux-gnu/libjasper.so.1 (0x00007f047b52f000)
# 	libgtk-x11-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgtk-x11-2.0.so.0 (0x00007f047aef5000)
# 	libgdk-x11-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgdk-x11-2.0.so.0 (0x00007f047ac43000)
# 	libgobject-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0 (0x00007f047a9f3000)
# 	libglib-2.0.so.0 => /lib/x86_64-linux-gnu/libglib-2.0.so.0 (0x00007f047a6d0000)
# 	libdc1394.so.22 => /usr/lib/x86_64-linux-gnu/libdc1394.so.22 (0x00007f047a45d000)
# 	libv4l1.so.0 => /usr/lib/x86_64-linux-gnu/libv4l1.so.0 (0x00007f047a256000)
# 	libavcodec.so.53 => /usr/lib/x86_64-linux-gnu/libavcodec.so.53 (0x00007f0479444000)
# 	libavformat.so.53 => /usr/lib/x86_64-linux-gnu/libavformat.so.53 (0x00007f0479143000)
# 	libavutil.so.51 => /usr/lib/x86_64-linux-gnu/libavutil.so.51 (0x00007f0478f22000)
# 	libswscale.so.2 => /usr/lib/x86_64-linux-gnu/libswscale.so.2 (0x00007f0478cdc000)
# 	libopencv_ocl.so.2.4 => /usr/local/lib/libopencv_ocl.so.2.4 (0x00007f04788f1000)
# 	/lib64/ld-linux-x86-64.so.2 (0x00007f0480900000)
# 	libpangocairo-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpangocairo-1.0.so.0 (0x00007f04786e5000)
# 	libX11.so.6 => /usr/lib/x86_64-linux-gnu/libX11.so.6 (0x00007f04783af000)
# 	libXfixes.so.3 => /usr/lib/x86_64-linux-gnu/libXfixes.so.3 (0x00007f04781a9000)
# 	libatk-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libatk-1.0.so.0 (0x00007f0477f86000)
# 	libcairo.so.2 => /usr/lib/x86_64-linux-gnu/libcairo.so.2 (0x00007f0477c7f000)
# 	libgdk_pixbuf-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgdk_pixbuf-2.0.so.0 (0x00007f0477a5f000)
# 	libgio-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 (0x00007f047770a000)
# 	libpangoft2-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpangoft2-1.0.so.0 (0x00007f04774df000)
# 	libpango-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpango-1.0.so.0 (0x00007f0477296000)
# 	libfontconfig.so.1 => /usr/lib/x86_64-linux-gnu/libfontconfig.so.1 (0x00007f0477060000)
# 	libXext.so.6 => /usr/lib/x86_64-linux-gnu/libXext.so.6 (0x00007f0476e4e000)
# 	libXrender.so.1 => /usr/lib/x86_64-linux-gnu/libXrender.so.1 (0x00007f0476c44000)
# 	libXinerama.so.1 => /usr/lib/x86_64-linux-gnu/libXinerama.so.1 (0x00007f0476a41000)
# 	libXi.so.6 => /usr/lib/x86_64-linux-gnu/libXi.so.6 (0x00007f0476830000)
# 	libXrandr.so.2 => /usr/lib/x86_64-linux-gnu/libXrandr.so.2 (0x00007f0476628000)
# 	libXcursor.so.1 => /usr/lib/x86_64-linux-gnu/libXcursor.so.1 (0x00007f047641e000)
# 	libXcomposite.so.1 => /usr/lib/x86_64-linux-gnu/libXcomposite.so.1 (0x00007f047621a000)
# 	libXdamage.so.1 => /usr/lib/x86_64-linux-gnu/libXdamage.so.1 (0x00007f0476017000)
# 	libffi.so.6 => /usr/lib/x86_64-linux-gnu/libffi.so.6 (0x00007f0475e0e000)
# 	libraw1394.so.11 => /usr/lib/x86_64-linux-gnu/libraw1394.so.11 (0x00007f0475bff000)
# 	libusb-1.0.so.0 => /lib/x86_64-linux-gnu/libusb-1.0.so.0 (0x00007f04759f0000)
# 	libv4l2.so.0 => /usr/lib/x86_64-linux-gnu/libv4l2.so.0 (0x00007f04757e4000)
# 	libvpx.so.1 => /usr/lib/libvpx.so.1 (0x00007f047553e000)
# 	libvorbisenc.so.2 => /usr/lib/x86_64-linux-gnu/libvorbisenc.so.2 (0x00007f047506f000)
# 	libvorbis.so.0 => /usr/lib/x86_64-linux-gnu/libvorbis.so.0 (0x00007f0474e43000)
# 	libtheoraenc.so.1 => /usr/lib/x86_64-linux-gnu/libtheoraenc.so.1 (0x00007f0474c05000)
# 	libtheoradec.so.1 => /usr/lib/x86_64-linux-gnu/libtheoradec.so.1 (0x00007f04749ea000)
# 	libspeex.so.1 => /usr/lib/x86_64-linux-gnu/libspeex.so.1 (0x00007f04747d1000)
# 	libschroedinger-1.0.so.0 => /usr/lib/libschroedinger-1.0.so.0 (0x00007f047451d000)
# 	libgsm.so.1 => /usr/lib/libgsm.so.1 (0x00007f047430f000)
# 	libva.so.1 => /usr/lib/x86_64-linux-gnu/libva.so.1 (0x00007f04740f9000)
# 	libbz2.so.1.0 => /lib/x86_64-linux-gnu/libbz2.so.1.0 (0x00007f0473ee8000)
# 	libfreetype.so.6 => /usr/lib/x86_64-linux-gnu/libfreetype.so.6 (0x00007f0473c4c000)
# 	libxcb.so.1 => /usr/lib/x86_64-linux-gnu/libxcb.so.1 (0x00007f0473a2d000)
# 	libpixman-1.so.0 => /usr/lib/x86_64-linux-gnu/libpixman-1.so.0 (0x00007f0473796000)
# 	libxcb-shm.so.0 => /usr/lib/x86_64-linux-gnu/libxcb-shm.so.0 (0x00007f0473593000)
# 	libxcb-render.so.0 => /usr/lib/x86_64-linux-gnu/libxcb-render.so.0 (0x00007f0473388000)
# 	libgmodule-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgmodule-2.0.so.0 (0x00007f0473184000)
# 	libselinux.so.1 => /lib/x86_64-linux-gnu/libselinux.so.1 (0x00007f0472f64000)
# 	libresolv.so.2 => /lib/x86_64-linux-gnu/libresolv.so.2 (0x00007f0472d48000)
# 	libexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007f0472b1d000)
# 	libv4lconvert.so.0 => /usr/lib/x86_64-linux-gnu/libv4lconvert.so.0 (0x00007f04728a8000)
# 	libogg.so.0 => /usr/lib/x86_64-linux-gnu/libogg.so.0 (0x00007f04726a1000)
# 	liborc-0.4.so.0 => /usr/lib/x86_64-linux-gnu/liborc-0.4.so.0 (0x00007f0472425000)
# 	libXau.so.6 => /usr/lib/x86_64-linux-gnu/libXau.so.6 (0x00007f0472222000)
# 	libXdmcp.so.6 => /usr/lib/x86_64-linux-gnu/libXdmcp.so.6 (0x00007f047201c000)

SPECS = {}
SPECS = {}
SPECS['Hyrule'] = {
    'cv2_version': r'2.4.8',
    'pillow_version': r'2.4.0',
    'pil_version': r'1.1.7',
}
#----
TEST_TARGETS['lena']['Hyrule'] = {
    'gpath': r'/home/joncrall/.config/utool/testdata/lena.jpg',
    'npshape': r'(220, 220, 3)',
    'npsum': r'18615625',
    'uuid1': r'd3daf98d-3035-65b7-2ff4-ca4076ab0cf1',
    'size': r'(220, 220)',
    'sum_': r'18615625',
    'img_bytes_': r'\xe2\x98U\xccib\xc8\x7fm\xbfXp\xb1cb ~~~TRUNCATED~~~ \xc7V\xb2',
    'hashbytes_16': r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\xab\x0c\xf1',
    'hashbytes_20': r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\x ~~~TRUNCATED~~~ 3\xd5\xcb',
    'bytes_sha1': r'<sha1 HASH object @ 0x2ba41c0>',
    'uuid_': r'd3daf98d-3035-65b7-2ff4-ca4076ab0cf1',
}
#----
TEST_TARGETS['jeff']['Hyrule'] = {
    'gpath': r'/home/joncrall/.config/utool/testdata/jeff.png',
    'npshape': r'(400, 400, 3)',
    'npsum': r'111817836',
    'uuid1': r'aed981a2-4116-9936-6311-e46bd17e25de',
    'size': r'(400, 400)',
    'sum_': r'152617836',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde',
    'hashbytes_20': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde<\xb3\xd6\xe7',
    'bytes_sha1': r'<sha1 HASH object @ 0x2ba4990>',
    'uuid_': r'aed981a2-4116-9936-6311-e46bd17e25de',
}
#----
TEST_TARGETS['easy1']['Hyrule'] = {
    'gpath': r'/home/joncrall/.config/utool/testdata/easy1.JPG',
    'npshape': r'(715, 1047, 3)',
    'npsum': r'354513878',
    'uuid1': r'4295b524-45df-8e25-52ca-71377109cebc',
    'size': r'(1047, 715)',
    'sum_': r'354513878',
    'img_bytes_': r'\xde\xe9\xfc\xdf\xea\xfe\xdf\xea\xfe ~~~TRUNCATED~~~ 9\x81\x85',
    'hashbytes_16': r'B\x95\xb5$E\xdf\x8e%R\xcaq7q\t\xce\xbc',
    'hashbytes_20': r'B\x95\xb5$E\xdf\x8e%R\xcaq7q\t\xce\xbcI\x11\x90\xf0',
    'bytes_sha1': r'<sha1 HASH object @ 0x2ba4850>',
    'uuid_': r'4295b524-45df-8e25-52ca-71377109cebc',
}
#----
TEST_TARGETS['http://i.imgur.com/QqSkNZe.png']['Hyrule'] = {
    'gpath': r'/home/joncrall/.config/utool/QqSkNZe.png',
    'npshape': r'(386, 564, 3)',
    'npsum': r'107691325',
    'uuid1': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
    'size': r'(564, 386)',
    'sum_': r'107691325',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e',
    'hashbytes_20': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e\xc5_H$',
    'bytes_sha1': r'<sha1 HASH object @ 0x2ba4c10>',
    'uuid_': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
}

# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imaging.so']['hendrik-vostro']
#   linux-vdso.so.1 =>  (0x00007fff5cdfe000)
#   libjpeg.so.9 => /usr/local/lib/libjpeg.so.9 (0x00007f1f568c0000)
#   libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f1f566a7000)
#   libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f1f56488000)
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f1f560c2000)
#   /lib64/ld-linux-x86-64.so.2 (0x00007f1f56d6c000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imagingft.so']['hendrik-vostro']
#   linux-vdso.so.1 =>  (0x00007fff08de3000)
#   libfreetype.so.6 => /usr/lib/x86_64-linux-gnu/libfreetype.so.6 (0x00007fbe7c200000)
#   libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fbe7bfe2000)
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fbe7bc1b000)
#   libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007fbe7ba02000)
#   libpng12.so.0 => /usr/lib/x86_64-linux-gnu/libpng12.so.0 (0x00007fbe7b7dc000)
#   /lib64/ld-linux-x86-64.so.2 (0x00007fbe7c6d1000)
#   libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fbe7b4d5000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imagingmath.so']['hendrik-vostro']
#   linux-vdso.so.1 =>  (0x00007fffc2d3b000)
#   libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f94a4665000)
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f94a429f000)
#   /lib64/ld-linux-x86-64.so.2 (0x00007f94a4ab1000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/cv2.so']['hendrik-vostro']
#   linux-vdso.so.1 =>  (0x00007fff373fe000)
#   libpython2.7.so.1.0 => /usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0 (0x00007f0fafb5b000)
#   libopencv_core.so.2.4 => /usr/local/lib/libopencv_core.so.2.4 (0x00007f0faf6ba000)
#   libopencv_flann.so.2.4 => /usr/local/lib/libopencv_flann.so.2.4 (0x00007f0faf445000)
#   libopencv_imgproc.so.2.4 => /usr/local/lib/libopencv_imgproc.so.2.4 (0x00007f0faef5c000)
#   libopencv_highgui.so.2.4 => /usr/local/lib/libopencv_highgui.so.2.4 (0x00007f0faeb23000)
#   libopencv_features2d.so.2.4 => /usr/local/lib/libopencv_features2d.so.2.4 (0x00007f0fae875000)
#   libopencv_calib3d.so.2.4 => /usr/local/lib/libopencv_calib3d.so.2.4 (0x00007f0fae5d9000)
#   libopencv_ml.so.2.4 => /usr/local/lib/libopencv_ml.so.2.4 (0x00007f0fae353000)
#   libopencv_video.so.2.4 => /usr/local/lib/libopencv_video.so.2.4 (0x00007f0fae0f9000)
#   libopencv_legacy.so.2.4 => /usr/local/lib/libopencv_legacy.so.2.4 (0x00007f0faddd7000)
#   libopencv_objdetect.so.2.4 => /usr/local/lib/libopencv_objdetect.so.2.4 (0x00007f0fadb54000)
#   libopencv_photo.so.2.4 => /usr/local/lib/libopencv_photo.so.2.4 (0x00007f0fad934000)
#   libopencv_nonfree.so.2.4 => /usr/local/lib/libopencv_nonfree.so.2.4 (0x00007f0fad6f8000)
#   libopencv_contrib.so.2.4 => /usr/local/lib/libopencv_contrib.so.2.4 (0x00007f0fad40d000)
#   libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f0fad108000)
#   libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f0face02000)
#   libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f0facbec000)
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f0fac825000)
#   libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f0fac607000)
#   libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f0fac3ee000)
#   libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f0fac1e9000)
#   libutil.so.1 => /lib/x86_64-linux-gnu/libutil.so.1 (0x00007f0fabfe6000)
#   librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f0fabdde000)
#   libpng12.so.0 => /usr/lib/x86_64-linux-gnu/libpng12.so.0 (0x00007f0fabbb7000)
#   libgtk-x11-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgtk-x11-2.0.so.0 (0x00007f0fab57b000)
#   libgdk-x11-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgdk-x11-2.0.so.0 (0x00007f0fab2c9000)
#   libgobject-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0 (0x00007f0fab077000)
#   libglib-2.0.so.0 => /lib/x86_64-linux-gnu/libglib-2.0.so.0 (0x00007f0faad6f000)
#   libavcodec.so.54 => /usr/lib/x86_64-linux-gnu/libavcodec.so.54 (0x00007f0faa01c000)
#   libavformat.so.54 => /usr/lib/x86_64-linux-gnu/libavformat.so.54 (0x00007f0fa9cf9000)
#   libavutil.so.52 => /usr/lib/x86_64-linux-gnu/libavutil.so.52 (0x00007f0fa9ad4000)
#   libswscale.so.2 => /usr/lib/x86_64-linux-gnu/libswscale.so.2 (0x00007f0fa988d000)
#   libopencv_ocl.so.2.4 => /usr/local/lib/libopencv_ocl.so.2.4 (0x00007f0fa94a9000)
#   /lib64/ld-linux-x86-64.so.2 (0x00007f0fb044c000)
#   libgmodule-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgmodule-2.0.so.0 (0x00007f0fa92a4000)
#   libpangocairo-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpangocairo-1.0.so.0 (0x00007f0fa9097000)
#   libX11.so.6 => /usr/lib/x86_64-linux-gnu/libX11.so.6 (0x00007f0fa8d62000)
#   libXfixes.so.3 => /usr/lib/x86_64-linux-gnu/libXfixes.so.3 (0x00007f0fa8b5b000)
#   libatk-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libatk-1.0.so.0 (0x00007f0fa8939000)
#   libcairo.so.2 => /usr/lib/x86_64-linux-gnu/libcairo.so.2 (0x00007f0fa862e000)
#   libgdk_pixbuf-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgdk_pixbuf-2.0.so.0 (0x00007f0fa840c000)
#   libgio-2.0.so.0 => /usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 (0x00007f0fa8099000)
#   libpangoft2-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpangoft2-1.0.so.0 (0x00007f0fa7e84000)
#   libpango-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libpango-1.0.so.0 (0x00007f0fa7c36000)
#   libfontconfig.so.1 => /usr/lib/x86_64-linux-gnu/libfontconfig.so.1 (0x00007f0fa79fa000)
#   libXrender.so.1 => /usr/lib/x86_64-linux-gnu/libXrender.so.1 (0x00007f0fa77ef000)
#   libXinerama.so.1 => /usr/lib/x86_64-linux-gnu/libXinerama.so.1 (0x00007f0fa75ec000)
#   libXi.so.6 => /usr/lib/x86_64-linux-gnu/libXi.so.6 (0x00007f0fa73dc000)
#   libXrandr.so.2 => /usr/lib/x86_64-linux-gnu/libXrandr.so.2 (0x00007f0fa71d2000)
#   libXcursor.so.1 => /usr/lib/x86_64-linux-gnu/libXcursor.so.1 (0x00007f0fa6fc7000)
#   libXcomposite.so.1 => /usr/lib/x86_64-linux-gnu/libXcomposite.so.1 (0x00007f0fa6dc4000)
#   libXdamage.so.1 => /usr/lib/x86_64-linux-gnu/libXdamage.so.1 (0x00007f0fa6bc1000)
#   libXext.so.6 => /usr/lib/x86_64-linux-gnu/libXext.so.6 (0x00007f0fa69ae000)
#   libffi.so.6 => /usr/lib/x86_64-linux-gnu/libffi.so.6 (0x00007f0fa67a6000)
#   libpcre.so.3 => /lib/x86_64-linux-gnu/libpcre.so.3 (0x00007f0fa6567000)
#   libxvidcore.so.4 => /usr/lib/x86_64-linux-gnu/libxvidcore.so.4 (0x00007f0fa6229000)
#   libx264.so.142 => /usr/lib/x86_64-linux-gnu/libx264.so.142 (0x00007f0fa5e93000)
#   libvpx.so.1 => /usr/lib/x86_64-linux-gnu/libvpx.so.1 (0x00007f0fa5ab3000)
#   libvorbisenc.so.2 => /usr/lib/x86_64-linux-gnu/libvorbisenc.so.2 (0x00007f0fa55e4000)
#   libvorbis.so.0 => /usr/lib/x86_64-linux-gnu/libvorbis.so.0 (0x00007f0fa53b7000)
#   libtheoraenc.so.1 => /usr/lib/x86_64-linux-gnu/libtheoraenc.so.1 (0x00007f0fa5176000)
#   libtheoradec.so.1 => /usr/lib/x86_64-linux-gnu/libtheoradec.so.1 (0x00007f0fa4f5d000)
#   libspeex.so.1 => /usr/lib/x86_64-linux-gnu/libspeex.so.1 (0x00007f0fa4d44000)
#   libschroedinger-1.0.so.0 => /usr/lib/x86_64-linux-gnu/libschroedinger-1.0.so.0 (0x00007f0fa4a7f000)
#   libopus.so.0 => /usr/lib/x86_64-linux-gnu/libopus.so.0 (0x00007f0fa4837000)
#   libopenjpeg.so.2 => /usr/lib/x86_64-linux-gnu/libopenjpeg.so.2 (0x00007f0fa4615000)
#   libmp3lame.so.0 => /usr/lib/x86_64-linux-gnu/libmp3lame.so.0 (0x00007f0fa4387000)
#   libgsm.so.1 => /usr/lib/x86_64-linux-gnu/libgsm.so.1 (0x00007f0fa4179000)
#   libva.so.1 => /usr/lib/x86_64-linux-gnu/libva.so.1 (0x00007f0fa3f63000)
#   librtmp.so.0 => /usr/lib/x86_64-linux-gnu/librtmp.so.0 (0x00007f0fa3d48000)
#   libgnutls.so.26 => /usr/lib/x86_64-linux-gnu/libgnutls.so.26 (0x00007f0fa3a8a000)
#   libbz2.so.1.0 => /lib/x86_64-linux-gnu/libbz2.so.1.0 (0x00007f0fa387a000)
#   libfreetype.so.6 => /usr/lib/x86_64-linux-gnu/libfreetype.so.6 (0x00007f0fa35d6000)
#   libxcb.so.1 => /usr/lib/x86_64-linux-gnu/libxcb.so.1 (0x00007f0fa33b7000)
#   libpixman-1.so.0 => /usr/lib/x86_64-linux-gnu/libpixman-1.so.0 (0x00007f0fa310d000)
#   libxcb-shm.so.0 => /usr/lib/x86_64-linux-gnu/libxcb-shm.so.0 (0x00007f0fa2f0a000)
#   libxcb-render.so.0 => /usr/lib/x86_64-linux-gnu/libxcb-render.so.0 (0x00007f0fa2d01000)
#   libselinux.so.1 => /lib/x86_64-linux-gnu/libselinux.so.1 (0x00007f0fa2add000)
#   libresolv.so.2 => /lib/x86_64-linux-gnu/libresolv.so.2 (0x00007f0fa28c2000)
#   libharfbuzz.so.0 => /usr/lib/x86_64-linux-gnu/libharfbuzz.so.0 (0x00007f0fa266d000)
#   libthai.so.0 => /usr/lib/x86_64-linux-gnu/libthai.so.0 (0x00007f0fa2463000)
#   libexpat.so.1 => /lib/x86_64-linux-gnu/libexpat.so.1 (0x00007f0fa2239000)
#   libogg.so.0 => /usr/lib/x86_64-linux-gnu/libogg.so.0 (0x00007f0fa202f000)
#   liborc-0.4.so.0 => /usr/lib/x86_64-linux-gnu/liborc-0.4.so.0 (0x00007f0fa1dad000)
#   libgcrypt.so.11 => /lib/x86_64-linux-gnu/libgcrypt.so.11 (0x00007f0fa1b2d000)
#   libtasn1.so.6 => /usr/lib/x86_64-linux-gnu/libtasn1.so.6 (0x00007f0fa1919000)
#   libp11-kit.so.0 => /usr/lib/x86_64-linux-gnu/libp11-kit.so.0 (0x00007f0fa16d7000)
#   libXau.so.6 => /usr/lib/x86_64-linux-gnu/libXau.so.6 (0x00007f0fa14d2000)
#   libXdmcp.so.6 => /usr/lib/x86_64-linux-gnu/libXdmcp.so.6 (0x00007f0fa12cc000)
#   libgraphite2.so.3 => /usr/lib/x86_64-linux-gnu/libgraphite2.so.3 (0x00007f0fa10af000)
#   libdatrie.so.1 => /usr/lib/x86_64-linux-gnu/libdatrie.so.1 (0x00007f0fa0ea8000)
#   libgpg-error.so.0 => /lib/x86_64-linux-gnu/libgpg-error.so.0 (0x00007f0fa0ca3000)

SPECS = {}
SPECS['hendrik-vostro'] = {
    'cv2_version': r'2.4.9',
    'pillow_version': r'2.4.0',
    'pil_version': r'1.1.7',
}
#----
TEST_TARGETS['lena']['hendrik-vostro'] = {
    'gpath': r'/home/hendrik/.config/utool/testdata/lena.jpg',
    'npshape': r'(220, 220, 3)',
    'npsum': r'18615625',
    'uuid1': r'f2c4658f-a722-793b-8578-c392fd550888',
    'size': r'(220, 220)',
    'sum_': r'18615820',
    'img_bytes_': r'\xe2\x99W\xccib\xc8\x7fm\xbfXn\xb1cb ~~~TRUNCATED~~~ \xc8V\xb1',
    'hashbytes_16': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88',
    'hashbytes_20': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88\xac.sk',
    'bytes_sha1': r'<sha1 HASH object @ 0x7fd5adf1e120>',
    'uuid_': r'f2c4658f-a722-793b-8578-c392fd550888',
}
#----
TEST_TARGETS['jeff']['hendrik-vostro'] = {
    'gpath': r'/home/hendrik/.config/utool/testdata/jeff.png',
    'npshape': r'(400, 400, 3)',
    'npsum': r'111817836',
    'uuid1': r'aed981a2-4116-9936-6311-e46bd17e25de',
    'size': r'(400, 400)',
    'sum_': r'152617836',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde',
    'hashbytes_20': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde<\xb3\xd6\xe7',
    'bytes_sha1': r'<sha1 HASH object @ 0x7fd5adf1e940>',
    'uuid_': r'aed981a2-4116-9936-6311-e46bd17e25de',
}
#----
TEST_TARGETS['easy1']['hendrik-vostro'] = {
    'gpath': r'/home/hendrik/.config/utool/testdata/easy1.JPG',
    'npshape': r'(715, 1047, 3)',
    'npsum': r'354513878',
    'uuid1': r'383dda50-f26b-200a-8baf-548e3ef88f9c',
    'size': r'(1047, 715)',
    'sum_': r'354510703',
    'img_bytes_': r'\xde\xe9\xfc\xdf\xea\xfe\xdf\xea\xfe ~~~TRUNCATED~~~ 9\x81\x85',
    'hashbytes_16': r'8=\xdaP\xf2k \n\x8b\xafT\x8e>\xf8\x8f\x9c',
    'hashbytes_20': r'8=\xdaP\xf2k \n\x8b\xafT\x8e>\xf8\x8f\x9c"\xdf\xcb\x08',
    'bytes_sha1': r'<sha1 HASH object @ 0x7fd5adf1ea30>',
    'uuid_': r'383dda50-f26b-200a-8baf-548e3ef88f9c',
}
#----
TEST_TARGETS['http://i.imgur.com/QqSkNZe.png']['hendrik-vostro'] = {
    'gpath': r'/home/hendrik/.config/utool/QqSkNZe.png',
    'npshape': r'(386, 564, 3)',
    'npsum': r'107691325',
    'uuid1': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
    'size': r'(564, 386)',
    'sum_': r'107691325',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e',
    'hashbytes_20': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e\xc5_H$',
    'bytes_sha1': r'<sha1 HASH object @ 0x7fd5adf1ed00>',
    'uuid_': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
}

# Listing libraries in '/usr/local/lib/python2.7/dist-packages/PIL'
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imaging.so']['debian2']
#       linux-gate.so.1 =>  (0xb77bd000)
#       libjpeg.so.8 => /usr/lib/i386-linux-gnu/libjpeg.so.8 (0xb7725000)
#       libz.so.1 => /lib/i386-linux-gnu/libz.so.1 (0xb770c000)
#       libpthread.so.0 => /lib/i386-linux-gnu/i686/cmov/libpthread.so.0 (0xb76f2000)
#       libc.so.6 => /lib/i386-linux-gnu/i686/cmov/libc.so.6 (0xb758e000)
#       /lib/ld-linux.so.2 (0xb77be000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imagingft.so']['debian2']
#       linux-gate.so.1 =>  (0xb77ba000)
#       libfreetype.so.6 => /usr/lib/i386-linux-gnu/libfreetype.so.6 (0xb7701000)
#       libpthread.so.0 => /lib/i386-linux-gnu/i686/cmov/libpthread.so.0 (0xb76e8000)
#       libc.so.6 => /lib/i386-linux-gnu/i686/cmov/libc.so.6 (0xb7583000)
#       libz.so.1 => /lib/i386-linux-gnu/libz.so.1 (0xb756a000)
#       /lib/ld-linux.so.2 (0xb77bb000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/PIL/_imagingmath.so']['debian2']
#       linux-gate.so.1 =>  (0xb7717000)
#       libpthread.so.0 => /lib/i386-linux-gnu/i686/cmov/libpthread.so.0 (0xb76e0000)
#       libc.so.6 => /lib/i386-linux-gnu/i686/cmov/libc.so.6 (0xb757c000)
#       /lib/ld-linux.so.2 (0xb7718000)
# ---------------
# DEPENDS['/usr/local/lib/python2.7/dist-packages/cv2.so']['debian2']
#       linux-gate.so.1 =>  (0xb773c000)
#       libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0xb72f9000)
#       libopencv_core.so.2.4 => /usr/local/lib/libopencv_core.so.2.4 (0xb70e0000)
#       libopencv_flann.so.2.4 => /usr/local/lib/libopencv_flann.so.2.4 (0xb706b000)
#       libopencv_imgproc.so.2.4 => /usr/local/lib/libopencv_imgproc.so.2.4 (0xb6de4000)
#       libopencv_highgui.so.2.4 => /usr/local/lib/libopencv_highgui.so.2.4 (0xb6c12000)
#       libopencv_features2d.so.2.4 => /usr/local/lib/libopencv_features2d.so.2.4 (0xb6b6a000)
#       libopencv_calib3d.so.2.4 => /usr/local/lib/libopencv_calib3d.so.2.4 (0xb6ad2000)
#       libopencv_ml.so.2.4 => /usr/local/lib/libopencv_ml.so.2.4 (0xb6a59000)
#       libopencv_video.so.2.4 => /usr/local/lib/libopencv_video.so.2.4 (0xb6a05000)
#       libopencv_legacy.so.2.4 => /usr/local/lib/libopencv_legacy.so.2.4 (0xb68f0000)
#       libopencv_objdetect.so.2.4 => /usr/local/lib/libopencv_objdetect.so.2.4 (0xb6870000)
#       libopencv_photo.so.2.4 => /usr/local/lib/libopencv_photo.so.2.4 (0xb6854000)
#       libopencv_gpu.so.2.4 => /usr/local/lib/libopencv_gpu.so.2.4 (0xb6812000)
#       libopencv_ocl.so.2.4 => /usr/local/lib/libopencv_ocl.so.2.4 (0xb6640000)
#       libopencv_nonfree.so.2.4 => /usr/local/lib/libopencv_nonfree.so.2.4 (0xb660f000)
#       libopencv_contrib.so.2.4 => /usr/local/lib/libopencv_contrib.so.2.4 (0xb6530000)
#       libstdc++.so.6 => /usr/lib/i386-linux-gnu/libstdc++.so.6 (0xb6444000)
#       libm.so.6 => /lib/i386-linux-gnu/i686/cmov/libm.so.6 (0xb641d000)
#       libgcc_s.so.1 => /lib/i386-linux-gnu/libgcc_s.so.1 (0xb6400000)
#       libpthread.so.0 => /lib/i386-linux-gnu/i686/cmov/libpthread.so.0 (0xb63e7000)
#       libc.so.6 => /lib/i386-linux-gnu/i686/cmov/libc.so.6 (0xb6283000)
#       libz.so.1 => /lib/i386-linux-gnu/libz.so.1 (0xb626a000)
#       libdl.so.2 => /lib/i386-linux-gnu/i686/cmov/libdl.so.2 (0xb6265000)
#       libutil.so.1 => /lib/i386-linux-gnu/i686/cmov/libutil.so.1 (0xb6261000)
#       librt.so.1 => /lib/i386-linux-gnu/i686/cmov/librt.so.1 (0xb6258000)
#       libjpeg.so.8 => /usr/lib/i386-linux-gnu/libjpeg.so.8 (0xb621f000)
#       libpng12.so.0 => /lib/i386-linux-gnu/libpng12.so.0 (0xb61f5000)
#       /lib/ld-linux.so.2 (0xb773d000)

SPECS = {}
SPECS['debian2'] = {
    'cv2_version': r'2.4.8',
    'pillow_version': r'2.4.0',
    'pil_version': r'1.1.7',
}
#----
TEST_TARGETS['lena']['debian2'] = {
    'gpath': r'/home/avi/.config/utool/testdata/lena.jpg',
    'npshape': r'(220, 220, 3)',
    'npsum': r'18615820',
    'uuid1': r'f2c4658f-a722-793b-8578-c392fd550888',
    'size': r'(220, 220)',
    'sum_': r'18615820',
    'img_bytes_': r'\xe2\x99W\xccib\xc8\x7fm\xbfXn\xb1cb ~~~TRUNCATED~~~ \xc8V\xb1',
    'hashbytes_16': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88',
    'hashbytes_20': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88\xac.sk',
    'bytes_sha1': r'<sha1 HASH object @ 0xa23e250>',
    'uuid_': r'f2c4658f-a722-793b-8578-c392fd550888',
}
#----
TEST_TARGETS['jeff']['debian2'] = {
    'gpath': r'/home/avi/.config/utool/testdata/jeff.png',
    'npshape': r'(400, 400, 3)',
    'npsum': r'111817836',
    'uuid1': r'aed981a2-4116-9936-6311-e46bd17e25de',
    'uuid1': r'aed981a2-4116-9936-6311-e46bd17e25de',
    'size': r'(400, 400)',
    'sum_': r'152617836',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde',
    'hashbytes_20': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde<\xb3\xd6\xe7',
    'bytes_sha1': r'<sha1 HASH object @ 0xa23e6b0>',
    'uuid_': r'aed981a2-4116-9936-6311-e46bd17e25de',
}
#----
TEST_TARGETS['easy1']['debian2'] = {
    'gpath': r'/home/avi/.config/utool/testdata/easy1.JPG',
    'npshape': r'(715, 1047, 3)',
    'npsum': r'354510703',
    'uuid1': r'383dda50-f26b-200a-8baf-548e3ef88f9c',
    'size': r'(1047, 715)',
    'sum_': r'354510703',
    'img_bytes_': r'\xde\xe9\xfc\xdf\xea\xfe\xdf\xea\xfe ~~~TRUNCATED~~~ 9\x81\x85',
    'hashbytes_16': r'8=\xdaP\xf2k \n\x8b\xafT\x8e>\xf8\x8f\x9c',
    'hashbytes_20': r'8=\xdaP\xf2k \n\x8b\xafT\x8e>\xf8\x8f\x9c"\xdf\xcb\x08',
    'bytes_sha1': r'<sha1 HASH object @ 0xa23e638>',
    'uuid_': r'383dda50-f26b-200a-8baf-548e3ef88f9c',
}
#----
TEST_TARGETS['http://i.imgur.com/QqSkNZe.png']['debian2'] = {
    'gpath': r'/home/avi/.config/utool/QqSkNZe.png',
    'npshape': r'(386, 564, 3)',
    'npsum': r'107691325',
    'uuid1': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
    'size': r'(564, 386)',
    'sum_': r'107691325',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e',
    'hashbytes_20': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e\xc5_H$',
    'bytes_sha1': r'<sha1 HASH object @ 0xa23ea20>',
    'uuid_': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
}


# Listing libraries in '/usr/lib/python2.7/site-packages/PIL'
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_imaging.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fff15dfe000)
# 	libjpeg.so.8 => /usr/lib/libjpeg.so.8 (0x00007f09916c6000)
# 	libz.so.1 => /usr/lib/libz.so.1 (0x00007f09914b0000)
# 	libtiff.so.5 => /usr/lib/libtiff.so.5 (0x00007f099123c000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007f0990e6e000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007f0990c50000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007f09908a2000)
# 	liblzma.so.5 => /usr/lib/liblzma.so.5 (0x00007f099067e000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007f099037a000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007f0990176000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007f098ff72000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007f0991ba9000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_imagingft.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fff675fe000)
# 	libfreetype.so.6 => /usr/lib/libfreetype.so.6 (0x00007f2a3195b000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007f2a3158e000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007f2a31370000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007f2a30fc1000)
# 	libz.so.1 => /usr/lib/libz.so.1 (0x00007f2a30dab000)
# 	libbz2.so.1.0 => /usr/lib/libbz2.so.1.0 (0x00007f2a30b9b000)
# 	libpng16.so.16 => /usr/lib/libpng16.so.16 (0x00007f2a30965000)
# 	libharfbuzz.so.0 => /usr/lib/libharfbuzz.so.0 (0x00007f2a3070f000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007f2a3050b000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007f2a30307000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007f2a30003000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007f2a31e4b000)
# 	libglib-2.0.so.0 => /usr/lib/libglib-2.0.so.0 (0x00007f2a2fcfb000)
# 	libgraphite2.so.3 => /usr/lib/libgraphite2.so.3 (0x00007f2a2fadc000)
# 	libpcre.so.1 => /usr/lib/libpcre.so.1 (0x00007f2a2f872000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_imagingmath.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fffdabfe000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007fde56cab000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007fde56a8d000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007fde566df000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007fde564da000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007fde562d7000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007fde55fd3000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007fde572bf000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_imagingtk.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fffb87fe000)
# 	libtcl8.6.so => /usr/lib/libtcl8.6.so (0x00007f9013113000)
# 	libtk8.6.so => /usr/lib/libtk8.6.so (0x00007f9012db9000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007f90129ec000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007f90127cd000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007f901241f000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007f901221b000)
# 	libz.so.1 => /usr/lib/libz.so.1 (0x00007f9012004000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007f9011d00000)
# 	libXft.so.2 => /usr/lib/libXft.so.2 (0x00007f9011aea000)
# 	libX11.so.6 => /usr/lib/libX11.so.6 (0x00007f90117a7000)
# 	libXss.so.1 => /usr/lib/libXss.so.1 (0x00007f90115a3000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007f90113a0000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007f90136f9000)
# 	libfontconfig.so.1 => /usr/lib/libfontconfig.so.1 (0x00007f9011162000)
# 	libfreetype.so.6 => /usr/lib/libfreetype.so.6 (0x00007f9010eb9000)
# 	libXrender.so.1 => /usr/lib/libXrender.so.1 (0x00007f9010caf000)
# 	libxcb.so.1 => /usr/lib/libxcb.so.1 (0x00007f9010a8e000)
# 	libXext.so.6 => /usr/lib/libXext.so.6 (0x00007f901087c000)
# 	libexpat.so.1 => /usr/lib/libexpat.so.1 (0x00007f9010652000)
# 	libbz2.so.1.0 => /usr/lib/libbz2.so.1.0 (0x00007f9010441000)
# 	libpng16.so.16 => /usr/lib/libpng16.so.16 (0x00007f901020c000)
# 	libharfbuzz.so.0 => /usr/lib/libharfbuzz.so.0 (0x00007f900ffb6000)
# 	libXau.so.6 => /usr/lib/libXau.so.6 (0x00007f900fdb1000)
# 	libXdmcp.so.6 => /usr/lib/libXdmcp.so.6 (0x00007f900fbab000)
# 	libglib-2.0.so.0 => /usr/lib/libglib-2.0.so.0 (0x00007f900f8a3000)
# 	libgraphite2.so.3 => /usr/lib/libgraphite2.so.3 (0x00007f900f684000)
# 	libpcre.so.1 => /usr/lib/libpcre.so.1 (0x00007f900f41a000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_webp.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fffd17fe000)
# 	libwebp.so.5 => /usr/lib/libwebp.so.5 (0x00007fb42978b000)
# 	libwebpmux.so.1 => /usr/lib/libwebpmux.so.1 (0x00007fb429583000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007fb4291b6000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007fb428f97000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007fb428be9000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007fb4288e5000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007fb4286e0000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007fb4284dd000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007fb429c2c000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/cv2.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fff6fffe000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007ff5e86d9000)
# 	libopencv_core.so.2.4 => /usr/lib/libopencv_core.so.2.4 (0x00007ff5e8284000)
# 	libopencv_flann.so.2.4 => /usr/lib/libopencv_flann.so.2.4 (0x00007ff5e8011000)
# 	libopencv_imgproc.so.2.4 => /usr/lib/libopencv_imgproc.so.2.4 (0x00007ff5e7b71000)
# 	libopencv_highgui.so.2.4 => /usr/lib/libopencv_highgui.so.2.4 (0x00007ff5e7919000)
# 	libopencv_features2d.so.2.4 => /usr/lib/libopencv_features2d.so.2.4 (0x00007ff5e7670000)
# 	libopencv_calib3d.so.2.4 => /usr/lib/libopencv_calib3d.so.2.4 (0x00007ff5e73d7000)
# 	libopencv_ml.so.2.4 => /usr/lib/libopencv_ml.so.2.4 (0x00007ff5e715b000)
# 	libopencv_video.so.2.4 => /usr/lib/libopencv_video.so.2.4 (0x00007ff5e6f05000)
# 	libopencv_legacy.so.2.4 => /usr/lib/libopencv_legacy.so.2.4 (0x00007ff5e6bef000)
# 	libopencv_objdetect.so.2.4 => /usr/lib/libopencv_objdetect.so.2.4 (0x00007ff5e6970000)
# 	libopencv_photo.so.2.4 => /usr/lib/libopencv_photo.so.2.4 (0x00007ff5e6751000)
# 	libopencv_nonfree.so.2.4 => /usr/lib/libopencv_nonfree.so.2.4 (0x00007ff5e6515000)
# 	libopencv_contrib.so.2.4 => /usr/lib/libopencv_contrib.so.2.4 (0x00007ff5e6235000)
# 	libstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007ff5e5f26000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007ff5e5c21000)
# 	libgcc_s.so.1 => /usr/lib/libgcc_s.so.1 (0x00007ff5e5a0b000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007ff5e565d000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007ff5e543e000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007ff5e523a000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007ff5e5037000)
# 	libz.so.1 => /usr/lib/libz.so.1 (0x00007ff5e4e20000)
# 	librt.so.1 => /usr/lib/librt.so.1 (0x00007ff5e4c18000)
# 	libGL.so.1 => /usr/lib/libGL.so.1 (0x00007ff5e49a8000)
# 	libtbb.so.2 => /usr/lib/libtbb.so.2 (0x00007ff5e4770000)
# 	libjpeg.so.8 => /usr/lib/libjpeg.so.8 (0x00007ff5e451b000)
# 	libpng16.so.16 => /usr/lib/libpng16.so.16 (0x00007ff5e42e6000)
# 	libtiff.so.5 => /usr/lib/libtiff.so.5 (0x00007ff5e4071000)
# 	libjasper.so.1 => /usr/lib/libjasper.so.1 (0x00007ff5e3e17000)
# 	libIlmImf-Imf_2_1.so.21 => /usr/lib/libIlmImf-Imf_2_1.so.21 (0x00007ff5e3b00000)
# 	libHalf.so.11 => /usr/lib/libHalf.so.11 (0x00007ff5e38bc000)
# 	libgtk-x11-2.0.so.0 => /usr/lib/libgtk-x11-2.0.so.0 (0x00007ff5e3279000)
# 	libgdk-x11-2.0.so.0 => /usr/lib/libgdk-x11-2.0.so.0 (0x00007ff5e2fc4000)
# 	libgobject-2.0.so.0 => /usr/lib/libgobject-2.0.so.0 (0x00007ff5e2d73000)
# 	libglib-2.0.so.0 => /usr/lib/libglib-2.0.so.0 (0x00007ff5e2a6b000)
# 	libgtkglext-x11-1.0.so.0 => /usr/lib/libgtkglext-x11-1.0.so.0 (0x00007ff5e2867000)
# 	libgdkglext-x11-1.0.so.0 => /usr/lib/libgdkglext-x11-1.0.so.0 (0x00007ff5e2602000)
# 	libgstreamer-0.10.so.0 => /usr/lib/libgstreamer-0.10.so.0 (0x00007ff5e2318000)
# 	libgstapp-0.10.so.0 => /usr/lib/libgstapp-0.10.so.0 (0x00007ff5e210b000)
# 	libgstvideo-0.10.so.0 => /usr/lib/libgstvideo-0.10.so.0 (0x00007ff5e1eed000)
# 	libdc1394.so.22 => /usr/lib/libdc1394.so.22 (0x00007ff5e1c78000)
# 	libxine.so.2 => /usr/lib/libxine.so.2 (0x00007ff5e1a0b000)
# 	libv4l1.so.0 => /usr/lib/libv4l1.so.0 (0x00007ff5e1804000)
# 	libavcodec.so.55 => /usr/lib/libavcodec.so.55 (0x00007ff5e072e000)
# 	libavformat.so.55 => /usr/lib/libavformat.so.55 (0x00007ff5e039c000)
# 	libavutil.so.52 => /usr/lib/libavutil.so.52 (0x00007ff5e0147000)
# 	libswscale.so.2 => /usr/lib/libswscale.so.2 (0x00007ff5dfed8000)
# 	libopencv_ocl.so.2.4 => /usr/lib/libopencv_ocl.so.2.4 (0x00007ff5dfaf2000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007ff5e8e49000)
# 	libglapi.so.0 => /usr/lib/libglapi.so.0 (0x00007ff5df8c9000)
# 	libXext.so.6 => /usr/lib/libXext.so.6 (0x00007ff5df6b6000)
# 	libXdamage.so.1 => /usr/lib/libXdamage.so.1 (0x00007ff5df4b3000)
# 	libXfixes.so.3 => /usr/lib/libXfixes.so.3 (0x00007ff5df2ad000)
# 	libX11-xcb.so.1 => /usr/lib/libX11-xcb.so.1 (0x00007ff5df0aa000)
# 	libX11.so.6 => /usr/lib/libX11.so.6 (0x00007ff5ded68000)
# 	libxcb-glx.so.0 => /usr/lib/libxcb-glx.so.0 (0x00007ff5deb50000)
# 	libxcb-dri2.so.0 => /usr/lib/libxcb-dri2.so.0 (0x00007ff5de94a000)
# 	libxcb-dri3.so.0 => /usr/lib/libxcb-dri3.so.0 (0x00007ff5de747000)
# 	libxcb-present.so.0 => /usr/lib/libxcb-present.so.0 (0x00007ff5de544000)
# 	libxcb-sync.so.1 => /usr/lib/libxcb-sync.so.1 (0x00007ff5de33d000)
# 	libxcb.so.1 => /usr/lib/libxcb.so.1 (0x00007ff5de11d000)
# 	libxshmfence.so.1 => /usr/lib/libxshmfence.so.1 (0x00007ff5ddf1a000)
# 	libXxf86vm.so.1 => /usr/lib/libXxf86vm.so.1 (0x00007ff5ddd13000)
# 	libdrm.so.2 => /usr/lib/libdrm.so.2 (0x00007ff5ddb07000)
# 	liblzma.so.5 => /usr/lib/liblzma.so.5 (0x00007ff5dd8e3000)
# 	libImath-2_1.so.11 => /usr/lib/libImath-2_1.so.11 (0x00007ff5dd6d1000)
# 	libIex-2_1.so.11 => /usr/lib/libIex-2_1.so.11 (0x00007ff5dd4b0000)
# 	libIlmThread-2_1.so.11 => /usr/lib/libIlmThread-2_1.so.11 (0x00007ff5dd2a8000)
# 	libgmodule-2.0.so.0 => /usr/lib/libgmodule-2.0.so.0 (0x00007ff5dd0a4000)
# 	libpangocairo-1.0.so.0 => /usr/lib/libpangocairo-1.0.so.0 (0x00007ff5dce97000)
# 	libatk-1.0.so.0 => /usr/lib/libatk-1.0.so.0 (0x00007ff5dcc70000)
# 	libcairo.so.2 => /usr/lib/libcairo.so.2 (0x00007ff5dc944000)
# 	libgdk_pixbuf-2.0.so.0 => /usr/lib/libgdk_pixbuf-2.0.so.0 (0x00007ff5dc720000)
# 	libgio-2.0.so.0 => /usr/lib/libgio-2.0.so.0 (0x00007ff5dc3ae000)
# 	libpangoft2-1.0.so.0 => /usr/lib/libpangoft2-1.0.so.0 (0x00007ff5dc198000)
# 	libpango-1.0.so.0 => /usr/lib/libpango-1.0.so.0 (0x00007ff5dbf4c000)
# 	libfontconfig.so.1 => /usr/lib/libfontconfig.so.1 (0x00007ff5dbd0e000)
# 	libXrender.so.1 => /usr/lib/libXrender.so.1 (0x00007ff5dbb04000)
# 	libXinerama.so.1 => /usr/lib/libXinerama.so.1 (0x00007ff5db901000)
# 	libXi.so.6 => /usr/lib/libXi.so.6 (0x00007ff5db6f0000)
# 	libXrandr.so.2 => /usr/lib/libXrandr.so.2 (0x00007ff5db4e6000)
# 	libXcursor.so.1 => /usr/lib/libXcursor.so.1 (0x00007ff5db2db000)
# 	libXcomposite.so.1 => /usr/lib/libXcomposite.so.1 (0x00007ff5db0d7000)
# 	libpcre.so.1 => /usr/lib/libpcre.so.1 (0x00007ff5dae6d000)
# 	libffi.so.6 => /usr/lib/libffi.so.6 (0x00007ff5dac65000)
# 	libGLU.so.1 => /usr/lib/libGLU.so.1 (0x00007ff5da9e5000)
# 	libXmu.so.6 => /usr/lib/libXmu.so.6 (0x00007ff5da7ca000)
# 	libXt.so.6 => /usr/lib/libXt.so.6 (0x00007ff5da563000)
# 	libpangox-1.0.so.0 => /usr/lib/libpangox-1.0.so.0 (0x00007ff5da342000)
# 	libfreetype.so.6 => /usr/lib/libfreetype.so.6 (0x00007ff5da099000)
# 	libgthread-2.0.so.0 => /usr/lib/libgthread-2.0.so.0 (0x00007ff5d9e96000)
# 	libxml2.so.2 => /usr/lib/libxml2.so.2 (0x00007ff5d9b2e000)
# 	libgstbase-0.10.so.0 => /usr/lib/libgstbase-0.10.so.0 (0x00007ff5d98d9000)
# 	liborc-0.4.so.0 => /usr/lib/liborc-0.4.so.0 (0x00007ff5d9652000)
# 	libraw1394.so.11 => /usr/lib/libraw1394.so.11 (0x00007ff5d9442000)
# 	libusb-1.0.so.0 => /usr/lib/libusb-1.0.so.0 (0x00007ff5d922a000)
# 	libresolv.so.2 => /usr/lib/libresolv.so.2 (0x00007ff5d9012000)
# 	libnsl.so.1 => /usr/lib/libnsl.so.1 (0x00007ff5d8dfa000)
# 	libv4l2.so.0 => /usr/lib/libv4l2.so.0 (0x00007ff5d8bec000)
# 	libva.so.1 => /usr/lib/libva.so.1 (0x00007ff5d89d5000)
# 	libxvidcore.so.4 => /usr/lib/libxvidcore.so.4 (0x00007ff5d86c1000)
# 	libx265.so.21 => /usr/lib/libx265.so.21 (0x00007ff5d824d000)
# 	libx264.so.142 => /usr/lib/libx264.so.142 (0x00007ff5d7ed9000)
# 	libvpx.so.1 => /usr/lib/libvpx.so.1 (0x00007ff5d7af5000)
# 	libvorbisenc.so.2 => /usr/lib/libvorbisenc.so.2 (0x00007ff5d784c000)
# 	libvorbis.so.0 => /usr/lib/libvorbis.so.0 (0x00007ff5d761e000)
# 	libtheoraenc.so.1 => /usr/lib/libtheoraenc.so.1 (0x00007ff5d73dd000)
# 	libtheoradec.so.1 => /usr/lib/libtheoradec.so.1 (0x00007ff5d71c4000)
# 	libspeex.so.1 => /usr/lib/libspeex.so.1 (0x00007ff5d6faa000)
# 	libschroedinger-1.0.so.0 => /usr/lib/libschroedinger-1.0.so.0 (0x00007ff5d6ce1000)
# 	libopus.so.0 => /usr/lib/libopus.so.0 (0x00007ff5d6a95000)
# 	libopenjpeg.so.1 => /usr/lib/libopenjpeg.so.1 (0x00007ff5d686b000)
# 	libopencore-amrwb.so.0 => /usr/lib/libopencore-amrwb.so.0 (0x00007ff5d6657000)
# 	libopencore-amrnb.so.0 => /usr/lib/libopencore-amrnb.so.0 (0x00007ff5d642d000)
# 	libmp3lame.so.0 => /usr/lib/libmp3lame.so.0 (0x00007ff5d61b5000)
# 	libgsm.so.1 => /usr/lib/libgsm.so.1 (0x00007ff5d5faa000)
# 	librtmp.so.1 => /usr/lib/librtmp.so.1 (0x00007ff5d5d8e000)
# 	libmodplug.so.1 => /usr/lib/libmodplug.so.1 (0x00007ff5d5a01000)
# 	libbluray.so.1 => /usr/lib/libbluray.so.1 (0x00007ff5d57cf000)
# 	libgnutls.so.28 => /usr/lib/libgnutls.so.28 (0x00007ff5d54b2000)
# 	libbz2.so.1.0 => /usr/lib/libbz2.so.1.0 (0x00007ff5d52a1000)
# 	libXau.so.6 => /usr/lib/libXau.so.6 (0x00007ff5d509d000)
# 	libXdmcp.so.6 => /usr/lib/libXdmcp.so.6 (0x00007ff5d4e96000)
# 	libharfbuzz.so.0 => /usr/lib/libharfbuzz.so.0 (0x00007ff5d4c40000)
# 	libpixman-1.so.0 => /usr/lib/libpixman-1.so.0 (0x00007ff5d4994000)
# 	libEGL.so.1 => /usr/lib/libEGL.so.1 (0x00007ff5d4770000)
# 	libxcb-shm.so.0 => /usr/lib/libxcb-shm.so.0 (0x00007ff5d456d000)
# 	libxcb-render.so.0 => /usr/lib/libxcb-render.so.0 (0x00007ff5d4362000)
# 	libexpat.so.1 => /usr/lib/libexpat.so.1 (0x00007ff5d4138000)
# 	libSM.so.6 => /usr/lib/libSM.so.6 (0x00007ff5d3f2f000)
# 	libICE.so.6 => /usr/lib/libICE.so.6 (0x00007ff5d3d12000)
# 	libudev.so.1 => /usr/lib/libudev.so.1 (0x00007ff5d3d01000)
# 	libv4lconvert.so.0 => /usr/lib/libv4lconvert.so.0 (0x00007ff5d3a88000)
# 	libogg.so.0 => /usr/lib/libogg.so.0 (0x00007ff5d3880000)
# 	libssl.so.1.0.0 => /usr/lib/libssl.so.1.0.0 (0x00007ff5d3612000)
# 	libcrypto.so.1.0.0 => /usr/lib/libcrypto.so.1.0.0 (0x00007ff5d3200000)
# 	libp11-kit.so.0 => /usr/lib/libp11-kit.so.0 (0x00007ff5d2fbe000)
# 	libtasn1.so.6 => /usr/lib/libtasn1.so.6 (0x00007ff5d2daa000)
# 	libnettle.so.4 => /usr/lib/libnettle.so.4 (0x00007ff5d2b7b000)
# 	libhogweed.so.2 => /usr/lib/libhogweed.so.2 (0x00007ff5d294c000)
# 	libgmp.so.10 => /usr/lib/libgmp.so.10 (0x00007ff5d26d5000)
# 	libgraphite2.so.3 => /usr/lib/libgraphite2.so.3 (0x00007ff5d24b6000)
# 	libxcb-xfixes.so.0 => /usr/lib/libxcb-xfixes.so.0 (0x00007ff5d22af000)
# 	libxcb-shape.so.0 => /usr/lib/libxcb-shape.so.0 (0x00007ff5d20ab000)
# 	libgbm.so.1 => /usr/lib/libgbm.so.1 (0x00007ff5d1ea2000)
# 	libwayland-client.so.0 => /usr/lib/libwayland-client.so.0 (0x00007ff5d1c94000)
# 	libwayland-server.so.0 => /usr/lib/libwayland-server.so.0 (0x00007ff5d1a82000)
# 	libuuid.so.1 => /usr/lib/libuuid.so.1 (0x00007ff5d187c000)
SPECS = {}
SPECS['ZackNet'] = {
    'cv2_version': r'2.4.9',
    'pillow_version': r'2.4.0',
    'pil_version': r'1.1.7',
}
#----
TEST_TARGETS['lena']['ZackNet'] = {
    'gpath': r'/home/zack/.config/utool/testdata/lena.jpg',
    'npshape': r'(220, 220, 3)',
    'npsum': r'18615625',
    'uuid1': r'd3daf98d-3035-65b7-2ff4-ca4076ab0cf1',
    'size': r'(220, 220)',
    'sum_': r'18615625',
    'img_bytes_': r'\xe2\x98U\xccib\xc8\x7fm\xbfXp\xb1cb ~~~TRUNCATED~~~ \xc7V\xb2',
    'hashbytes_16': r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\xab\x0c\xf1',
    'hashbytes_20': r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\x ~~~TRUNCATED~~~ 3\xd5\xcb',
    'bytes_sha1': r'<sha1 HASH object @ 0x7feb95400620>',
    'uuid_': r'd3daf98d-3035-65b7-2ff4-ca4076ab0cf1',
}
#----
TEST_TARGETS['jeff']['ZackNet'] = {
    'gpath': r'/home/zack/.config/utool/testdata/jeff.png',
    'npshape': r'(400, 400, 3)',
    'npsum': r'111817836',
    'uuid1': r'aed981a2-4116-9936-6311-e46bd17e25de',
    'size': r'(400, 400)',
    'sum_': r'152617836',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde',
    'hashbytes_20': r'\xae\xd9\x81\xa2A\x16\x996c\x11\xe4k\xd1~%\xde<\xb3\xd6\xe7',
    'bytes_sha1': r'<sha1 HASH object @ 0x7feb95400ee0>',
    'uuid_': r'aed981a2-4116-9936-6311-e46bd17e25de',
}
#----
TEST_TARGETS['easy1']['ZackNet'] = {
    'gpath': r'/home/zack/.config/utool/testdata/easy1.JPG',
    'npshape': r'(715, 1047, 3)',
    'npsum': r'354513878',
    'uuid1': r'4295b524-45df-8e25-52ca-71377109cebc',
    'size': r'(1047, 715)',
    'sum_': r'354513878',
    'img_bytes_': r'\xde\xe9\xfc\xdf\xea\xfe\xdf\xea\xfe ~~~TRUNCATED~~~ 9\x81\x85',
    'hashbytes_16': r'B\x95\xb5$E\xdf\x8e%R\xcaq7q\t\xce\xbc',
    'hashbytes_20': r'B\x95\xb5$E\xdf\x8e%R\xcaq7q\t\xce\xbcI\x11\x90\xf0',
    'bytes_sha1': r'<sha1 HASH object @ 0x7feb95400f30>',
    'uuid_': r'4295b524-45df-8e25-52ca-71377109cebc',
}
#----
TEST_TARGETS['http://i.imgur.com/QqSkNZe.png']['ZackNet'] = {
    'gpath': r'/home/zack/.config/utool/QqSkNZe.png',
    'npshape': r'(386, 564, 3)',
    'npsum': r'107691325',
    'uuid1': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
    'size': r'(564, 386)',
    'sum_': r'107691325',
    'img_bytes_': r'\xff\xff\xff\xff\xff\xff\xff\xff\xff ~~~TRUNCATED~~~ f\xff\xff',
    'hashbytes_16': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e',
    'hashbytes_20': r'\xa6;\xec\xe9\xbb\\\x13^!s\xee\x8e\x99\xa2T\x0e\xc5_H$',
    'bytes_sha1': r'<sha1 HASH object @ 0x7feb953bf030>',
    'uuid_': r'a63bece9-bb5c-135e-2173-ee8e99a2540e',
}


lines = []
writeln = lines.append


def print_var(key, indent='    '):
    locals_ = utool.get_parent_locals()
    val = utool.truncate_str(repr(str(locals_[key])), maxlen=64)
    line = indent + '%r: r%s,' % (key, val)
    writeln(line)


def get_gpath_from_key(key):
    if key.find('http') != -1:
        gpath = utool.grab_file_url(key)
    else:
        gpath = grabdata.get_test_gpaths(names=key)[0]
    return gpath


def test_uuid(key):
    print('\n\n-----------------------')
    print('[TEST_UUID] TESTING: %r' % key)
    gpath = get_gpath_from_key(key)
    npimg2 = cv2.imread(gpath)
    npshape = npimg2.shape
    npsum = npimg2.sum()

    pil_img = Image.open(gpath)
    uuid1 = preproc_image.get_image_uuid(pil_img)
    npimg = np.asarray(pil_img)
    img_bytes_ = npimg.ravel()[::64].tostring()
    bytes_sha1 = hashlib.sha1(img_bytes_)
    hashbytes_20 = bytes_sha1.digest()
    hashbytes_16 = hashbytes_20[0:16]
    uuid_ = uuid.UUID(bytes=hashbytes_16)

    sum_ = npimg.sum()
    size = pil_img.size

    test_varnames = [
        'gpath',
        'npshape',
        'npsum',
        'uuid1',
        'size',
        'sum_',
        'img_bytes_',
        'hashbytes_16',
        'hashbytes_20',
        'bytes_sha1',
        'uuid_',
    ]
    writeln('#----')
    writeln('TEST_TARGETS[%r][%r] = {' % (key, MACHINE_NAME))
    for varname in test_varnames:
        print_var(varname)
    writeln('}')

    try:
        target_uuid = TEST_TARGETS[key][MACHINE_NAME]
        #print_var('target_uuid')
        assert uuid1 == target_uuid, 'uuid and target_uuid do not match'
        assert uuid_ == uuid1, 'uuid_ does not match uuid1'
    except Exception as ex:
        utool.printex(ex)
        return False
    return True


def test_specs():
    try:
        print('Image.PILLOW_VERSION: %r' % Image.PILLOW_VERSION)
        assert Image.PILLOW_VERSION == '2.4.0'
    except Exception as ex:
        utool.printex(ex)
        pass
    if len(PIL.__path__) > 1:
        writeln('# WARNING THERE ARE MULTIPLE PILS! %r ' % PIL.__path__)
    for pil_path in PIL.__path__:
        writeln('# Listing libraries in %r' % (pil_path))
        lib_list = utool.ls_libs(pil_path)
        for libpath in lib_list:
            depend_out = utool.get_dynlib_dependencies(libpath)
            writeln('# ---------------')
            writeln('# DEPENDS[%r][%r]' % (libpath, MACHINE_NAME))
            writeln(utool.indentjoin(depend_out.splitlines(), '\n# ').strip())
    cv2_file = cv2.__file__
    cv2_depends = utool.get_dynlib_dependencies(cv2_file)
    cv2_version = cv2.__version__
    writeln('# ---------------')
    writeln('# DEPENDS[%r][%r]' % (cv2_file, MACHINE_NAME))
    writeln(utool.indentjoin(cv2_depends.splitlines(), '\n# ').strip())
    pillow_version = Image.PILLOW_VERSION
    pil_version = PIL.VERSION
    writeln('SPECS = {}')
    writeln('SPECS[%r] = {' % (MACHINE_NAME))
    print_var('cv2_version')
    print_var('cv2_file')
    print_var('pillow_version')
    print_var('pil_version')
    writeln('}')


def TEST_UUID():
    print('This test is disabled. UUIDS cannot be generated from jpeg pixels, even if there is no write')
    return locals()
    #print('Machine Name: %r' % MACHINE_NAME)
    #test_specs()
    #right = 'http://i.imgur.com/QqSkNZe.png'
    #test_image_keys = ['lena', 'jeff', 'easy1', right]
    #test_image_passed = {}

    #for key in test_image_keys:
    #    flag = test_uuid(key)
    #    test_image_passed[key] = flag

    ## Write out all that buffered info
    #sys.stdout.write('\n'.join(lines) + '\n')
    #sys.stdout.flush()

    #assert all(test_image_passed.values()), 'this test is hard'
    #return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_UUID)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
