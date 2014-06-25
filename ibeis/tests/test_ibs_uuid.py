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

# Listing libraries in '/usr/lib/python2.7/site-packages/PIL'
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_imaging.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fff6c5fe000)
# 	libjpeg.so.8 => /usr/lib/libjpeg.so.8 (0x00007faed593b000)
# 	libz.so.1 => /usr/lib/libz.so.1 (0x00007faed5725000)
# 	libtiff.so.5 => /usr/lib/libtiff.so.5 (0x00007faed54b1000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007faed50e3000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007faed4ec5000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007faed4b17000)
# 	liblzma.so.5 => /usr/lib/liblzma.so.5 (0x00007faed48f3000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007faed45ef000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007faed43eb000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007faed41e7000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007faed5e1e000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_imagingft.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fff13dfe000)
# 	libfreetype.so.6 => /usr/lib/libfreetype.so.6 (0x00007f7fbf133000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007f7fbed66000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007f7fbeb48000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007f7fbe799000)
# 	libz.so.1 => /usr/lib/libz.so.1 (0x00007f7fbe583000)
# 	libbz2.so.1.0 => /usr/lib/libbz2.so.1.0 (0x00007f7fbe373000)
# 	libpng16.so.16 => /usr/lib/libpng16.so.16 (0x00007f7fbe13d000)
# 	libharfbuzz.so.0 => /usr/lib/libharfbuzz.so.0 (0x00007f7fbdee7000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007f7fbdce3000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007f7fbdadf000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007f7fbd7db000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007f7fbf623000)
# 	libglib-2.0.so.0 => /usr/lib/libglib-2.0.so.0 (0x00007f7fbd4d3000)
# 	libgraphite2.so.3 => /usr/lib/libgraphite2.so.3 (0x00007f7fbd2b4000)
# 	libpcre.so.1 => /usr/lib/libpcre.so.1 (0x00007f7fbd04a000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_imagingmath.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fff5c7fe000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007f097f8d4000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007f097f6b6000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007f097f308000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007f097f103000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007f097ef00000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007f097ebfc000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007f097fee8000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_imagingtk.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fff367fe000)
# 	libtcl8.6.so => /usr/lib/libtcl8.6.so (0x00007ff030245000)
# 	libtk8.6.so => /usr/lib/libtk8.6.so (0x00007ff02feeb000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007ff02fb1e000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007ff02f8ff000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007ff02f551000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007ff02f34d000)
# 	libz.so.1 => /usr/lib/libz.so.1 (0x00007ff02f136000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007ff02ee32000)
# 	libXft.so.2 => /usr/lib/libXft.so.2 (0x00007ff02ec1c000)
# 	libX11.so.6 => /usr/lib/libX11.so.6 (0x00007ff02e8d9000)
# 	libXss.so.1 => /usr/lib/libXss.so.1 (0x00007ff02e6d5000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007ff02e4d2000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007ff03082b000)
# 	libfontconfig.so.1 => /usr/lib/libfontconfig.so.1 (0x00007ff02e294000)
# 	libfreetype.so.6 => /usr/lib/libfreetype.so.6 (0x00007ff02dfeb000)
# 	libXrender.so.1 => /usr/lib/libXrender.so.1 (0x00007ff02dde1000)
# 	libxcb.so.1 => /usr/lib/libxcb.so.1 (0x00007ff02dbc0000)
# 	libXext.so.6 => /usr/lib/libXext.so.6 (0x00007ff02d9ae000)
# 	libexpat.so.1 => /usr/lib/libexpat.so.1 (0x00007ff02d784000)
# 	libbz2.so.1.0 => /usr/lib/libbz2.so.1.0 (0x00007ff02d573000)
# 	libpng16.so.16 => /usr/lib/libpng16.so.16 (0x00007ff02d33e000)
# 	libharfbuzz.so.0 => /usr/lib/libharfbuzz.so.0 (0x00007ff02d0e8000)
# 	libXau.so.6 => /usr/lib/libXau.so.6 (0x00007ff02cee3000)
# 	libXdmcp.so.6 => /usr/lib/libXdmcp.so.6 (0x00007ff02ccdd000)
# 	libglib-2.0.so.0 => /usr/lib/libglib-2.0.so.0 (0x00007ff02c9d5000)
# 	libgraphite2.so.3 => /usr/lib/libgraphite2.so.3 (0x00007ff02c7b6000)
# 	libpcre.so.1 => /usr/lib/libpcre.so.1 (0x00007ff02c54c000)
# ---------------
# DEPENDS['/usr/lib/python2.7/site-packages/PIL/_webp.so']['ZackNet']
# 	linux-vdso.so.1 (0x00007fff9e7fe000)
# 	libwebp.so.5 => /usr/lib/libwebp.so.5 (0x00007f7bbdd75000)
# 	libwebpmux.so.1 => /usr/lib/libwebpmux.so.1 (0x00007f7bbdb6d000)
# 	libpython2.7.so.1.0 => /usr/lib/libpython2.7.so.1.0 (0x00007f7bbd7a0000)
# 	libpthread.so.0 => /usr/lib/libpthread.so.0 (0x00007f7bbd581000)
# 	libc.so.6 => /usr/lib/libc.so.6 (0x00007f7bbd1d3000)
# 	libm.so.6 => /usr/lib/libm.so.6 (0x00007f7bbcecf000)
# 	libdl.so.2 => /usr/lib/libdl.so.2 (0x00007f7bbccca000)
# 	libutil.so.1 => /usr/lib/libutil.so.1 (0x00007f7bbcac7000)
# 	/usr/lib64/ld-linux-x86-64.so.2 (0x00007f7bbe216000)
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
    'bytes_sha1': r'<sha1 HASH object @ 0x7fa96d215210>',
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
    'bytes_sha1': r'<sha1 HASH object @ 0x7fa96d215ad0>',
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
    'bytes_sha1': r'<sha1 HASH object @ 0x7fa96d215e90>',
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
    'bytes_sha1': r'<sha1 HASH object @ 0x7fa96d215d50>',
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


def test_pil_info():
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


def test_specs():
    cv2_version = cv2.__version__
    pillow_version = Image.PILLOW_VERSION
    pil_version = PIL.VERSION
    writeln('SPECS = {}')
    writeln('SPECS[%r] = {' % (MACHINE_NAME))
    print_var('cv2_version')
    print_var('pillow_version')
    print_var('pil_version')
    writeln('}')


def TEST_UUID():
    print('Machine Name: %r' % MACHINE_NAME)
    test_pil_info()
    test_specs()
    right = 'http://i.imgur.com/QqSkNZe.png'
    test_image_keys = ['lena', 'jeff', 'easy1', right]
    test_image_passed = {}

    for key in test_image_keys:
        flag = test_uuid(key)
        test_image_passed[key] = flag

    # Write out all that buffered info
    sys.stdout.write('\n'.join(lines) + '\n')
    sys.stdout.flush()

    assert all(test_image_passed.values()), 'this test is hard'
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_UUID)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
