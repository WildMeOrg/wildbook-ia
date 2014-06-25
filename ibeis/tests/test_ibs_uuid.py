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

#TEST_TARGETS[('hendrik-vostro', 'lena')] = {
#    'GPATH' : {
#        'hendrik-vostro': '/home/hendrik/.config/utool/testdata/lena.jpg',
#    },

#    'UUID': {
#        'hendrik-vostro': UUID('f2c4658f-a722-793b-8578-c392fd550888'),
#    },

#    'SUM' : {
#        'hyrule':         18615625,
#        'hendrik-vostro': 18615820,
#    },

#    'BYTES': {
#        'hyrule':         r'\xe2\x98U\xccib\xc8\x7fm\xbfXp\xb1cb\xccbl\xdfTb\xe1;[\xd1ba\xae\x81\xa3\xc1\x84n\xb5df\xcd`k\xd2]m\xe5H^\xd2a^\xd9\x80[\xce\x81Q\xc9e_\xcaNj\xfc]k\xd9N]\xcfb`\xe4}^`\x8f ~~~TRUNCATED~~~ u\x1cJ\xdcq\x86\x8e<<\xde\x1bf\x95^j\xdd\xc7V\xb2',
#        'hendrik-vostro': r'\xe2\x99W\xccib\xc8\x7fm\xbfXn\xb1cb\xccbl\xdfTb\xe1;[\xd1aa\xae\x81\xa1\xc1\x84n\xb5df\xcd_m\xd0]m\xe5H^\xd2a^\xd7\x80]\xd1\x81R\xc9e_\xcaNj\xfb]k\xd9N]\xcfb`\xe4}^`\x8f ~~~TRUNCATED~~~ u\x1cJ\xdaq\x86\x8e<<\xde\x1bf\x95^j\xdd\xc8V\xb1',
#    },

#    'HASH16' : {
#        'hyrule':         r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\xab\x0c\xf1',
#        'hendrik-vostro': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88',
#    },

#    'HASH20' : {
#        'hyrule':         r'\xd3\xda\xf9\x8d05e\xb7/\xf4\xca@v\xab\x0c\xf1\x99\x93\xd5\xcb',
#        'hendrik-vostro': r'\xf2\xc4e\x8f\xa7"y;\x85x\xc3\x92\xfdU\x08\x88\xac.sk',
#    },
#}

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
        print('WARNING THERE ARE MULTIPLE PILS! %r ' % PIL.__path__)
    pil_path = PIL.__path__[0]
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
