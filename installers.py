#!/usr/bin/env python
from os.path import dirname, realpath, join, exists, normpath
import utool
import sys


def get_setup_dpath():
    assert exists('setup.py'), 'must be run in ibeis directory'
    assert exists('../ibeis/ibeis'), 'must be run in ibeis directory'
    cwd = normpath(realpath(dirname(__file__)))
    return cwd


def clean_pyinstaller():
    cwd = get_setup_dpath()
    utool.remove_files_in_dir(cwd, 'IBEISApp.pkg', recursive=False)
    utool.remove_files_in_dir(cwd, 'qt_menu.nib', recursive=False)
    utool.remove_files_in_dir(cwd, 'qt_menu.nib', recursive=False)
    utool.delete(join(cwd, 'dist'))
    utool.delete(join(cwd, 'build'))
    utool.delete(join(cwd, 'pyrf'))
    utool.delete(join(cwd, 'pyhesaff'))


def build_pyinstaller():
    clean_pyinstaller()
    # Run the pyinstaller command (does all the work)
    utool.cmd('pyinstaller', '_installers/pyinstaller-ibeis.spec')
    # Perform some post processing steps on the mac
    if sys.platform == 'darwin' and exists("dist/IBEIS.app/Contents/"):
        copy_list = [
            ('ibsicon.icns', 'Resources/icon-windowed.icns'),
            ('Info.plist', 'Info.plist'),
        ]
        srcdir = '_installers'
        dstdir = 'dist/IBEIS.app/Contents/'
        for srcname, dstname in copy_list:
            src = join(srcdir, srcname)
            dst = join(dstdir, dstname)
            utool.copy(src, dst)
        print("RUN: ./_installers/mac_dmg_builder.sh")
        # utool.cmd('./_scripts/mac_dmg_builder.sh')


if __name__ == '__main__':
    build_pyinstaller()
