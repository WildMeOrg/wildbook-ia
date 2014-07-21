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
    print('[installer] clean_pyinstaller()')
    cwd = get_setup_dpath()
    utool.remove_files_in_dir(cwd, 'IBEISApp.pkg', recursive=False)
    utool.remove_files_in_dir(cwd, 'qt_menu.nib', recursive=False)
    utool.remove_files_in_dir(cwd, 'qt_menu.nib', recursive=False)
    utool.delete(join(cwd, 'dist'))
    utool.delete(join(cwd, 'build'))
    utool.delete(join(cwd, 'pyrf'))
    utool.delete(join(cwd, 'pyhesaff'))
    print('[installer] finished clean_pyinstaller()')


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


def build_win32_inno_installer():
    inno_dir = r'C:\Program Files (x86)\Inno Setup 5'
    inno_fname = 'ISCC.exe'
    inno_fpath = join(inno_dir, inno_fname)
    cwd = get_setup_dpath()
    iss_script = join(cwd, '_installers', 'win_installer_script.iss')
    assert utool.checkpath(inno_fpath, verbose=True)
    assert utool.checkpath(iss_script, verbose=True)
    utool.cmd([inno_fpath, iss_script])
    import shutil
    installer_src = join(cwd, '_installers', 'Output', 'ibeis-win32-setup.exe')
    installer_dst = join(cwd, 'dist', 'ibeis-win32-setup.exe')
    shutil.move(installer_src, installer_dst)


def package_installer():
    if sys.platform.startswith('win32'):
        build_win32_inno_installer()
    elif sys.platform.startswith('darwin'):
        raise NotImplementedError('TODO: package into dmg')
        pass
    elif sys.platform.startswith('linux'):
        raise NotImplementedError('no linux packager (rpm or deb) supported')
        pass


if __name__ == '__main__':
    if 'all' in sys.argv:
        build_pyinstaller()
        package_installer()
    elif 'inno' in sys.argv:
        build_win32_inno_installer()
    else:
        build_pyinstaller()
