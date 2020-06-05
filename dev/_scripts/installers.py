# flake8: noqa
#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
On mac need to run with sudo


Testing:
    python %HOME%/code/wbia/_installers/wbia_pyinstaller_data_helper.py --test-get_data_list
    python ~/code/wbia/_installers/wbia_pyinstaller_data_helper.py --test-get_data_list



SeeAlso:
    _installers/wbia_pyinstaller_data_helper.py
    _installers/pyinstaller-wbia.spec

WindowsNew:
    python installers --build
    python installers --inno
    python installers --test

References:
    https://groups.google.com/forum/#!topic/pyinstaller/178I9ANuk14

This script is often flaky. here are workarounds

CommonIssues:
    Is the correct opencv being found?
    Is 3.0 being built? I think we are on 2.4.8

InstallPyinstaller:
    pip install pyinstaller
    pip install pyinstaller --upgrade

Win32CommandLine:
    # Uninstallation
    python installers.py --clean

    # Build Installer
    pyinstaller --runtime-hook rthook_pyqt4.py _installers/pyinstaller-wbia.spec -y
    "C:\Program Files (x86)\Inno Setup 5\ISCC.exe" _installers\win_installer_script.iss

    # Install
    dist\wbia-win32-setup.exe

    # Test
    "C:\Program Files (x86)\IBEIS\IBEISApp.exe"

"""
from __future__ import absolute_import, division, print_function
from os.path import dirname, realpath, join, exists, normpath

# import six
import utool as ut
import sys
import importlib
from os.path import join  # NOQA


def use_development_pyinstaller():
    """
    sudo pip uninstall pyinstaller
    pip uninstall pyinstaller
    code
    git clone https://github.com/pyinstaller/pyinstaller.git
    cd pyinstaller
    sudo python setup.py develop
    sudo python setup.py install
    ib
    which pyinstaller

    export PATH=$PATH:/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin

    had to uninstall sphinx
    sudo pip uninstall sphinx
    sudo pip uninstall sphinx
    """


def fix_pyinstaller_sip_api():
    """
    Hack to get the correct version of SIP for win32

    References:
        http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    """
    import PyInstaller
    from os.path import dirname, join  # NOQA

    hook_fpath = join(
        dirname(PyInstaller.__file__), 'loader', 'rthooks', 'pyi_rth_qt4plugins.py'
    )
    patch_code = ut.codeblock(
        """
        try:
            import sip
            # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
            sip.setapi('QVariant', 2)
            sip.setapi('QString', 2)
            sip.setapi('QTextStream', 2)
            sip.setapi('QTime', 2)
            sip.setapi('QUrl', 2)
            sip.setapi('QDate', 2)
            sip.setapi('QDateTime', 2)
            if hasattr(sip, 'setdestroyonexit'):
                sip.setdestroyonexit(False)  # This prevents a crash on windows
        except ValueError as ex:
            print('Warning: Value Error: %s' % str(ex))
        pass
        """
    )
    fpath = hook_fpath
    # Patch the hook file
    tag = 'SIP_API_2'
    ut.inject_python_code(fpath, patch_code, tag)
    # ut.editfile(hook_fpath)
    pass


def get_setup_dpath():
    assert exists('setup.py'), 'must be run in wbia directory'
    # assert exists('main.py'), 'must be run in wbia directory'
    assert exists('../wbia/wbia'), 'must be run in wbia directory'
    cwd = normpath(realpath(dirname(__file__)))
    return cwd


def clean_pyinstaller():
    print('[installer] +--- CLEAN_PYINSTALLER ---')
    cwd = get_setup_dpath()
    ut.remove_files_in_dir(cwd, 'IBEISApp.pkg', recursive=False)
    ut.remove_files_in_dir(cwd, 'qt_menu.nib', recursive=False)
    ut.remove_files_in_dir(cwd, 'qt_menu.nib', recursive=False)
    ut.delete(join(cwd, 'dist/wbia'))
    ut.delete(join(cwd, 'wbia-win32-setup.exe'))
    ut.delete(join(cwd, 'build'))
    # ut.delete(join(cwd, 'pyrf'))
    # ut.delete(join(cwd, 'pyhesaff'))
    print('[installer] L___ FINSHED CLEAN_PYINSTALLER ___')


def build_pyinstaller():
    """
    build_pyinstaller creates build/wbia/* and dist/wbia/*
    """
    print('[installer] +--- BUILD_PYINSTALLER ---')
    # 1) RUN: PYINSTALLER
    # Run the pyinstaller command (does all the work)
    utool_python_path = dirname(dirname(ut.__file__))
    # import os
    # os.environ['PYTHONPATH'] = os.pathsep.join([utool_python_path] + os.environ['PYTHONPATH'].strip(os.pathsep).split(os.pathsep))
    import os

    sys.path.insert(1, utool_python_path)
    if not ut.WIN32:
        pathcmd = 'export PYTHONPATH=%s%s$PYTHONPATH && ' % (
            utool_python_path,
            os.pathsep,
        )
    else:
        pathcmd = ''
    installcmd = '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/pyinstaller --runtime-hook _installers/rthook_pyqt4.py _installers/pyinstaller-wbia.spec -y'
    output, err, ret = ut.cmd(pathcmd + installcmd)
    if ret != 0:
        raise AssertionError('Pyinstalled failed with return code = %r' % (ret,))
    # ut.cmd(installcmd)
    # ut.cmd('pyinstaller --runtime-hook rthook_pyqt4.py _installers/pyinstaller-wbia.spec -y')
    # else:
    # ut.cmd('pyinstaller', '_installers/pyinstaller-wbia.spec', '-y')
    # ut.cmd('pyinstaller', '--runtime-hook rthook_pyqt4.py', '_installers/pyinstaller-wbia.spec')
    # 2) POST: PROCESSING
    # Perform some post processing steps on the mac

    if sys.platform == 'darwin' and exists('dist/IBEIS.app/Contents/'):
        copy_list = [
            ('ibsicon.icns', 'Resources/icon-windowed.icns'),
            ('Info.plist', 'Info.plist'),
        ]
        srcdir = '_installers'
        dstdir = 'dist/IBEIS.app/Contents/'
        for srcname, dstname in copy_list:
            src = join(srcdir, srcname)
            dst = join(dstdir, dstname)
            ut.copy(src, dst)
        # TODO: make this take arguments instead of defaulting to ~/code/wbia/build
        # print("RUN: sudo ./_installers/mac_dmg_builder.sh")
    app_fpath = get_dist_app_fpath()
    print('[installer] app_fpath = %s' % (app_fpath,))

    print('[installer] L___ FINISH BUILD_PYINSTALLER ___')
    # ut.cmd('./_scripts/mac_dmg_builder.sh')


def ensure_inno_isinstalled():
    """ Ensures that the current machine has INNO installed. returns path to the
    executable """
    assert ut.WIN32, 'Can only build INNO on windows'
    inno_fpath = ut.search_in_dirs('Inno Setup 5\ISCC.exe', ut.get_install_dirs())
    # Make sure INNO is installed
    if inno_fpath is None:
        print('WARNING: cannot find inno_fpath')
        AUTO_FIXIT = ut.WIN32
        print('Inno seems to not be installed. AUTO_FIXIT=%r' % AUTO_FIXIT)
        if AUTO_FIXIT:
            print('Automaticaly trying to downoad and install INNO')
            # Download INNO Installer
            inno_installer_url = 'http://www.jrsoftware.org/download.php/ispack.exe'
            inno_installer_fpath = ut.download_url(inno_installer_url)
            print('Automaticaly trying to install INNO')
            # Install INNO Installer
            ut.cmd(inno_installer_fpath)
        else:
            inno_homepage_url = 'http://www.jrsoftware.org/isdl.php'
            ut.open_url_in_browser(inno_homepage_url)
            raise AssertionError('Cannot find INNO and AUTOFIX it is false')
        # Ensure that it has now been installed
        inno_fpath = ut.search_in_dirs('Inno Setup 5\ISCC.exe', ut.get_install_dirs())
        assert ut.checkpath(
            inno_fpath, verbose=True, info=True
        ), 'inno installer is still not installed!'
    return inno_fpath


def ensure_inno_script():
    """ writes inno script to disk for win32 installer build """
    cwd = get_setup_dpath()
    iss_script_fpath = join(cwd, '_installers', 'win_installer_script.iss')
    # THE ISS USES {} AS SYNTAX. CAREFUL
    # app_publisher = 'Rensselaer Polytechnic Institute'
    # app_name = 'IBEIS'
    import wbia

    iss_script_code = ut.codeblock(
        r"""
        ; Script generated by the Inno Setup Script Wizard.
        ; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!
        ; http://www.jrsoftware.org/isdl.php

        [Setup]
        ; NOTE: The value of AppId uniquely identifies this application.
        ; Do not use the same AppId value in installers for other applications.
        ; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
        ; Also it seems like the off-balanced curly brace is necessary
        AppId={{47BE3DA2-261D-4672-9849-18BB2EB382FC}
        AppName=IBEIS
        AppVersion="""
        + str(wbia.__version__)
        + """
        ;AppVerName=IBEIS 1
        AppPublisher=Rensselaer Polytechnic Institute
        AppPublisherURL=wbia.org ;www.rpi.edu/~crallj/
        AppSupportURL=wbia.org ;ww.rpi.edu/~crallj/
        AppUpdatesURL=wbia.org ;www.rpi.edu/~crallj/
        DefaultDirName={pf}\IBEIS
        DefaultGroupName=IBEIS
        OutputBaseFilename=wbia-win32-setup
        SetupIconFile=ibsicon.ico
        Compression=lzma
        SolidCompression=yes

        [Languages]
        Name: "english"; MessagesFile: "compiler:Default.isl"

        [Tasks]
        Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

        [Files]
        Source: "..\dist\wbia\IBEISApp.exe"; DestDir: "{app}"; Flags: ignoreversion
        Source: "..\dist\wbia\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
        ; NOTE: Don't use "Flags: ignoreversion" on any shared system files

        [Icons]
        Name: "{group}\wbia"; Filename: "{app}\IBEISApp.exe"
        Name: "{commondesktop}\wbia"; Filename: "{app}\IBEISApp.exe"; Tasks: desktopicon

        [Run]
        Filename: "{app}\IBEISApp.exe"; Description: "{cm:LaunchProgram,IBEIS}"; Flags: nowait postinstall skipifsilent
        """
    )
    ut.write_to(iss_script_fpath, iss_script_code, onlyifdiff=True)
    assert ut.checkpath(
        iss_script_fpath, verbose=True, info=True
    ), 'cannot find iss_script_fpath'
    return iss_script_fpath


def build_win32_inno_installer():
    """ win32 self-executable package """
    print('[installer] +--- BUILD_WIN32_INNO_INSTALLER ---')
    assert ut.WIN32, 'Can only build INNO on windows'
    # Get inno executable
    inno_fpath = ensure_inno_isinstalled()
    # Get IBEIS inno script
    iss_script_fpath = ensure_inno_script()
    print(
        'Trying to run '
        + ' '.join(['"' + inno_fpath + '"', '"' + iss_script_fpath + '"'])
    )
    try:
        command_args = ' '.join((inno_fpath, iss_script_fpath))
        ut.cmd(command_args)
    except Exception as ex:
        ut.printex(ex, 'error running script')
        raise
    # Move the installer into dist and make a timestamped version
    # Uninstall exe in case we need to cleanup
    # uninstall_wbia_exe = 'unins000.exe'
    cwd = get_setup_dpath()
    installer_fpath = join(cwd, '_installers', 'Output', 'wbia-win32-setup.exe')
    print('[installer] L___ BUILD_WIN32_INNO_INSTALLER ___')
    return installer_fpath


def build_osx_dmg_installer():
    # outputs dmg to
    ut.cmd('./_installers/mac_dmg_builder.sh', sudo=True)
    cwd = get_setup_dpath()
    installer_fpath = join(cwd, 'dist', 'IBEIS.dmg')
    return installer_fpath


def build_linux_zip_binaries():
    fpath_list = ut.ls('dist/wbia')
    archive_fpath = 'dist/wbia-linux-binary.zip'
    ut.archive_files(archive_fpath, fpath_list)
    return archive_fpath


def package_installer():
    """
    system dependent post pyinstaller step
    """
    print('[installer] +--- PACKAGE_INSTALLER ---')
    # build_win32_inno_installer()
    cwd = get_setup_dpath()
    # Build the os-appropriate package
    if sys.platform.startswith('win32'):
        installer_src = build_win32_inno_installer()
        installer_fname_fmt = 'wbia-win32-install-{timestamp}.exe'
    elif sys.platform.startswith('darwin'):
        installer_src = build_osx_dmg_installer()
        installer_fname_fmt = 'wbia-osx-install-{timestamp}.dmg'
    elif sys.platform.startswith('linux'):
        installer_src = build_linux_zip_binaries()
        installer_fname_fmt = 'wbia-linux-binary-{timestamp}.zip'
        # try:
        #    raise NotImplementedError('no linux packager (rpm or deb) supported. try running with --build')
        # except Exception as ex:
        #    ut.printex(ex)
        # pass
    # timestamp the installer name
    installer_fname = installer_fname_fmt.format(timestamp=ut.get_timestamp())
    installer_dst = join(cwd, 'dist', installer_fname)
    try:
        ut.move(installer_src, installer_dst)
    except Exception as ex:
        ut.printex(ex, 'error moving setups', iswarning=True)
    print('[installer] L___ FINISH PACKAGE_INSTALLER ___')


def fix_importlib_hook():
    """ IMPORTLIB FIX

    References:
        http://stackoverflow.com/questions/18596410/importerror-no-module-named-mpl-toolkits-with-maptlotlib-1-3-0-and-py2exe
    """
    try:
        dpath_ = importlib.import_module('mpl_toolkits').__path__
        if isinstance(dpath_, (list, tuple)):
            for dpath in dpath_:
                fpath = join(dpath, '__init__.py')
                break
        else:
            dpath = dpath_
        if not ut.checkpath(dpath, verbose=True, info=True):
            ut.touch(fpath)

    except ImportError as ex:
        ut.printex(ex, 'pip install mpl_toolkits?')


def get_dist_app_fpath():
    app_fpath = ut.unixpath('dist/wbia/IBEISApp')
    if ut.DARWIN:
        app_fpath = ut.unixpath('dist/IBEIS.app/Contents/MacOS/IBEISApp')
    if ut.WIN32:
        app_fpath += '.exe'
    return app_fpath


def run_suite_test():
    app_fpath = get_dist_app_fpath()
    ut.assert_exists(app_fpath, 'app fpath must exist', info=True, verbose=True)
    ut.cmd(app_fpath + ' --run-utool-tests')
    # ut.cmd(app_fpath + ' --run-vtool_ibeis-tests')
    # ut.cmd(app_fpath + ' --run-wbia-tests')


def run_app_test():
    """
    Execute the installed app
    """
    print('[installer] +--- TEST_APP ---')
    app_fpath = get_dist_app_fpath()
    ut.assert_exists(app_fpath, 'app fpath must exist', info=True, verbose=True)
    if ut.DARWIN:
        # ut.cmd('open ' + ut.unixpath('dist/IBEIS.app'))
        """
        rm -rf ~/Desktop/IBEIS.app
        rm -rf /Applications/IBEIS.app
        ls /Applications/IBEIS.app
        cd /Volumes/IBEIS

        ib
        cd dist

        # Install to /Applications
        hdiutil attach ~/code/wbia/dist/IBEIS.dmg
        cp -R /Volumes/IBEIS/IBEIS.app /Applications/IBEIS.app
        hdiutil unmount /Volumes/IBEIS
        open -a /Applications/IBEIS.app

        chmod +x  /Applications/IBEIS.app/Contents/MacOS/IBEISApp

        cp -R /Volumes/IBEIS/IBEIS.app ~/Desktop
        open -a ~/Desktop/IBEIS.app
        chmod +x  ~/code/wbia/dist/IBEIS.app/Contents/MacOS/IBEISApp
        open -a ~/code/wbia/dist/IBEIS.app
        open ~/code/wbia/dist/IBEIS.app/Contents/MacOS/IBEISApp

        open ~/Desktop/IBEIS.app

        ./dist/IBEIS.app/Contents/MacOS/IBEISApp --run-tests
        """
        ut.cmd(app_fpath)
    else:
        ut.cmd(app_fpath)

    print('[installer] L___ FINISH TEST_APP ___')
    # ut.cmd(ut.unixpath('dist/wbia/wbia-win32-setup.exe'))


def main():
    """
    CommandLine:
        python installers.py --clean
        python installers.py --all
        python installers.py --inno
        # For linux
        python installers.py --clean
        python installers.py --build
        python installers.py --test
        python installers.py --clean --build --test
        python installers.py --build --test

    """
    print('For a full run use: python installers.py --all')
    print('[installer] +--- MAIN ---')
    import functools

    get_argflag = functools.partial(ut.get_argflag, need_prefix=False)
    BUILD_APP = get_argflag(('--build'))
    BUILD_INSTALLER = get_argflag(('--inno', '--package', '--pkg'))
    TEST_RUN = get_argflag(('--run'))
    TEST_CODE = get_argflag(('--test'))
    CLEAN_BUILD = get_argflag(('--clean'))
    ALL = get_argflag('--all')

    fix_importlib_hook()
    # default behavior is full build
    DEFAULT_RUN = len(sys.argv) == 1
    # or not (CLEAN_BUILD or BUILD_APP or BUILD_INSTALLER or TEST_APP)

    # 1) SETUP: CLEAN UP
    if CLEAN_BUILD or ALL:
        clean_pyinstaller()
    if BUILD_APP or ALL or DEFAULT_RUN:
        build_pyinstaller()
    if BUILD_INSTALLER or ALL:
        package_installer()
    if TEST_CODE or ALL:
        run_suite_test()
    if TEST_RUN or ALL:
        run_app_test()
    print('[installer] L___ FINISH MAIN ___')


if __name__ == '__main__':
    main()

"""
dist\wbia-win32-setup.exe
dist\wbia\IBEISApp.exe
"""
