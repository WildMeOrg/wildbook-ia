#!/usr/bin/env python
r"""
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
    pyinstaller --runtime-hook rthook_pyqt4.py _installers/pyinstaller-ibeis.spec -y
    "C:\Program Files (x86)\Inno Setup 5\ISCC.exe" _installers\win_installer_script.iss

    # Install
    dist\ibeis-win32-setup.exe

    # Test
    "C:\Program Files (x86)\IBEIS\IBEISApp.exe"

"""
from __future__ import absolute_import, division, print_function
from os.path import dirname, realpath, join, exists, normpath
import six
import utool as ut
import sys
import importlib
from os.path import join  # NOQA


def fix_pyinstaller_sip_api():
    """
    Hack to get the correct version of SIP

    References:
        http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    """
    import PyInstaller
    from os.path import dirname, join  # NOQA
    hook_fpath = join(dirname(PyInstaller.__file__), 'loader', 'rthooks', 'pyi_rth_qt4plugins.py')
    patch_code = ut.codeblock(
        '''
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
        ''')
    fpath = hook_fpath
    # Patch the hook file
    tag = 'SIP_API_2'
    ut.inject_python_code(fpath, patch_code, tag)
    #ut.editfile(hook_fpath)
    pass


def fix_command_tuple(command_tuple, sudo=False, shell=False, win32=ut.WIN32):
    r"""
    Args:
        command_tuple (?):
        sudo (bool):
        shell (bool):

    Returns:
        tuple: (None, None, None)

    CommandLine:
        python -m utool.util_cplat --test-fix_command_tuple:0
        python -m utool.util_cplat --test-fix_command_tuple:1

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> command_tuple = ('pyinstaller', '_installers/pyinstaller-ibeis.spec') #, '-y'
        >>> result = fix_command_tuple(command_tuple)
        >>> print(result)

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> command_tuple = 'pyinstaller --runtime-hook rthook_pyqt4.py _installers/pyinstaller-ibeis.spec -y'
        >>> result = fix_command_tuple(command_tuple)
        >>> print(result)
    """
    args = command_tuple
    print(type(args))
    print(args)
    if shell:
        # Popen only accepts strings is shell is True, which
        # it really shouldn't be.
        if  isinstance(args, (list, tuple)) and len(args) > 1:
            # Input is ['cmd', 'arg1', 'arg2']
            args = ' '.join(args)
        elif isinstance(args, (list, tuple)) and len(args) == 1:
            if isinstance(args[0], (tuple, list)):
                # input got nexted
                args = ' '.join(args)
            elif isinstance(args[0], six.string_types):
                # input is just nested string
                args = args[0]
        elif isinstance(args, six.string_types):
            pass
    if sudo is True:
        # On Windows it doesnt seem to matter if shlex splits the string or not
        # However on linux it seems like you need to split the string if you are
        # not using sudo, but if you use sudo you cannot split the string
        if not win32:
            if isinstance(args, six.string_types):
                import shlex
                args = shlex.split(args)
            args = ['sudo'] + args
            args = ' '.join(args)
        else:
            # TODO: strip out sudos
            pass
    return args


def system_command(command_tuple, detatch=False, sudo=False, shell=False, verbose=True):
    """
    Version 2 of util_cplat.cmd, hopefully it will work

    Args:
        command_tuple (?):
        detatch (bool):
        sudo (bool):
        shell (bool):
        verbose (bool):  verbosity flag, shows process output if True

    Returns:
        tuple: (None, None, None)

    CommandLine:
        python -m utool.util_cplat --test-system_command

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_cplat import *  # NOQA
        >>> # build test data
        >>> command_tuple = ('pyinstaller', '_installers/pyinstaller-ibeis.spec') #, '-y'
        >>> detatch = False
        >>> sudo = False
        >>> shell = False
        >>> verbose = True
        >>> # execute function
        >>> system_command(command_tuple, detatch, sudo, shell, verbose)
        >>> # verify results
        >>> result = str((None, None, None))
        >>> print(result)
    """
    sys.stdout.flush()
    try:
        # Parse the keyword arguments
        # Do fancy things with args
        # Print what you are about to do
        args = fix_command_tuple(command_tuple, sudo, shell)
        print('[ut.cmd] RUNNING: %r' % (args,))
        # Open a subprocess with a pipe
        import subprocess
        proc = subprocess.Popen(args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                shell=shell)
        if detatch:
            print('[ut.cmd] PROCESS DETATCHING')
            return None, None, 1
        if verbose and not detatch:
            print('[ut.cmd] RUNNING WITH VERBOSE OUTPUT')
            logged_out = []
            def _run_process(proc):
                while True:
                    # returns None while subprocess is running
                    retcode = proc.poll()
                    line = proc.stdout.readline()
                    yield line
                    if retcode is not None:
                        raise StopIteration('process finished')
            for line in _run_process(proc):
                line_ = line if six.PY2 else line.decode('utf-8')
                sys.stdout.write(line_)
                sys.stdout.flush()
                logged_out.append(line)
            out = '\n'.join(logged_out)
            (out_, err) = proc.communicate()
            #print('[ut.cmd] out: %s' % (out,))
            print('[ut.cmd] stdout: %s' % (out_,))
            print('[ut.cmd] stderr: %s' % (err,))
        else:
            # Surpress output
            #print('[ut.cmd] RUNNING WITH SUPRESSED OUTPUT')
            (out, err) = proc.communicate()
        # Make sure process if finished
        ret = proc.wait()
        print('[ut.cmd] PROCESS FINISHED')
        return out, err, ret
    except Exception as ex:
        import utool as ut
        if isinstance(args, tuple):
            print(ut.truepath(args[0]))
        elif isinstance(args, six.string_types):
            print(ut.unixpath(args))
        ut.printex(ex, 'Exception running ut.cmd',
                   keys=['verbose', 'detatch', 'shell', 'sudo'],
                   tb=True)


def get_setup_dpath():
    assert exists('setup.py'), 'must be run in ibeis directory'
    assert exists('main.py'), 'must be run in ibeis directory'
    assert exists('../ibeis/ibeis'), 'must be run in ibeis directory'
    cwd = normpath(realpath(dirname(__file__)))
    return cwd


def clean_pyinstaller():
    print('[installer] +--- CLEAN_PYINSTALLER ---')
    cwd = get_setup_dpath()
    ut.remove_files_in_dir(cwd, 'IBEISApp.pkg', recursive=False)
    ut.remove_files_in_dir(cwd, 'qt_menu.nib', recursive=False)
    ut.remove_files_in_dir(cwd, 'qt_menu.nib', recursive=False)
    ut.delete(join(cwd, 'dist/ibeis'))
    ut.delete(join(cwd, 'ibeis-win32-setup.exe'))
    ut.delete(join(cwd, 'build'))
    #ut.delete(join(cwd, 'pyrf'))
    #ut.delete(join(cwd, 'pyhesaff'))
    print('[installer] L___ FINSHED CLEAN_PYINSTALLER ___')


def build_pyinstaller():
    """
    build_pyinstaller creates build/ibeis/* and dist/ibeis/*
    """
    print('[installer] +--- BUILD_PYINSTALLER ---')
    # 1) RUN: PYINSTALLER
    # Run the pyinstaller command (does all the work)
    ut.cmd('pyinstaller --runtime-hook rthook_pyqt4.py _installers/pyinstaller-ibeis.spec -y')
    #else:
    #ut.cmd('pyinstaller', '_installers/pyinstaller-ibeis.spec', '-y')
    #ut.cmd('pyinstaller', '--runtime-hook rthook_pyqt4.py', '_installers/pyinstaller-ibeis.spec')
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
        print("RUN: sudo ./_installers/mac_dmg_builder.sh")
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
        assert ut.checkpath(inno_fpath, verbose=True, info=True), 'inno installer is still not installed!'
    return inno_fpath


def ensure_inno_script():
    """ writes inno script to distk """
    cwd = get_setup_dpath()
    iss_script_fpath = join(cwd, '_installers', 'win_installer_script.iss')
    # THE ISS USES {} AS SYNTAX. CAREFUL
    #app_publisher = 'Rensselaer Polytechnic Institute'
    #app_name = 'IBEIS'
    iss_script_code = ut.codeblock(
        '''
        ; Script generated by the Inno Setup Script Wizard.
        ; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!
        ; http://www.jrsoftware.org/isdl.php

        [Setup]
        ; NOTE: The value of AppId uniquely identifies this application.
        ; Do not use the same AppId value in installers for other applications.
        ; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
        AppId={{47BE3DA2-261D-4672-9849-18BB2EB382FC}
        AppName=IBEIS
        AppVersion=1
        ;AppVerName=IBEIS 1
        AppPublisher=Rensselaer Polytechnic Institute
        AppPublisherURL=www.rpi.edu/~crallj/
        AppSupportURL=www.rpi.edu/~crallj/
        AppUpdatesURL=www.rpi.edu/~crallj/
        DefaultDirName={pf}\IBEIS
        DefaultGroupName=IBEIS
        OutputBaseFilename=ibeis-win32-setup
        SetupIconFile=ibsicon.ico
        Compression=lzma
        SolidCompression=yes

        [Languages]
        Name: "english"; MessagesFile: "compiler:Default.isl"

        [Tasks]
        Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

        [Files]
        Source: "..\dist\ibeis\IBEISApp.exe"; DestDir: "{app}"; Flags: ignoreversion
        Source: "..\dist\ibeis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
        ; NOTE: Don't use "Flags: ignoreversion" on any shared system files

        [Icons]
        Name: "{group}\ibeis"; Filename: "{app}\IBEISApp.exe"
        Name: "{commondesktop}\ibeis"; Filename: "{app}\IBEISApp.exe"; Tasks: desktopicon

        [Run]
        Filename: "{app}\IBEISApp.exe"; Description: "{cm:LaunchProgram,IBEIS}"; Flags: nowait postinstall skipifsilent
        '''
    )
    ut.write_to(iss_script_fpath, iss_script_code, onlyifdiff=True)
    assert ut.checkpath(iss_script_fpath, verbose=True, info=True), 'cannot find iss_script_fpath'
    return iss_script_fpath


def inno_installer_postprocess():
    """ Move the built installer into a more reasonable directory """
    try:
        cwd = get_setup_dpath()
        installer_src = join(cwd, '_installers', 'Output', 'ibeis-win32-setup.exe')
        installer_dst = join(cwd, 'dist', 'ibeis-win32-setup.exe')
        # Make a timestamped version
        timestamped_fname = 'ibeis-win32-setup-{timestamp}.exe'.format(timestamp=ut.get_timestamp())
        installer_dst2 = join(cwd, 'dist', timestamped_fname)
        ut.move(installer_src, installer_dst)
        ut.copy(installer_dst, installer_dst2)
    except Exception as ex:
        ut.printex(ex, 'error moving setups', iswarning=True)


def build_win32_inno_installer():
    """ win32 self-executable package """
    print('[installer] +--- BUILD_WIN32_INNO_INSTALLER ---')
    assert ut.WIN32, 'Can only build INNO on windows'
    # Get inno executable
    inno_fpath = ensure_inno_isinstalled()
    # Get IBEIS inno script
    iss_script_fpath = ensure_inno_script()
    print('Trying to run ' + ' '.join(['"' + inno_fpath + '"', '"' + iss_script_fpath + '"']))
    try:
        command_args = ' '.join((inno_fpath, iss_script_fpath))
        ut.cmd(command_args)
    except Exception as ex:
        ut.printex(ex, 'error running script')
        raise
    # Move the installer into dist and make a timestamped version
    inno_installer_postprocess()
    # Uninstall exe in case we need to cleanup
    #uninstall_ibeis_exe = 'unins000.exe'
    print('[installer] L___ BUILD_WIN32_INNO_INSTALLER ___')


def package_installer():
    """
    system dependent post pyinstaller step
    """
    print('[installer] +--- PACKAGE_INSTALLER ---')
    #build_win32_inno_installer()
    if sys.platform.startswith('win32'):
        build_win32_inno_installer()
    elif sys.platform.startswith('darwin'):
        raise NotImplementedError('TODO: package into dmg')
        pass
    elif sys.platform.startswith('linux'):
        raise NotImplementedError('no linux packager (rpm or deb) supported. try running with --build')
        pass
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


def test_app():
    print('[installer] +--- TEST_APP ---')
    app_fpath = ut.unixpath('dist/ibeis/IBEISApp')
    app_ext = '.exe' if ut.WIN32 else ''
    ut.cmd(app_fpath + app_ext)
    print('[installer] L___ FINISH TEST_APP ___')
    #ut.cmd(ut.unixpath('dist/ibeis/ibeis-win32-setup.exe'))


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

    """
    print('For a full run use: python installers.py --all')
    print('[installer] +--- MAIN ---')
    BUILD_APP       = ut.get_argflag(('--build'))
    BUILD_INSTALLER = ut.get_argflag(('--inno', '--package', '--pkg'))
    TEST_APP        = ut.get_argflag(('--test'))
    CLEAN_BUILD     = ut.get_argflag(('--clean'))
    ALL             = ut.get_argflag('--all')

    fix_importlib_hook()
    # default behavior is full build
    BUILD_ALL = ALL or not (CLEAN_BUILD or BUILD_APP or BUILD_INSTALLER or TEST_APP)

    # 1) SETUP: CLEAN UP
    if CLEAN_BUILD or BUILD_ALL:
        clean_pyinstaller()
    if BUILD_APP or BUILD_ALL:
        build_pyinstaller()
    if BUILD_INSTALLER or BUILD_ALL:
        package_installer()
    if TEST_APP or BUILD_ALL:
        test_app()
    print('[installer] L___ FINISH MAIN ___')


if __name__ == '__main__':
    main()

'''
dist\ibeis-win32-setup.exe
dist\ibeis\IBEISApp.exe
'''
