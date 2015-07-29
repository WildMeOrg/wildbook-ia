#def fix_command_tuple(command_tuple, sudo=False, shell=False, win32=ut.WIN32):
#    r"""
#    Args:
#        command_tuple (?):
#        sudo (bool):
#        shell (bool):
#
#    Returns:
#        tuple: (None, None, None)
#
#    CommandLine:
#        python -m utool.util_cplat --test-fix_command_tuple:0
#        python -m utool.util_cplat --test-fix_command_tuple:1
#
#    Example0:
#        >>> # DISABLE_DOCTEST
#        >>> from utool.util_cplat import *  # NOQA
#        >>> command_tuple = ('pyinstaller', '_installers/pyinstaller-ibeis.spec') #, '-y'
#        >>> result = fix_command_tuple(command_tuple)
#        >>> print(result)
#
#    Example1:
#        >>> # DISABLE_DOCTEST
#        >>> from utool.util_cplat import *  # NOQA
#        >>> command_tuple = 'pyinstaller --runtime-hook rthook_pyqt4.py _installers/pyinstaller-ibeis.spec -y'
#        >>> result = fix_command_tuple(command_tuple)
#        >>> print(result)
#    """
#    args = command_tuple
#    print(type(args))
#    print(args)
#    if shell:
#        # Popen only accepts strings is shell is True, which
#        # it really shouldn't be.
#        if  isinstance(args, (list, tuple)) and len(args) > 1:
#            # Input is ['cmd', 'arg1', 'arg2']
#            args = ' '.join(args)
#        elif isinstance(args, (list, tuple)) and len(args) == 1:
#            if isinstance(args[0], (tuple, list)):
#                # input got nexted
#                args = ' '.join(args)
#            elif isinstance(args[0], six.string_types):
#                # input is just nested string
#                args = args[0]
#        elif isinstance(args, six.string_types):
#            pass
#    if sudo is True:
#        # On Windows it doesnt seem to matter if shlex splits the string or not
#        # However on linux it seems like you need to split the string if you are
#        # not using sudo, but if you use sudo you cannot split the string
#        if not win32:
#            if isinstance(args, six.string_types):
#                import shlex
#                args = shlex.split(args)
#            args = ['sudo'] + args
#            args = ' '.join(args)
#        else:
#            # TODO: strip out sudos
#            pass
#    return args


#def system_command(command_tuple, detatch=False, sudo=False, shell=False, verbose=True):
#    """
#    Version 2 of util_cplat.cmd, hopefully it will work
#
#    Args:
#        command_tuple (?):
#        detatch (bool):
#        sudo (bool):
#        shell (bool):
#        verbose (bool):  verbosity flag, shows process output if True
#
#    Returns:
#        tuple: (None, None, None)
#
#    CommandLine:
#        python -m utool.util_cplat --test-system_command
#
#    Example:
#        >>> # DISABLE_DOCTEST
#        >>> from utool.util_cplat import *  # NOQA
#        >>> # build test data
#        >>> command_tuple = ('pyinstaller', '_installers/pyinstaller-ibeis.spec') #, '-y'
#        >>> detatch = False
#        >>> sudo = False
#        >>> shell = False
#        >>> verbose = True
#        >>> # execute function
#        >>> system_command(command_tuple, detatch, sudo, shell, verbose)
#        >>> # verify results
#        >>> result = str((None, None, None))
#        >>> print(result)
#    """
#    sys.stdout.flush()
#    try:
#        # Parse the keyword arguments
#        # Do fancy things with args
#        # Print what you are about to do
#        args = fix_command_tuple(command_tuple, sudo, shell)
#        print('[ut.cmd] RUNNING: %r' % (args,))
#        # Open a subprocess with a pipe
#        import subprocess
#        proc = subprocess.Popen(args,
#                                stdout=subprocess.PIPE,
#                                stderr=subprocess.STDOUT,
#                                shell=shell)
#        if detatch:
#            print('[ut.cmd] PROCESS DETATCHING')
#            return None, None, 1
#        if verbose and not detatch:
#            print('[ut.cmd] RUNNING WITH VERBOSE OUTPUT')
#            logged_out = []
#            def _run_process(proc):
#                while True:
#                    # returns None while subprocess is running
#                    retcode = proc.poll()
#                    line = proc.stdout.readline()
#                    yield line
#                    if retcode is not None:
#                        raise StopIteration('process finished')
#            for line in _run_process(proc):
#                line_ = line if six.PY2 else line.decode('utf-8')
#                sys.stdout.write(line_)
#                sys.stdout.flush()
#                logged_out.append(line)
#            out = '\n'.join(logged_out)
#            (out_, err) = proc.communicate()
#            #print('[ut.cmd] out: %s' % (out,))
#            print('[ut.cmd] stdout: %s' % (out_,))
#            print('[ut.cmd] stderr: %s' % (err,))
#        else:
#            # Surpress output
#            #print('[ut.cmd] RUNNING WITH SUPRESSED OUTPUT')
#            (out, err) = proc.communicate()
#        # Make sure process if finished
#        ret = proc.wait()
#        print('[ut.cmd] PROCESS FINISHED')
#        return out, err, ret
#    except Exception as ex:
#        import utool as ut
#        if isinstance(args, tuple):
#            print(ut.truepath(args[0]))
#        elif isinstance(args, six.string_types):
#            print(ut.unixpath(args))
#        ut.printex(ex, 'Exception running ut.cmd',
#                   keys=['verbose', 'detatch', 'shell', 'sudo'],
#                   tb=True)
