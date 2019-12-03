

def main():  # nocover
    import vtool
    print('Looks like the imports worked')
    print('vtool = {!r}'.format(vtool))
    print('vtool.__file__ = {!r}'.format(vtool.__file__))
    print('vtool.__version__ = {!r}'.format(vtool.__version__))

    from vtool._pyflann_backend import pyflann
    print('pyflann = {!r}'.format(pyflann))
    from vtool import sver_c_wrapper
    print('sver_c_wrapper.lib_fname = {!r}'.format(sver_c_wrapper.lib_fname))
    print('sver_c_wrapper.lib_fname_cand = {!r}'.format(sver_c_wrapper.lib_fname_cand))


if __name__ == '__main__':
    """
    CommandLine:
       python -m vtool
    """
    main()
