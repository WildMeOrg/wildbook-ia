

def main():  # nocover
    import vtool_ibeis
    print('Looks like the imports worked')
    print('vtool_ibeis = {!r}'.format(vtool_ibeis))
    print('vtool_ibeis.__file__ = {!r}'.format(vtool_ibeis.__file__))
    print('vtool_ibeis.__version__ = {!r}'.format(vtool_ibeis.__version__))

    from vtool_ibeis._pyflann_backend import pyflann
    print('pyflann = {!r}'.format(pyflann))
    from vtool_ibeis import sver_c_wrapper
    print('sver_c_wrapper.lib_fname = {!r}'.format(sver_c_wrapper.lib_fname))
    print('sver_c_wrapper.lib_fname_cand = {!r}'.format(sver_c_wrapper.lib_fname_cand))


if __name__ == '__main__':
    """
    CommandLine:
       python -m vtool_ibeis
    """
    main()
