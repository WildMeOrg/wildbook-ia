# -*- coding: utf-8 -*-
"""
Check which subpackages I've updated but not released on pypi yet.

pip install version-query
pip install yolk3k
"""
# import yolk
from distutils.version import LooseVersion
import ubelt as ub


def pypi_info(package_name, version=None):
    """
    package_name = 'wbia'
    version = '2.0.0'
    version = '2.0.1'
    version = '2.0.2'
    version = '2.2.1'
    version = '2.2.2'
    version = '2.2.3'

    UNFINISHED

    """
    import requests

    if version is None:
        url = 'https://pypi.python.org/pypi/{}/json'.format(package_name)
    else:
        url = 'https://pypi.python.org/pypi/{}/{}/json'.format(package_name, version)

    package = requests.get(url).json()

    package_info = package['info']
    requirements = package_info['requires_dist']

    availble_versions = package['releases'].keys()

    max_ver = max(package['releases'].keys())
    latest_wheel_info_list = package['releases'][max_ver]


def query_module_pypi_info(modname, verbose=0):
    """
    Determine the lastest version of a module on pypi and the current installed
    version.
    """
    cmdinfo = ub.cmd('yolk -V {}'.format(modname), verbose=verbose, check=True)
    pypi_version = LooseVersion(cmdinfo['out'].strip().split(' ')[1])
    try:
        module = ub.import_module_from_name(modname)
    except ImportError:
        local_version = None
    else:
        local_version = LooseVersion(module.__version__)
    info = {
        'modname': modname,
        'pypi_version': pypi_version,
        'local_version': local_version,
    }
    return info


def main():
    modnames = [
        'utool',
        'vtool_ibeis',
        'guitool_ibeis',
        'plottool_ibeis',
        'pyhesaff',
        'pyflann_ibeis',
        'wbia',
    ]

    print('--- force module side effects --- ')

    for modname in modnames:
        ub.import_module_from_name(modname)

    print('--- begin module query ---')

    for modname in modnames:
        info = query_module_pypi_info(modname)
        print(ub.repr2(info))
        if info['local_version'] > info['pypi_version']:
            print('--------')
            print('NEED TO PUBLISH {}'.format(modname))
            print('https://travis-ci.org/Erotemic/{}'.format(modname))
            print('--------')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/wbia/dev/check_pypi_versions.py
    """
    main()
