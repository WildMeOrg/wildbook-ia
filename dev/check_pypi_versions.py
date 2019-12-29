"""
Check which subpackages I've updated but not released on pypi yet.

pip install version-query
pip install yolk3k
"""
# import yolk
from distutils.version import LooseVersion
import ubelt as ub

modnames = [
    'utool',
    'vtool_ibeis',
    'guitool_ibeis',
    'plottool_ibeis',
    'pyhesaff',
    'pyflann_ibeis',
    'ibeis',
]
modname_to_info = {}
for modname in modnames:
    info = ub.cmd('yolk -V {}'.format(modname), verbose=3)
    modname_to_info[modname] = info

for modname in modnames:
    module = ub.import_module_from_name(modname)


for modname in modnames:
    module = ub.import_module_from_name(modname)
    pypi_version = LooseVersion(modname_to_info[modname]['out'].strip().split(' ')[1])
    local_version = LooseVersion(module.__version__)
    print('modname = {!r}'.format(modname))
    print('pypi_version = {!r}'.format(pypi_version))
    print('local_version = {!r}'.format(local_version))
    if local_version > pypi_version:
        print('--------')
        print("NEED TO PUBLISH {}".format(modname))
        print('https://travis-ci.org/Erotemic/{}'.format(modname))
        print('--------')
