import sys
# FIXME: setup for more than just win32

WIN32 = sys.platform.startswith('win32')


def get_install_cmd(modname):
    if WIN32:
        install_cmd = ('_scripts\win32bootstrap.py --run --dl ' + modname)
    else:
        install_cmd = 'sudo pip install ' + modname
    return install_cmd


# Order is important here
modlist = [
    'patsy',
    'pandas',
    'statsmodels',
    'simplejson',
]

for modname in modlist:
    try:
        level = 0
        module = __import__(modname, globals(), locals(), fromlist=[], level=level)
    except ImportError as ex:
        install_cmd = get_install_cmd(modname)
        print('Please Run follow instruction and then rerun fix_common_issues.py: ')
        print(install_cmd)
        import utool as ut
        ut.cmd(install_cmd, shell=True)
        sys.exit(0)
