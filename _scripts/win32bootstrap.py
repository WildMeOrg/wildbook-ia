r"""
Hacky file to download win packages
Please only download files as needed.

Args:
    --dl {pkgname:str} : package name to download
    --run : if true runs installer on win32

CommandLine:
    python _scripts\win32bootstrap.py --dl winapi --run
    python _scripts\win32bootstrap.py --dl pyperclip --run

"""
from __future__ import division, print_function
import parse
import sys
#import os
import utool as ut
#from six.moves import filterfalse
import urllib2

'http://downloads.sourceforge.net/project/opencvlibrary/opencv-win/2.4.10/opencv-2.4.10.exe'

opencv_alt_ext_href = 'https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0-beta/'

UNOFFICIAL_WEBURL = 'http://www.lfd.uci.edu/~gohlke/pythonlibs/'
OS_VERSION = 'win32'
# cpython 27
PY_VERSION = 'cp27'
#PY_VERSION = 'py2.7'
#PY_VERSION = 'py3.4'

# force redownload of hrefs
FORCE = ut.get_argflag('--force')

AMD64 = False

os_to_pkgmanager = {
    'win32': 'win32unoff',
    'darwin': 'ports',
    'debian_family': 'apt-get',
    'fedora_family': 'yum',
}

default_pkgmanager = os_to_pkgmanager[sys.platform]


# TODO: implement this
class CPlatPkg(object):
    def __init__(self,
                 default_name,
                 pkgmanager_map={},
                 alias_list=[],
                 platform_specific=False,
                 import_name=None,
                 alt_link=None,
                 ):
        self.default_name = default_name
        self.pkgmanager_map = {
            'default': None,
            'yum': None,
            'apt-get': None,
            'ports': None,
            'win32unoff': None,
            'pip': None,
        }
        self.alias_list = alias_list
        self.import_name = import_name
        if import_name is not None:
            self.alias_list.append(import_name)
        self.pkgmanager_map.update(pkgmanager_map)
        # True if only available on certain platforms
        self.platform_specific = platform_specific

    def is_alias(self, pkgname):
        return pkgname in self.alias_list or pkgname == self.default_name

    def get_platform_pkgmanager_name(self, pkgname, pkgmanager=default_pkgmanager):
        """ returns name for current platforms package manager """
        pkgmanager_name = self.pkgmanager_map[pkgmanager]
        if pkgmanager_name is None:
            pkgmanager_name = self.default_name
        return pkgmanager_name

    def get_import_name(self):
        if self.import_name is not None:
            return self.import_name
        else:
            return self.default_name

    def is_installed(self):
        import_name = self.get_import_name()
        try:
            globals_ = globals()
            locals_ = locals()
            exec('import ' + import_name, globals_, locals_)
        except ImportError:
            return False
        return True


cplat_alias_pkglist = [
    CPlatPkg(
        'pywin32',
        import_name='win32api',
        platform_specific=True,
    ),

    CPlatPkg(
        'pyperclip',
        pkgmanager_map={
            'pip': 'pyperclip'
        }
    ),

    CPlatPkg(
        'line-profiler',
        {'win32': 'line_profiler'},
        ['kernprof']),

    CPlatPkg(
        'numpy',
        #{'win32': 'numpy-MKL'}
        {'win32': 'numpy-1.9.2rc1+mkl-cp27-none-win32.whl'}
        #'numpy-1.9.2rc1+mkl-cp27-none-win32.whl'
    ),
    # alias_tup = (std_dict, alias_list)
    # std_dict = keys=packagemanager, vals=truename
    # alias_list = list of names
    #({'default': 'line_profiler', }, ['line-profiler'],),
]


def resolve_alias(pkgname):
    for cplat_pkg in cplat_alias_pkglist:
        if cplat_pkg.is_alias(pkgname):
            return cplat_pkg.get_platform_pkgmanager_name(pkgname)
    return pkgname


KNOWN_PKG_LIST = [
    'pip',
    'python-dateutil',
    'pyzmq',
    'setuptools',
    'Pygments',
    'Cython',
    'requests',
    #'colorama',
    'psutil',
    #'functools32',
    #'six',  # use pip for this
    'dateutil',
    'pyreadline',
    'pyparsing',
    #'sip',
    'PyQt4',
    'Pillow',
    #'numpy-MKL-1.9',  # 'numpy',
    'scipy',
    'ipython',
    'tornado',
    'matplotlib',
    'scikit-learn',

    'statsmodels',
    'pandas',  # statsmodel uses pandas :(
    'patsy',  # statsmodel uses patsy

    'simplejson',

    # 'flask',  #  cant do flask
]


def get_uninstalled_project_names():
    try:
        import pip
        pkg_set = set(KNOWN_PKG_LIST)
        pathmeta_list = pip.get_installed_distributions()
        installed_set = set([meta.project_name for meta in pathmeta_list])
        uninstalled_set = pkg_set.difference(installed_set)
        uninstalled_list = list(uninstalled_set)
    except Exception as ex:
        print(ex)
        uninstalled_list = KNOWN_PKG_LIST
    return uninstalled_list


def build_uninstall_script():
    #import utool as ut
    from os.path import join
    #import parse
    pydir = 'C:/Python27'
    uninstall_list = ut.glob(pydir, 'Remove*.exe')
    cmd_list = []
    for exefname in uninstall_list:
        parse_result = parse.parse('{pypath}Remove{pkgname}.exe', exefname)
        pkgname = parse_result['pkgname']
        logfname = pkgname + '-wininst.log'
        logfpath = join(pydir, logfname)
        exefpath = join(pydir, exefname)
        cmd = '"' + exefpath + '" -u "' + logfpath + '"'
        cmd_list.append(cmd)

    script_text = ('\n'.join(cmd_list))
    print(script_text)


def main():
    r"""
    python win32bootstrap.py --dl numpy --force
    python win32bootstrap.py --dl numpy-1.9.2rc1 --force
    python win32bootstrap.py --dl numpy-1.9.2rc1 --run
    python win32bootstrap.py --force
    python win32bootstrap.py --dryrun
    python win32bootstrap.py --dryrun --dl numpy scipy
    python win32bootstrap.py --dl numpy

    C:\Users\jon.crall\AppData\Roaming\utool\numpy-1.9.2rc1+mkl-cp27-none-win32.whl
    pip instal C:/Users/jon.crall/AppData/Roaming/utool/numpy-1.9.2rc1+mkl-cp27-none-win32.whl

    """
    # Packages that you are requesting
    pkg_list = []
    if ut.get_argflag('--all'):
        pkg_list = KNOWN_PKG_LIST
    else:
        print('specify --all to download all packages')
        print('or specify --dl pkgname to download that package')
    pkg_list.extend(ut.get_argval('--dl', list, []))
    force = ut.get_argflag('--force')
    dryrun = ut.get_argflag('--dryrun')
    pkg_exe_list = bootstrap_sysreq(pkg_list, force, dryrun)
    if ut.get_argflag('--run'):
        for pkg_exe in pkg_exe_list:
            if pkg_exe.endswith('.whl'):
                ut.cmd('pip install ' + pkg_exe)
                #ut.cmd(pkg_exe)


def bootstrap_sysreq(pkg_list='all', force=False, dryrun=False):
    """
    pkg_list = ['line_profiler']
    """
    # Still very hacky
    if pkg_list == 'all':
        pkg_list = get_uninstalled_project_names()

    pkg_list_ = [resolve_alias(pkg) for pkg in  pkg_list]

    py_version = PY_VERSION
    #python34_win32_x64_url = 'https://www.python.org/ftp/python/3.4.1/python-3.4.1.amd64.msi'
    #python34_win32_x86_exe = ut.grab_file_url(python34_win32_x64_url)
    all_href_list, page_str = get_unofficial_package_hrefs(force=force)
    if len(all_href_list) > 0:
        print('all_href_list[0] = ' + str(all_href_list[0]))
    href_list = get_win_packages_href(all_href_list, py_version, pkg_list_)
    print('Available hrefs are:\n' +  '\n'.join(href_list))
    if not dryrun:
        pkg_exe_list = download_win_packages(href_list)
        text = '\n'.join(href_list) + '\n'
        text += ('Please Run:') + '\n'
        text += ('\n'.join(['pip install ' + pkg for pkg in pkg_exe_list]))
        #print('TODO: Figure out how to run these installers without the GUI: ans use the new wheels')
        print(text)
        print(pkg_list_)
    else:
        print('dryrun=True')
        print('href_list = %r' % (href_list,))
        pkg_exe_list = []
    return pkg_exe_list


def download_win_packages(href_list):
    pkg_exe_list = []
    #href = href_list[0]
    #pkg_exe = ut.util_grabdata.grab_file_url(href, delay=3, spoof=True)
    #pkg_exe_list += [pkg_exe]
    ## Execute download
    for href in href_list:
        # Download the file if you havent already done so
        pkg_exe = ut.util_grabdata.grab_file_url(href, delay=3, spoof=True)
        # Check to make sure it worked
        nBytes = ut.get_file_nBytes(pkg_exe)
        if nBytes < 1000:
            print('There may be a problem with %r' % (pkg_exe,))
            RETRY_PROBLEMS = False
            if RETRY_PROBLEMS:
                # retry if file was probably corrupted
                ut.delete(pkg_exe)
                pkg_exe = ut.util_grabdata.grab_file_url(href, delay=3, spoof=True)
        pkg_exe_list += [pkg_exe]
    return pkg_exe_list


def get_win_packages_href(all_href_list, py_version, pkg_list):
    """ Returns the urls to download the requested installers """
    href_list1, missing  = filter_href_list(all_href_list, pkg_list, OS_VERSION, py_version)
    href_list2, missing2 = filter_href_list(all_href_list, missing, OS_VERSION, py_version)
    href_list3, missing3 = filter_href_list(all_href_list, missing2, 'x64', py_version.replace('p', 'P'))
    href_list = href_list1 + href_list2 + href_list3
    return href_list


def filter_href_list(all_href_list, win_pkg_list, os_version, py_version):
    """
    Ignore:
        win_pkg_list = ['pywin32']
        OS_VERSION = 'win32'
        PY_VERSION = 'py2.7'
        os_version = OS_VERSION
        py_version = PY_VERSION
    """
    candidate_list = []
    # hack
    ignore_list = [
        'vigranumpy',
    ]
    for pkgname in win_pkg_list:
        amdfunc = lambda x: x.find('amd64') > -1 if AMD64 else lambda x: x.find('amd64') == -1
        from os.path import basename
        filter_funcs = [
            lambda x: x.find(pkgname) > -1,
            amdfunc,
            lambda x: py_version in x,
            lambda x: os_version in x,
            lambda x: not any([bad in x for bad in ignore_list]),
            lambda x: basename(x).lower().startswith(pkgname.lower()),
        ]
        _candidates = all_href_list
        for func_ in filter_funcs:
            _candidates = list(filter(func_, _candidates))
        candidates = list(_candidates)
        if len(candidates) > 1:
            #print('\n\n\n')
            #print(pkgname)
            # parse out version
            def get_href_version(href):
                y = basename(href).split('-' + py_version)[0]
                version = y.lower().split(pkgname.lower() + '-')[1]
                return version
            version_strs = list(map(get_href_version, candidates))
            # Argsort the versions
            from distutils.version import LooseVersion
            from operator import itemgetter
            versions = list(map(LooseVersion, version_strs))
            sorted_tups = sorted(list(enumerate(versions)), key=itemgetter(1), reverse=True)
            # Choose highest version
            index = sorted_tups[0][0]
            candidates = candidates[index:index + 1]
            #print(candidates)
            #print('Conflicting candidates: %r' % (candidates))
            #print('\n\n\n')
        candidate_list.extend(candidates)

    missing = []
    for pkgname in win_pkg_list:
        if not any([pkgname in href for href in candidate_list]):
            print('missing: %r' % pkgname)
            missing += [pkgname]
    return candidate_list, missing


def get_unofficial_package_hrefs(force=None):
    """
    Downloads the entire webpage of available hrefs
    """
    if force is None:
        force = FORCE

    cachedir = ut.get_app_resource_dir('utool')
    try:
        if force:
            raise Exception('cachemiss')
        all_href_list = ut.load_cache(cachedir, 'win32_hrefs', 'all_href_list')
        page_str      = ut.load_cache(cachedir, 'win32_hrefs', 'page_str')
        print('all_href_list cache hit')
        return all_href_list, page_str
    except Exception:
        print('all_href_list cache miss')
        pass
    # Read page html
    headers = { 'User-Agent' : 'Mozilla/5.0' }
    print('Sending request to %r' % (UNOFFICIAL_WEBURL,))
    req = urllib2.Request(UNOFFICIAL_WEBURL, None, headers)
    page = urllib2.urlopen(req)
    page_str = page.read()
    encrypted_lines = list(filter(lambda x: x.find('onclick') > -1, page_str.split('\n')))

    print('Read %d encrypted lines ' % (len(encrypted_lines,)))
    # List of all download links, now choose wisely, because we don't want
    # to hack for evil
    #line = encrypted_lines[0]
    def parse_encrypted(line):
        """
        <script type="text/javascript">
        // <![CDATA[
        if (top.location!=location) top.location.href=location.href;
        function dc(ml,mi){
            var ot="";
            for(var j=0;j<mi.length;j++)
                ot+=String.fromCharCode(ml[mi.charCodeAt(j)-48]);
            document.write(ot);
            }
        function dl1(ml,mi){
            var ot="";
            for(var j=0;j<mi.length;j++)
                ot+=String.fromCharCode(ml[mi.charCodeAt(j)-48]);
            location.href=ot;
            }
        function dl(ml,mi){
        mi=mi.replace('&lt;','<');
        mi=mi.replace('&gt;','>');
        mi=mi.replace('&amp;','&');
        setTimeout(function(){ dl1(ml,mi) }, 1500);}
        // ]]>
        </script>
        #start = line.find('javascript:dl') + len('javascript:dl') + 2
        #end   = line.find('title') - 4
        #code = line[start: end]
        #mid = code.find(']')
        #left = code[0:mid]
        #right = code[mid + 4:]
        #ml = left
        #mi = right
        """
        _, ml, mi, _ = parse.parse('{}javascript:dl([{}], "{}"){}', line)
        mi_ = mi.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

        #ml_ = eval('[' + ml + ']')
        ml_ = eval(ml)
        href_ = ''.join([chr(ml_[ord(michar) - 48]) for michar in mi_])
        href  = ''.join([UNOFFICIAL_WEBURL, href_])
        return href
    all_href_list = list(map(parse_encrypted, encrypted_lines))
    print('decrypted %d lines' % (len(all_href_list)))
    ut.save_cache(cachedir, 'win32_hrefs', 'all_href_list', all_href_list)
    ut.save_cache(cachedir, 'win32_hrefs', 'page_str', page_str)
    return all_href_list, page_str


def uninstall_everything_win32():
    r"""
    try to figure out way to uninstall things easy

    References:
        http://msdn.microsoft.com/en-us/library/aa372024(VS.85).aspx
        http://www.sevenforums.com/tutorials/272460-programs-uninstall-using-command-prompt-windows.html
    """
    uninstall_script = ut.codeblock(  # NOQA
        """
        Msiexec /uninstall "Python 2.7 ipython-2.1.0"
        Msiexec /uninstall "Python 2.7 ipython-2.1.0"
        Msiexec /x opencv-python-py2.7.msi


        Msiexec /x "Python 2.7 opencv-python-2.4.10" /passive

        C:/Python27/Removenumpy.exe
        C:\Python27\Removeopencv-python.exe
        Msiexec /uninstall "C:\Python27\Removeopencv-python.exe"

        Unofficial package uninstall commands were found in the regestry here:
        HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall

        python -c "import utool"

        # THESE COMMAND WORK
        "C:/Python27/Removeopencv-python.exe" -u "C:/Python27/opencv-python-wininst.log"
        "C:/Python27/Removenumpy.exe" -u "C:/Python27/numpy-wininst.log"
        "C:/Python27/Removescipy.exe" -u "C:/Python27/scipy-wininst.log"
        "C:/Python27/RemoveCython.exe" -u "C:/Python27/Cython-wininst.log"
        "C:/Python27/Removeh5py.exe" -u "C:/Python27/h5py-wininst.log"
        "C:/Python27/Removeipython.exe" -u "C:/Python27/ipython-wininst.log"
        "C:/Python27/RemovePillow.exe" -u "C:/Python27/Pillow-wininst.log"
        "C:/Python27/Removematplotlib.exe" -u "C:/Python27/matplotlib-wininst.log"
        "C:/Python27/Removepsutil.exe" -u "C:/Python27/psutil-wininst.log"
        "C:/Python27/Removeline_profiler.exe" -u "C:/Python27/line_profiler-wininst.log"
        "C:/Python27/RemovePygments.exe" -u "C:/Python27/Pygments-wininst.log"
        "C:/Python27/Removepyparsing.exe" -u "C:/Python27/pyparsing-wininst.log"
        "C:/Python27/Removepyreadline.exe" -u "C:/Python27/pyreadline-wininst.log"
        "C:/Python27/Removepywin32.exe" -u "C:/Python27/pywin32-wininst.log"
        "C:/Python27/Removepyzmq.exe" -u "C:/Python27/pyzmq-wininst.log"
        "C:/Python27/Removesix.exe" -u "C:/Python27/six-wininst.log"
        "C:/Python27/RemoveSphinx.exe" -u "C:/Python27/Sphinx-wininst.log"
        "C:/Python27/Removetornado.exe" -u "C:/Python27/tornado-wininst.log"

        "C:/Python27/Removetables.exe" -u "C:/Python27/tables-wininst.log"
        "C:/Python27/Removenumexpr.exe" -u "C:/Python27/numexpr-wininst.log"
        "C:/Python27/Removepandas.exe" -u "C:/Python27/pandas-wininst.log"

        python -c "import utool as ut; print('\n'.join(ut.glob('C:/Python27', 'Remove*.exe')))"


        pip list
        pip uninstall Sphinx -y
        pip uninstall UNKNOWN -y
        pip uninstall Theano -y
        pip uninstall Pillow -y
        pip uninstall python-qt -y
        pip uninstall pyrf -y
        pip uninstall pyfiglet -y
        pip uninstall pyhesaff -y
        pip uninstall networkx -y
        pip uninstall detecttools -y
        pip uninstall Flask -y
        pip uninstall flann -y
        pip uninstall psutil -y
        pip uninstall simplejson -y
        pip uninstall objgraph -y
        pip uninstall selenium -y
        pip uninstall scikit-image -y
        pip uninstall scikit-learn -y
        pip uninstall statsmodels -y
        pip uninstall parse -y
        pip uninstall astroid -y
        pip uninstall colorama -y
        pip uninstall coverage -y
        pip uninstall decorator -y
        pip uninstall greenlet -y
        pip uninstall gevent -y
        pip uninstall guppy -y
        pip uninstall memory-profiler -y
        pip uninstall nose -y
        pip uninstall utool -y
        pip uninstall rope -y
        pip uninstall requests -y
        pip uninstall sphinxcontrib-napoleon -y
        pip uninstall RunSnakeRun -y
        pip uninstall SquareMap -y
        pip uninstall PyInstaller -y
        pip uninstall pytest -y
        pip uninstall pylru -y
        pip uninstall setproctitle -y
        pip uninstall functools32 -y

        pip install argparse --upgrade
        pip install virtualenv --upgrade
        pip install astor --upgrade
        pip install autopep8 --upgrade
        pip install docutils --upgrade
        pip install editdistance --upgrade
        pip install flake8 --upgrade
        pip install importlib --upgrade
        pip install openpyxl --upgrade
        pip install pep8 --upgrade
        pip install pip --upgrade
        pip install pyfiglet --upgrade
        pip install pyflakes --upgrade
        pip install pylint --upgrade
        pip install python-dateutil --upgrade
        pip install python-Levenshtein --upgrade
        pip install pytz --upgrade
        pip install rope --upgrade
        pip install setuptools --upgrade
        pip install six --upgrade
        pip install tox --upgrade
        pip install Werkzeug --upgrade
        pip install WinSys-3.x --upgrade
        pip install backports.ssl-match-hostname --upgrade
        pip install certifi --upgrade
        pip install distribute --upgrade
        pip install dragonfly --upgrade
        pip install itsdangerous --upgrade
        pip install jedi --upgrade
        pip install Jinja2 --upgrade
        pip install logilab-common --upgrade
        pip install MarkupSafe --upgrade
        pip install mccabe --upgrade
        pip install patsy --upgrade
        pip install py --upgrade
        pip install pycom --upgrade


        C:\Users\joncrall\AppData\Roaming\utool\pip-1.5.6.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\python-dateutil-2.2.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\setuptools-5.8.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\scipy-0.14.0.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\PyQt4-4.11.3-gpl-Py2.7-Qt4.8.6-x64.exe

        C:\Users\joncrall\AppData\Roaming\utool\requests-2.4.3.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\psutil-2.1.3.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\python-dateutil-2.2.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\pyreadline-2.0.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\pyparsing-2.0.3.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\Pygments-2.0.1.win32-py2.7.exe

        C:\Users\joncrall\AppData\Roaming\utool\pyzmq-14.4.1.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\tornado-4.0.2.win32-py2.7.exe

        C:\Users\joncrall\AppData\Roaming\utool\Pillow-2.6.1.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\Cython-0.21.1.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\ipython-2.3.1.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\numpy-MKL-1.9.1.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\scipy-0.15.0b1.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\scikit-learn-0.15.2.win32-py2.7.exe
        C:\Users\joncrall\AppData\Roaming\utool\matplotlib-1.4.2.win32-py2.7.exe
        """)


if __name__ == '__main__':
    main()
