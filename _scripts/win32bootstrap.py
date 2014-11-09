"""
Please be nice to this guys server.
This file is a big freaking hack
Use it responsibly. Dont be a dick.
"""
from __future__ import division, print_function
import parse
import utool
from six.moves import filterfalse
import urllib2

unofficial_weburl = 'http://www.lfd.uci.edu/~gohlke/pythonlibs/'
OS_VERSION = 'win32'
PY_VERSION = 'py2.7'
#PY_VERSION = 'py3.4'


def main():
    get_win_python_packages()


def get_win_python_packages():
    py_version = PY_VERSION
    #python34_win32_x64_url = 'https://www.python.org/ftp/python/3.4.1/python-3.4.1.amd64.msi'
    #python34_win32_x86_exe = utool.grab_file_url(python34_win32_x64_url)
    href_list = _get_win_packages_href(py_version)
    pkg_exe_list = []
    href = href_list[0]
    pkg_exe = utool.util_grabdata.grab_file_url(href, delay=3, spoof=True)
    pkg_exe_list += [pkg_exe]

    for href in href_list:
        pkg_exe = utool.util_grabdata.grab_file_url(href, delay=3, spoof=True)
        pkg_exe_list += [pkg_exe]
    print('\n'.join(href_list))
    print('Please Run:')
    print('\n'.join(pkg_exe_list))


def _get_win_packages_href(py_version):
    all_href_list, page_str = get_unofficial_package_hrefs(unofficial_weburl)
    win_pkg_list = [
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
        'six',
        'dateutil',
        'pyreadline',
        'pyparsing',
        #'sip',
        'PyQt4',
        'Pillow',
        'numpy-MKL-1.9',  # 'numpy',
        'scipy',
        'ipython',
        'tornado',
        'matplotlib',
        'scikit-learn',
    ]
    href_list1, missing  = filter_href_list(all_href_list, win_pkg_list, OS_VERSION, py_version)
    href_list2, missing2 = filter_href_list(all_href_list, missing, OS_VERSION, py_version)
    href_list3, missing3 = filter_href_list(all_href_list, missing2, 'x64', py_version.replace('p', 'P'))
dd
    href_list = href_list1 + href_list2 + href_list3
    return href_list


def get_unofficial_package_hrefs(unofficial_weburl):
    # Read page html
    headers = { 'User-Agent' : 'Mozilla/5.0' }
    req = urllib2.Request(unofficial_weburl, None, headers)
    page = urllib2.urlopen(req)
    page_str = page.read()
    encrypted_lines = filter(lambda x: x.find('onclick') > -1, page_str.split('\n'))
    # List of all download links, now choose wisely, because we don't want
    # to hack for evil
    #line = encrypted_lines[0]
    all_href_list = list(map(parse_encrypted, encrypted_lines))
    return all_href_list, page_str


def filter_href_list(all_href_list, win_pkg_list, os_version, py_version):
    candidate_list = []
    for pkgname in win_pkg_list:
        candidates = filter(lambda x: x.find(pkgname) > -1, all_href_list)
        candidate_list.extend(candidates)
    filtered_list1 = candidate_list
    filtered_list2 = filter(lambda x: py_version in x, filtered_list1)
    filtered_list3 = filter(lambda x: os_version in x, filtered_list2)
    bad_list = [
        'vigranumpy',
    ]
    filtered_list4 = list(filterfalse(lambda x: any([bad in x for bad in bad_list]), filtered_list3))

    missing = []
    for pkgname in win_pkg_list:
        if not any([pkgname in href for href in filtered_list4]):
            print('missing: %r' % pkgname)
            missing += [pkgname]
    return filtered_list4, missing


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
    href  = ''.join([unofficial_weburl, href_])
    return href


#if __name__ == '__main__':
#    main()
