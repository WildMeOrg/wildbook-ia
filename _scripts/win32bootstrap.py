from __future__ import division, print_function
from selenium import webdriver
import os
import parse
import distutils
from os.path import join, normpath
# Get packages we are updating


pkgname_list = [
    'Pygments>=1.6',
    'argparse>=1.2.1',
    'openpyxl>=1.6.2',  # reads excel xlsx files
    'parse>=1.6.2',
    'psutil>=1.0.1',
    'pyglet>=1.1.4',
    'pyparsing>=2.0.1',
    'pyreadline>=2.0',
    'python-dateutil>=1.5',
    'pyzmq>=13.1.0',  # distributed computing
    'six>=1.3.0',  # python 3 support
]


USE_FIREFOX = False

OS_VERSION = 'win32'
PY_VERSION = 'py2.7'
TARGET_PLATFORM = '.%s-%s.exe' % (OS_VERSION, PY_VERSION)

UNSUPPORTED = [
    'parse',
    'openpyxl',
    'pylint',
    'PIL',  # has pillow though
]

cwd = os.getcwd()
dldir = join(cwd, 'tmpdl')
try:
    os.makedirs(dldir)
except Exception as ex:
    print(ex)
    pass

if USE_FIREFOX:
    browser = webdriver.Firefox()

    fp = browser.firefox_profile
    fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/exe")
    fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/exe")
    fp.set_preference("browser.download.manager.showWhenStarting", False)
    fp.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/binary")
    fp.set_preference('browser.download.folderList', 2)
    fp.set_preference('browser.download.dir', dldir)
else:
    #install32 = normpath(r'C:\Program Files (x86)')
    #chromepath = normpath(r'\Google\Chrome\Application')
    #chromeexe = join(install32, chromepath, 'chrome.exe')
    import utool
    DRIVER_URL = 'http://chromedriver.storage.googleapis.com/2.9/chromedriver_win32.zip'
    chromedriverexe = utool.get_app_resource_dir('utool') + '/chromedriver.exe'
    if not utool.checkpath(chromedriverexe):
        utool.grab_zipped_url(DRIVER_URL, appname='utool')
    print(type(chromedriverexe))
    print(chromedriverexe)
    #chromedriverexe = normpath(r'C:\Users\joncrall\local\PATH\chromedriver.exe')
    browser = webdriver.Chrome(executable_path=chromedriverexe)

weburl = 'http://www.lfd.uci.edu/~gohlke/pythonlibs'
browser.get(weburl)

source = browser.page_source


def clean_package_names(pkgname_list):
    clean_list = []
    unsuported_list = []
    for pkgname in pkgname_list:
        if pkgname == 'numpy':
            pkgname = 'numpy-MKL'
        elif pkgname in UNSUPPORTED:
            unsuported_list += [pkgname]
            continue
        clean_list += [pkgname]
    return clean_list, unsuported_list


def find_elements(pkgname):
    element_list = browser.find_elements_by_partial_link_text(pkgname)
    element_list2 = []
    print('Searching for: %r' % pkgname)
    for element in element_list:
        if element.text.find(TARGET_PLATFORM) != -1:
            print('  Found: %r' % element.text)
            element_list2 += [element]
    return (pkgname, element_list2)


def choose_element(pkgname, element_list2):
    if len(element_list2) == 0:
        print('choose %r = %r' % (pkgname, None))
        return (pkgname, None)
    if len(element_list2) == 1:
        element = element_list2[0]
    else:
        def version_from_text(text):
            text = text.replace(pkgname + '-', '')
            text = text.replace(TARGET_PLATFORM, '')
            if ord(text[0]) >= 48 and ord(text[0]) <= 57:
                return distutils.version.LooseVersion(text)
            else:
                return '0.0.0*'
        element = None
        print('options: ')
        for element_ in element_list2:
            print(' * %r' % element_.text)
            version_ = version_from_text(element_.text)
            if element_ is None:
                continue
            elif element is None:
                element = element_
            else:
                version = version_from_text(element.text)
                if version_ > version:
                    element = element_
    print('choose %r = %r' % (pkgname, element.text))
    return (pkgname, element)


def get_true_href(element):
    javascript = element.get_attribute('onclick')
    # Javascript port dl
    ml, mi = parse.parse('javascript:dl([{}], "{}")', javascript)
    ml = '[%s]' % ml
    mi = '%s' % mi
    mi = mi.replace('&lt;', '<')
    mi = mi.replace('&gt;', '>')
    mi = mi.replace('&amp;', '&')
    ml = eval(ml)

    # Javascript port dl1
    href_ = ''.join([chr(ml[ord(char) - 48]) for char in mi])
    href  = '/'.join([weburl, href_])
    return href

pkgname_list, unsuported_list  = clean_package_names(pkgname_list)
print('Unsupported packages: %r' % (unsuported_list,))

# Find elements in the browser
elements_dict = dict(find_elements(pkgname) for pkgname in pkgname_list)
# Fix conflicts
element_list = [choose_element(*tup) for tup in elements_dict.iteritems()]
