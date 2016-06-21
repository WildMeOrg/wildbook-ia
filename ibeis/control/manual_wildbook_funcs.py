# -*- coding: utf-8 -*-
"""
CommandLine;
    # Reset IBEIS database (can skip if done)
    python -m ibeis.tests.reset_testdbs --reset_mtest
    python -m ibeis --tf reset_mtest

Notes:
    Moving compoments: java, tomcat, wildbook.war.


CommandLine;
    # Start IA server
    python -m ibeis --web --db PZ_MTEST

    # Reset Wildbook database
    python -m ibeis purge_local_wildbook

    # Install Wildbook
    python -m ibeis install_wildbook

    # Startup Wildbook
    python -m ibeis startup_wildbook_server
    --show

    # Login to wildbook (can skip)
    python -m ibeis test_wildbook_login

    # Ship ImageSets to wildbook
    python -m ibeis wildbook_signal_imgsetid_list

    # Change annotations names to a single name
    python -m ibeis wildbook_signal_annot_name_changes:1

    # Change annotations names back to normal
    python -m ibeis wildbook_signal_annot_name_changes:2

Utils:

    # TODO go to http://localhost:8080/ibeis/createAssetStore.jsp

        tail -f ~/.config/ibeis/tomcat/logs/catalina.out

        python -m ibeis shutdown_wildbook_server

MySQL:
    sudo apt-get install mysql-server-5.6
    sudo apt-get install mysql-common-5.6
    sudo apt-get install mysql-client-5.6

    mysql -u root -p

    create user 'ibeiswb'@'localhost' identified by 'somepassword';
    create database ibeiswbtestdb;
    grant all privileges on ibeiswbtestdb.* to 'ibeiswb'@'localhost';


"""
from __future__ import absolute_import, division, print_function
import six  # NOQA
import re
import utool as ut
import lockfile
import os
import requests
import time
import subprocess
from os.path import dirname, join, basename, splitext
from ibeis.control import controller_inject
from ibeis.control.controller_inject import make_ibs_register_decorator
print, rrr, profile = ut.inject2(__name__, '[manual_wildbook]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)

register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


#PREFERED_BROWSER = 'chrome'
#webbrowser._tryorder
PREFERED_BROWSER = None
if ut.get_computer_name() == 'hyrule':
    PREFERED_BROWSER = 'firefox'


# FIXME add as controller config
ALLOW_SYSTEM_TOMCAT = ut.get_argflag('--allow-system-tomcat')


def get_tomcat_startup_tmpdir():
    dpath_list = [
        #os.environ.get('CATALINA_TMPDIR', None),
        ut.ensure_app_resource_dir('ibeis', 'tomcat', 'ibeis_startup_tmpdir'),
    ]
    tomcat_startup_dir = ut.search_candidate_paths(dpath_list, verbose=True)
    return tomcat_startup_dir


@ut.tracefunc_xml
def find_tomcat(verbose=ut.NOT_QUIET):
    r"""
    Returns:
        str: tomcat_dpath

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --test-find_tomcat
        python -m ibeis --tf find_tomcat

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> tomcat_dpath = find_tomcat()
        >>> result = ('tomcat_dpath = %s' % (str(tomcat_dpath),))
        >>> print(result)
    """
    fname_list = ['Tomcat', 'tomcat']
    if ALLOW_SYSTEM_TOMCAT:
        # Places for system install of tomcat
        if ut.WIN32:
            dpath_list = ['C:/Program Files (x86)', 'C:/Program Files']
        else:
            dpath_list = ['/var/lib', '/usr/share', '/opt', '/lib']
        if ut.DARWIN:
            dpath_list = ['/Library'] + dpath_list
    else:
        dpath_list = []
    priority_paths = [
        # Number one preference is the CATALINA_HOME directory
        os.environ.get('CATALINA_HOME', None),
        # We put tomcat here if we can't find it
        ut.get_app_resource_dir('ibeis', 'tomcat')
    ]
    required_subpaths = ['webapps', 'bin', 'bin/catalina.sh']
    return_path = ut.search_candidate_paths(
        dpath_list, fname_list, priority_paths, required_subpaths,
        verbose=verbose)
    tomcat_dpath = return_path
    print('tomcat_dpath = %r ' % (tomcat_dpath,))
    return tomcat_dpath


@ut.tracefunc_xml
def download_tomcat():
    """
    Put tomcat into a directory controlled by ibeis

    CommandLine:
        # Reset
        python -c "import utool as ut; ut.delete(ut.unixjoin(ut.get_app_resource_dir('ibeis'), 'tomcat'))"
    """
    print('Grabbing tomcat')
    # FIXME: need to make a stable link
    if ut.WIN32:
        tomcat_binary_url = 'http://mirrors.advancedhosters.com/apache/tomcat/tomcat-8/v8.0.36/bin/apache-tomcat-8.0.36-windows-x86.zip'
    else:
        tomcat_binary_url = 'http://mirrors.advancedhosters.com/apache/tomcat/tomcat-8/v8.0.36/bin/apache-tomcat-8.0.36.zip'
    zip_fpath = ut.grab_file_url(tomcat_binary_url, appname='ibeis')
    # Download tomcat into the IBEIS resource directory
    tomcat_dpath = join(dirname(zip_fpath), 'tomcat')
    if not ut.checkpath(tomcat_dpath, verbose=True):
        # hack because unzipping is still weird
        ut.unzip_file(zip_fpath)
        tomcat_dpath_tmp = splitext(zip_fpath)[0]
        ut.move(tomcat_dpath_tmp, tomcat_dpath)
    if ut.checkpath(join(tomcat_dpath, 'bin'), verbose=True):
        scriptnames = ['catalina.sh', 'startup.sh', 'shutdown.sh']
        for fname in scriptnames:
            fpath = join(tomcat_dpath, 'bin', fname)
            if not ut.is_file_executable(fpath):
                print('Adding executable bits to script %r' % (fpath,))
                ut.chmod_add_executable(fpath)
    return tomcat_dpath


@ut.tracefunc_xml
def find_installed_tomcat(check_unpacked=True, strict=True):
    """
    Asserts that tomcat was properly installed

    Args:
        check_unpacked (bool): (default = True)

    Returns:
        str: tomcat_dpath

    CommandLine:
        python -m ibeis --tmod ibeis.control.manual_wildbook_funcs --exec-find_installed_tomcat
        python -m ibeis.control.manual_wildbook_funcs --exec-find_installed_tomcat
        python -m ibeis -tm ibeis.control.manual_wildbook_funcs --exec-find_installed_tomcat
        python -m ibeis --tf find_installed_tomcat

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> check_unpacked = True
        >>> strict = False
        >>> tomcat_dpath = find_installed_tomcat(check_unpacked, strict)
        >>> result = ('tomcat_dpath = %s' % (str(tomcat_dpath),))
        >>> print(result)
    """
    tomcat_dpath = find_tomcat()
    if tomcat_dpath is None:
        msg = 'Cannot find tomcat'
        if strict:
            raise ImportError(msg)
        else:
            print(msg)
            return None
    if check_unpacked:
        import ibeis
        # Check that webapps was unpacked
        wb_target = ibeis.const.WILDBOOK_TARGET
        webapps_dpath = join(tomcat_dpath, 'webapps')
        unpacked_war_dpath = join(webapps_dpath, wb_target)
        ut.assertpath(unpacked_war_dpath)
    return tomcat_dpath


def find_or_download_tomcat():
    r"""
    Returns:
        str: tomcat_dpath

    CommandLine:
        # Reset
        python -m ibeis.control.manual_wildbook_funcs --test-purge_local_wildbook
        python -m ibeis.control.manual_wildbook_funcs --test-find_or_download_tomcat

        python -m ibeis --tf purge_local_wildbook
        python -m ibeis --tf find_or_download_tomcat

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> tomcat_dpath = find_or_download_tomcat()
        >>> result = ('tomcat_dpath = %s' % (str(tomcat_dpath),))
        >>> print(result)
    """
    tomcat_dpath = find_tomcat()
    if tomcat_dpath is None:
        tomcat_dpath = download_tomcat()
    else:
        ut.assertpath(tomcat_dpath)
    return tomcat_dpath


def find_java_jvm():
    r"""
    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --test-find_java_jvm

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> result = find_java_jvm()
        >>> print(result)
    """
    candidate_path_list = [
        #os.environ.get('JAVA_HOME', None),
        #'/usr/lib/jvm/java-7-openjdk-amd64',
    ]
    jvm_fpath = ut.search_candidate_paths(candidate_path_list, verbose=True)
    ut.assertpath(jvm_fpath, 'IBEIS cannot find Java Runtime Environment')
    return jvm_fpath


def find_or_download_wilbook_warfile(redownload=False):
    r"""
    scp jonc@pachy.cs.uic.edu:/var/lib/tomcat/webapps/ibeis.war \
            ~/Downloads/pachy_ibeis.war wget
    http://dev.wildme.org/ibeis_data_dir/ibeis.war
    """
    #war_url = 'http://dev.wildme.org/ibeis_data_dir/ibeis.war'
    war_url = 'http://springbreak.wildbook.org/tools/latest.war'
    war_fpath = ut.grab_file_url(war_url, appname='ibeis',
                                 redownload=redownload, fname='ibeis.war')
    return war_fpath


def purge_local_wildbook():
    r"""
    Shuts down the server and then purges the server on disk

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs purge_local_wildbook

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> purge_local_wildbook()
    """
    try:
        shutdown_wildbook_server()
    except ImportError:
        pass
    ut.delete(ut.unixjoin(ut.get_app_resource_dir('ibeis'), 'tomcat'))


@ut.tracefunc_xml
def install_wildbook(verbose=ut.NOT_QUIET):
    """
    Script to setup wildbook on a unix based system
    (hopefully eventually this will generalize to win32)

    CommandLine:
        # Reset
        ibeis purge_local_wildbook
        # Setup
        ibeis install_wildbook
        # Startup
        ibeis startup_wildbook_server --show

        Alternates:
            ibeis install_wildbook --redownload-war
            ibeis startup_wildbook_server --show

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> verbose = True
        >>> result = install_wildbook()
        >>> print(result)
    """
    import requests
    # TODO: allow custom specified tomcat directory
    try:
        output = subprocess.check_output(['java', '-version'],
                                         stderr=subprocess.STDOUT)
        _java_version = output.split('\n')[0]
        _java_version = _java_version.replace('java version ', '')
        java_version = _java_version.replace('"', '')
        print('java_version = %r' % (java_version,))
        if not java_version.startswith('1.7'):
            print('Warning wildbook is only supported for java 1.7')
    except OSError:
        output = None
    if output is None:
        raise ImportError(
            'Cannot find java on this machine. '
            'Please install java: http://www.java.com/en/download/')

    tomcat_dpath = find_or_download_tomcat()
    assert tomcat_dpath is not None, 'Could not find tomcat'
    redownload = ut.get_argflag('--redownload-war')
    war_fpath = find_or_download_wilbook_warfile(redownload=redownload)
    war_fname = basename(war_fpath)
    wb_target = splitext(war_fname)[0]

    # Ensure environment variables
    #os.environ['JAVA_HOME'] = find_java_jvm()
    #os.environ['TOMCAT_HOME'] = tomcat_dpath
    #os.environ['CATALINA_HOME'] = tomcat_dpath

    # Move the war file to tomcat webapps if not there
    webapps_dpath = join(tomcat_dpath, 'webapps')
    deploy_war_fpath = join(webapps_dpath, war_fname)
    if not ut.checkpath(deploy_war_fpath, verbose=verbose):
        ut.copy(war_fpath, deploy_war_fpath)

    # Ensure that the war file has been unpacked

    unpacked_war_dpath = join(webapps_dpath, wb_target)
    tomcat_startup_dir = get_tomcat_startup_tmpdir()
    fresh_install = not ut.checkpath(unpacked_war_dpath, verbose=verbose)
    if fresh_install:
        # Need to make sure you start catalina in the same directory otherwise
        # the derby databsae gets put in in cwd
        with ut.ChdirContext(tomcat_startup_dir):
            # Starting and stoping catalina should be sufficient to unpack the
            # war
            startup_fpath  = join(tomcat_dpath, 'bin', 'startup.sh')
            #shutdown_fpath = join(tomcat_dpath, 'bin', 'shutdown.sh')
            ut.cmd(ut.quote_single_command(startup_fpath))
            print('It is NOT ok if the startup.sh fails\n')

            # wait for the war to be unpacked
            for retry_count in range(0, 6):
                time.sleep(1)
                if ut.checkpath(unpacked_war_dpath, verbose=True):
                    break
                else:
                    print('Retrying')

            # ensure that the server is ruuning
            print('Checking if we can ping the server')
            response = requests.get('http://localhost:8080')
            if response is None or response.status_code != 200:
                print('There may be an error starting the server')
            else:
                print('Seem able to ping the server')

            # assert tht the war was unpacked
            ut.assertpath(unpacked_war_dpath, (
                'Wildbook war might have not unpacked correctly.  This may '
                'be ok. Try again. If it fails a second time, then there is a '
                'problem.'), verbose=True)

            # shutdown the server
            #ut.cmd(ut.quote_single_command(shutdown_fpath))
            #print('It is ok if the shutdown.sh fails')
            #time.sleep(.5)

    #if ut.get_argflag('--vd'):
    #    ut.vd(unpacked_war_dpath)

    #find_installed_tomcat

    # Make sure permissions are correctly set in wildbook
    # Comment out the line that requires authentication
    permission_fpath = join(unpacked_war_dpath, 'WEB-INF/web.xml')
    ut.assertpath(permission_fpath)
    permission_text = ut.readfrom(permission_fpath)
    lines_to_remove = [
        # '/ImageSetSetMarkedIndividual = authc, roles[admin]'
        '/EncounterSetMarkedIndividual = authc, roles[admin]'
    ]
    new_permission_text = permission_text[:]
    for line in lines_to_remove:
        re.search(re.escape(line), permission_text)
        prefix = ut.named_field('prefix', '\\s*')
        suffix = ut.named_field('suffix', '\\s*\n')
        pattern = ('^' + prefix + re.escape(line) + suffix)
        match = re.search(pattern, permission_text,
                          flags=re.MULTILINE | re.DOTALL)
        if match is None:
            continue
        newline = '<!--%s -->' % (line,)
        repl = ut.bref_field('prefix') + newline + ut.bref_field('suffix')
        new_permission_text = re.sub(pattern, repl, permission_text,
                                     flags=re.MULTILINE | re.DOTALL)
        assert new_permission_text != permission_text, (
            'text should have changed')
    if new_permission_text != permission_text:
        print('Need to write new permission texts')
        ut.writeto(permission_fpath, new_permission_text)
    else:
        print('Permission file seems to be ok')

    # Make sure we are using a non-process based database
    jdoconfig_fpath = join(unpacked_war_dpath, 'WEB-INF/classes/bundles/jdoconfig.properties')
    print('Fixing backend database config')
    print('jdoconfig_fpath = %r' % (jdoconfig_fpath,))
    ut.assertpath(jdoconfig_fpath)
    jdoconfig_text = ut.readfrom(jdoconfig_fpath)
    #ut.vd(dirname(jdoconfig_fpath))
    #ut.editfile(jdoconfig_fpath)

    def toggle_comment_lines(text, pattern, flag):
        lines = text.split('\n')
        if flag:
            lines = ['#' + line if re.search(pattern, line) else line for line in lines]
        else:
            lines = [line.lstrip('#') if re.search(pattern, line) else line for line in lines]
        return '\n'.join(lines)
    #jdoconfig_text = toggle_comment_lines(jdoconfig_text, 'derby', False)
    sql_mode = True
    if sql_mode:
        #jdoconfig_text = toggle_comment_lines(jdoconfig_text, 'sql', True)
        # TODO: Set these lines appropriately
        jdoconfig_text = re.sub('datanucleus.ConnectionUserName = .*$', 'datanucleus.ConnectionUserName = ibeiswb', jdoconfig_text, flags=re.MULTILINE)
        jdoconfig_text = re.sub('datanucleus.ConnectionPassword = .*$', 'datanucleus.ConnectionPassword = somepassword', jdoconfig_text, flags=re.MULTILINE)
        jdoconfig_text = re.sub('oldWildbook', 'ibeiswbtestdb', jdoconfig_text, flags=re.MULTILINE)
        """
        datanucleus.ConnectionUserName = wildbook
        datanucleus.ConnectionPassword = wildbook
        """
        pass
    else:
        jdoconfig_text = toggle_comment_lines(jdoconfig_text, 'sqlite', False)
        jdoconfig_text = toggle_comment_lines(jdoconfig_text, 'mysql', True)
    ut.writeto(jdoconfig_fpath, jdoconfig_text)

    # Need to make sure wildbook can store information in a reasonalbe place
    #tomcat_data_dir = join(tomcat_startup_dir, 'webapps', 'wildbook_data_dir')
    tomcat_data_dir = join(webapps_dpath, 'wildbook_data_dir')
    ut.ensuredir(tomcat_data_dir)
    ut.writeto(join(tomcat_data_dir, 'test.txt'), 'A hosted test file')
    asset_store_fpath = join(unpacked_war_dpath, 'createAssetStore.jsp')
    asset_store_text = ut.read_from(asset_store_fpath)
    #data_path_pat = ut.named_field('data_path', 'new File(".*?").toPath')
    new_line = 'LocalAssetStore as = new LocalAssetStore("example Local AssetStore", new File("%s").toPath(), "%s", true);' % (
        tomcat_data_dir,
        'http://localhost:8080/' + basename(tomcat_data_dir)
    )
    # HACKY
    asset_store_text2 = re.sub('^LocalAssetStore as = .*$', new_line, asset_store_text, flags=re.MULTILINE)
    ut.writeto(asset_store_fpath, asset_store_text2)
    #ut.editfile(asset_store_fpath)

    # Pinging the server to create asset store
    # Ensureing that createAssetStore exists
    if fresh_install:
        #web_url = startup_wildbook_server(verbose=False)
        print('Creating asset store')
        response = requests.get('http://localhost:8080/' + wb_target + '/createAssetStore.jsp')
        if response is None or response.status_code != 200:
            print('There may be an error starting the server')
            #if response.status_code == 500:
            print(response.text)
            import utool
            utool.embed()
        else:
            print('Created asset store')
        shutdown_wildbook_server(verbose=False)
        print('It is ok if the shutdown fails')

    #127.0.0.1:8080/wildbook_data_dir/test.txt
    print('Wildbook is installed and waiting to be started')


@ut.tracefunc_xml
def startup_wildbook_server(verbose=ut.NOT_QUIET):
    r"""
    Args:
        verbose (bool):  verbosity flag(default = True)

    CommandLine:
        python -m ibeis startup_wildbook_server
        python -m ibeis startup_wildbook_server  --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> verbose = True
        >>> wb_url = startup_wildbook_server()
        >>> ut.quit_if_noshow()
        >>> ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(wb_url)
    """
    # TODO: allow custom specified tomcat directory
    import ibeis
    tomcat_dpath = find_installed_tomcat()

    # Ensure environment variables
    #os.environ['JAVA_HOME'] = find_java_jvm()
    #os.environ['TOMCAT_HOME'] = tomcat_dpath
    #os.environ['CATALINA_HOME'] = tomcat_dpath

    with ut.ChdirContext(get_tomcat_startup_tmpdir()):
        startup_fpath  = join(tomcat_dpath, 'bin', 'startup.sh')
        ut.cmd(ut.quote_single_command(startup_fpath))
        time.sleep(1)
    wb_url = 'http://localhost:8080/' + ibeis.const.WILDBOOK_TARGET
    # TODO go to http://localhost:8080/ibeis/createAssetStore.jsp
    return wb_url


def shutdown_wildbook_server(verbose=ut.NOT_QUIET):
    r"""

    Args:
        verbose (bool):  verbosity flag(default = True)

    Ignore:
        tail -f ~/.config/ibeis/tomcat/logs/catalina.out

    CommandLine:
        python -m ibeis shutdown_wildbook_server

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> verbose = True
        >>> wb_url = shutdown_wildbook_server()
        >>> ut.quit_if_noshow()
        >>> ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(wb_url)
    """
    # TODO: allow custom specified tomcat directory
    tomcat_dpath = find_installed_tomcat(check_unpacked=False)
    # TODO: allow custom specified tomcat directory
    #tomcat_dpath = find_installed_tomcat(check_unpacked=False)
    #catalina_out_fpath = join(tomcat_dpath, 'logs', 'catalina.out')

    # Ensure environment variables
    #os.environ['JAVA_HOME'] = find_java_jvm()
    #os.environ['TOMCAT_HOME'] = tomcat_dpath
    #os.environ['CATALINA_HOME'] = tomcat_dpath

    with ut.ChdirContext(get_tomcat_startup_tmpdir()):
        shutdown_fpath = join(tomcat_dpath, 'bin', 'shutdown.sh')
        #ut.cmd(shutdown_fpath)
        ut.cmd(ut.quote_single_command(shutdown_fpath))
        time.sleep(.5)


def monitor_wildbook_logs(verbose=ut.NOT_QUIET):
    r"""
    Args:
        verbose (bool):  verbosity flag(default = True)

    CommandLine:
        python -m ibeis monitor_wildbook_logs  --show

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> monitor_wildbook_logs()
    """
    # TODO: allow custom specified tomcat directory
    import ibeis
    tomcat_dpath = find_installed_tomcat()

    # Ensure environment variables
    #os.environ['JAVA_HOME'] = find_java_jvm()
    #os.environ['TOMCAT_HOME'] = tomcat_dpath
    #os.environ['CATALINA_HOME'] = tomcat_dpath

    with ut.ChdirContext(get_tomcat_startup_tmpdir()):
        startup_fpath  = join(tomcat_dpath, 'bin', 'startup.sh')
        ut.cmd(ut.quote_single_command(startup_fpath))
        time.sleep(1)
    wb_url = 'http://localhost:8080/' + ibeis.const.WILDBOOK_TARGET
    return wb_url


def test_wildbook_login():
    r"""
    Helper function to test wildbook login automagically

    Returns:
        tuple: (wb_target, tomcat_dpath)

    CommandLine:
        python -m ibeis test_wildbook_login

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> test_wildbook_login()
    """
    # Use selenimum to login to wildbook
    import ibeis
    manaul_login = False
    wb_target = ibeis.const.WILDBOOK_TARGET
    wb_url = 'http://localhost:8080/' + wb_target
    if manaul_login:
        ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(wb_url)
    else:
        print('Grabbing Driver')
        driver = ut.grab_selenium_driver(PREFERED_BROWSER)
        print('Going to URL')
        if False:
            driver.get(wb_url)
            print('Finding Login Button')
            #login_button = driver.find_element_by_partial_link_text('Log in')
            login_button = driver.find_element_by_partial_link_text('welcome.jps')
            login_button.click()
        else:
            driver.get(wb_url + '/login.jsp')
        print('Finding Login Elements')
        username_field = driver.find_element_by_name('username')
        password_field = driver.find_element_by_name('password')
        submit_login_button = driver.find_element_by_name('submit')
        rememberMe_button = driver.find_element_by_name('rememberMe')
        print('Executing Login Elements')
        username_field.send_keys('tomcat')
        password_field.send_keys('tomcat123')
        rememberMe_button.click()
        submit_login_button.click()
        # Accept agreement
        import selenium.common.exceptions
        try:
            accept_aggrement_button = driver.find_element_by_name('acceptUserAgreement')
            accept_aggrement_button.click()
        except selenium.common.exceptions.NoSuchElementException:
            pass

        # Goto individuals page
        individuals = driver.find_element_by_partial_link_text('Individuals')
        individuals.click()
        view_all = driver.find_element_by_partial_link_text('View All')
        view_all.click()


def testdata_wildbook_server():
    """
    DEPRICATE
    SeeAlso:
        ~/local/build_scripts/init_wildbook.sh
    """
    # Very hacky and specific testdata script.
    #if ut.is_developer():
    #    tomcat_dpath = join(os.environ['CODE_DIR'],
    #                        'Wildbook/tmp/apache-tomcat-8.0.36')
    #else:
    #    tomcat_dpath = '/var/lib/tomcat'
    import ibeis
    tomcat_dpath = find_installed_tomcat()
    wb_target = ibeis.const.WILDBOOK_TARGET
    return wb_target, tomcat_dpath


@register_ibs_method
def get_wildbook_target(ibs):
    return ibs.const.WILDBOOK_TARGET


@register_ibs_method
def get_wildbook_info(ibs, tomcat_dpath=None, wb_target=None):
    # TODO: Clean this up
    wildbook_base_url = ibs.get_wildbook_base_url(wb_target)
    wildbook_tomcat_path = ibs.get_wildbook_tomcat_path(tomcat_dpath, wb_target)
    # Setup
    print('Looking for WildBook installation: %r' % ( wildbook_tomcat_path, ))
    ut.assert_exists(wildbook_tomcat_path,
                     'Wildbook is not installed on this machine', info=True)
    return wildbook_base_url, wildbook_tomcat_path


@register_ibs_method
def get_wildbook_tomcat_path(ibs, tomcat_dpath=None, wb_target=None):
    DEFAULT_TOMCAT_PATH = find_installed_tomcat()
    tomcat_dpath = DEFAULT_TOMCAT_PATH if tomcat_dpath is None else tomcat_dpath
    wb_target = ibs.const.WILDBOOK_TARGET if wb_target is None else wb_target
    wildbook_tomcat_path = join(tomcat_dpath, 'webapps', wb_target)
    return wildbook_tomcat_path


@register_ibs_method
def get_wildbook_base_url(ibs, wb_target=None):
    wb_target = ibs.const.WILDBOOK_TARGET if wb_target is None else wb_target
    hostname = '127.0.0.1'
    wb_port = 8080
    wildbook_base_url = 'http://' + str(hostname) + ':' + str(wb_port) + '/' + wb_target
    return wildbook_base_url


def submit_wildbook_url(url, payload=None, browse_on_error=True, dryrun=False,
                        timeout=2):
    """
    mirroring the one in IBEISController.py, but with changed functionality
    """
    if dryrun:
        print('[DRYrun_submit] URL=%r' % (url, ))
        response = None
        status = True
    else:
        if ut.VERBOSE:
            print('[submit] URL=%r' % (url, ))
        try:
            if payload is None:
                response = requests.get(url, timeout=timeout)
                #response = requests.get(url, auth=('tomcat', 'tomcat123'))
            else:
                response = requests.post(url, data=payload, timeout=timeout)
                #r = requests.post(url, data=None, auth=('tomcat', 'tomcat123'))
        except requests.ConnectionError as ex:
            ut.printex(ex, 'Could not connect to Wildbook server url=%r' % url)
            raise
        else:
            status = True
        if response is None or response.status_code != 200:
            errmsg_list = ([('Wildbook response NOT ok (200)')])
            if response is None:
                errmsg_list.extend([
                    ('WILDBOOK SERVER RESPONSE = %r' % (response, )),
                ])
            else:
                errmsg_list.extend([
                    ('WILDBOOK SERVER STATUS = %r' % (response.status_code,)),
                    ('WILDBOOK SERVER RESPONSE TEXT = %r' % (response.text,)),
                ])
            errmsg = '\n'.join(errmsg_list)
            print(errmsg)
            if browse_on_error:
                ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(url)
            raise AssertionError(errmsg)
            status = False
    return status, response


@ut.tracefunc_xml
def update_wildbook_config(ibs, wildbook_tomcat_path, dryrun=False):
    wildbook_properteis_dpath = join(wildbook_tomcat_path,
                                     'WEB-INF/classes/bundles/')
    print('[ibs.update_wildbook_config()] Wildbook properties=%r' % (
        wildbook_properteis_dpath, ))
    # The src file is non-standard. It should be remove here as well
    wildbook_config_fpath_dst = join(wildbook_properteis_dpath,
                                     'commonConfiguration.properties')
    ut.assert_exists(wildbook_properteis_dpath)
    # for come reason the .default file is not there, that should be ok though
    orig_content = ut.read_from(wildbook_config_fpath_dst)
    content = orig_content
    content = re.sub('IBEIS_DB_path = .*',
                     'IBEIS_DB_path = ' + ibs.get_db_core_path(), content)
    content = re.sub('IBEIS_image_path = .*',
                     'IBEIS_image_path = ' + ibs.get_imgdir(), content)

    # Make sure wildbook knows where to find us
    ia_rest_prefix = ut.named_field('prefix', 'IBEISIARestUrl.*')
    host_port = ut.named_field('host_port', 'http://.*?:[0-9]+')
    new_hostport = 'http://localhost:5000'
    content = re.sub(ia_rest_prefix + host_port, ut.bref_field('prefix') + new_hostport, content)

    # Write to the configuration if it is different
    if orig_content != content:
        need_sudo = not ut.is_file_writable(wildbook_config_fpath_dst)
        if need_sudo:
            quoted_content = '"%s"' % (content, )
            print('Attempting to gain sudo access to update wildbook config')
            command = ['sudo', 'sh', '-c', '\'', 'echo',
                       quoted_content, '>', wildbook_config_fpath_dst, '\'']
            # ut.cmd(command, sudo=True)
            command = ' '.join(command)
            if not dryrun:
                os.system(command)
        else:
            ut.write_to(wildbook_config_fpath_dst, content)


@register_ibs_method
@register_api('/api/wildbook/signal_annot_name_changes/', methods=['PUT'])
def wildbook_signal_annot_name_changes(ibs, aid_list=None, tomcat_dpath=None, wb_target=None, dryrun=False):
    r"""
    Args:
        aid_list (int):  list of annotation ids(default = None)
        tomcat_dpath (None): (default = None)
        wb_target (None): (default = None)
        dryrun (bool): (default = False)

    CommandLine:
        python -m ibeis wildbook_signal_annot_name_changes:0 --dryrun
        python -m ibeis wildbook_signal_annot_name_changes:1 --dryrun
        python -m ibeis wildbook_signal_annot_name_changes:1
        python -m ibeis wildbook_signal_annot_name_changes:2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> #gid_list = ibs.get_valid_gids()[0:10]
        >>> gid_list = ibs.get_valid_gids()[3:5]
        >>> aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        >>> # Test case where some names change, some do not. There are no new names.
        >>> old_nid_list = ibs.get_annot_name_rowids(aid_list)
        >>> new_nid_list = ut.list_roll(old_nid_list, 1)
        >>> ibs.set_annot_name_rowids(aid_list, new_nid_list)
        >>> dryrun = ut.get_argflag('--dryrun')
        >>> wb_target, tomcat_dpath = testdata_wildbook_server()
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, tomcat_dpath, wb_target, dryrun)
        >>> ibs.set_annot_name_rowids(aid_list, old_nid_list)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> #gid_list = ibs.get_valid_gids()[0:10]
        >>> gid_list = ibs.get_valid_gids()[3:5]
        >>> aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        >>> # Test case where all names change to one known name
        >>> #old_nid_list = ibs.get_annot_name_rowids(aid_list)
        >>> #new_nid_list = [old_nid_list[0]] * len(old_nid_list)
        >>> old_nid_list = [1, 2]
        >>> new_nid_list = [1, 1]
        >>> print('old_nid_list = %r' % (old_nid_list,))
        >>> print('new_nid_list = %r' % (new_nid_list,))
        >>> ibs.set_annot_name_rowids(aid_list, new_nid_list)
        >>> dryrun = ut.get_argflag('--dryrun')
        >>> wb_target, tomcat_dpath = testdata_wildbook_server()
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, tomcat_dpath, wb_target, dryrun)
        >>> # Undo changes here (not undone in wildbook)
        >>> #ibs.set_annot_name_rowids(aid_list, old_nid_list)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> gid_list = ibs.get_valid_gids()[3:5]
        >>> aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        >>> old_nid_list = [1, 2]
        >>> ibs.set_annot_name_rowids(aid_list, old_nid_list)
        >>> # Signal what currently exists (should put them back to normal)
        >>> dryrun = ut.get_argflag('--dryrun')
        >>> wb_target, tomcat_dpath = testdata_wildbook_server()
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, tomcat_dpath, wb_target, dryrun)
    """
    print('[ibs.wildbook_signal_imgsetid_list()] signaling any annotation name changes to wildbook')

    wildbook_base_url, wildbook_tomcat_path = ibs.get_wildbook_info(tomcat_dpath, wb_target)
    # url_command = 'ImageSetSetMarkedIndividual'
    url_command = 'EncounterSetMarkedIndividual'
    #BASIC_AUTH = False
    #if BASIC_AUTH:
    #    #url_command += '=authcBasicWildbook'
    #    username = 'tomcat'
    #    password = 'tomcat123'
    #    wildbook_base_url = ('http://' + username + ':' + password + '@' +
    #                         wildbook_base_url.replace('http://', ''))
    url_args_fmtstr = '&'.join([
        'annotID={annot_uuid!s}',
        'individualID={name_text!s}',
    ])
    submit_namchange_url_fmtstr = (
        wildbook_base_url + '/' + url_command + '?' + url_args_fmtstr)

    if aid_list is None:
        aid_list = ibs.get_valid_aids(is_known=True)
    # Build URLs to submit
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    annot_name_text_list = ibs.get_annot_name_texts(aid_list)
    submit_url_list = [
        submit_namchange_url_fmtstr.format(
            annot_uuid=str(annot_uuid), name_text=str(name_text))
        for annot_uuid, name_text in zip(annot_uuid_list, annot_name_text_list)
    ]

    payload = {}

    # Submit each URL
    status_list = []
    print('Submitting URL list')
    print(ut.indentjoin(submit_url_list))
    message_list = []
    for url in ut.ProgressIter(submit_url_list, lbl='submitting URL', freq=1):
        print(url)
        status, response = submit_wildbook_url(url, payload, dryrun=dryrun)
        #print(ut.dict_str(response.__dict__, truncate=0))
        status_list.append(status)
        try:
            response_json = response.json()
            # imageset in this message is a wb-imageset not our ia-imageset
            #if ut.VERBOSE:
            print(response_json['message'])
            message_list.append(str(response_json['message']))
        except Exception as ex:
            print(ut.indentjoin(message_list))
            ut.printex(ex, ('Failed getting json from response. '
                            'Is there an authentication issue?'))
            raise
        assert response_json['success']
    print(ut.indentjoin(message_list))
    return status_list


@register_ibs_method
@register_api('/api/wildbook/signal_imgsetid_list/', methods=['PUT'])
def wildbook_signal_imgsetid_list(ibs, imgsetid_list=None,
                                  set_shipped_flag=True,
                                  open_url_on_complete=True, tomcat_dpath=None,
                                  wb_target=None, dryrun=False):
    """
    Exports specified imagesets to wildbook. This is a synchronous call.

    Args:
        imgsetid_list (list): (default = None)
        set_shipped_flag (bool): (default = True)
        open_url_on_complete (bool): (default = True)

    RESTful:
        Method: PUT
        URL:    /api/wildbook/signal_imgsetid_list/

    Ignore:
        cd $CODE_DIR/Wildbook/tmp

        # Reset IBEIS database
        python -m ibeis.tests.reset_testdbs --reset_mtest
        python -m ibeis  reset_mtest

        # Completely remove Wildbook database
        python -m ibeis  purge_local_wildbook

        # Install Wildbook
        python -m ibeis  install_wildbook

        # Startup Wildbook
        python -m ibeis  startup_wildbook_server

        # Login to wildbook
        python -m ibeis  test_wildbook_login

        # Ship ImageSets to wildbook
        python -m ibeis  wildbook_signal_imgsetid_list

        # Change annotations names to a single name
        python -m ibeis  wildbook_signal_annot_name_changes:1

        # Change annotations names back to normal
        python -m ibeis  wildbook_signal_annot_name_changes:2

    CommandLine:
        python -m ibeis wildbook_signal_imgsetid_list
        python -m ibeis wildbook_signal_imgsetid_list --dryrun
        python -m ibeis wildbook_signal_imgsetid_list --break

    SeeAlso:
        ~/local/build_scripts/init_wildbook.sh

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> dryrun = ut.get_argflag('--dryrun')
        >>> wb_target, tomcat_dpath = testdata_wildbook_server()
        >>> import ibeis
        >>> # Need to start a web server for wildbook to hook into
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = ibeis.opendb(defaultdb=defaultdb)
        >>> #gid_list = ibs.get_valid_gids()[0:10]
        >>> gid_list = ibs.get_valid_gids()[3:6]
        >>> new_imgsetid = ibs.create_new_imageset_from_images(gid_list)  # NOQA
        >>> imgsetid = new_imgsetid
        >>> print('new imageset uuid = %r' % (ibs.get_imageset_uuid(new_imgsetid),))
        >>> print('new imageset text = %r' % (ibs.get_imageset_text(new_imgsetid),))
        >>> imgsetid_list = [new_imgsetid]
        >>> ibs.set_imageset_processed_flags([new_imgsetid], [1])
        >>> gid_list = ibs.get_imageset_gids(new_imgsetid)
        >>> ibs.set_image_reviewed(gid_list, [1] * len(gid_list))
        >>> set_shipped_flag = True
        >>> open_url_on_complete = True
        >>> if ut.get_argflag('--bg'):
        >>>     web_ibs = ibeis.opendb_bg_web(defaultdb)
        >>> result = ibs.wildbook_signal_imgsetid_list(imgsetid_list, set_shipped_flag, open_url_on_complete, tomcat_dpath, wb_target, dryrun)
        >>> # cleanup
        >>> #ibs.delete_imagesets(new_imgsetid)
        >>> print(result)
        >>> if ut.get_argflag('--bg'):
        >>>     web_ibs.terminate2()

    """
    # Configuration
    use_config_file = False
    wildbook_base_url, wildbook_tomcat_path = ibs.get_wildbook_info(
        tomcat_dpath, wb_target)

    # VIEW OCCURRENCE
    #http://localhost:8080/ibeis/occurrence.jsp?number=_______
    # VIEW ENCOUNTER
    #http://localhost:8080/ibeis/encounters/encounter.jsp?number=826c83fa-f15b-42a5-8382-74100a086d56

    if imgsetid_list is None:
        imgsetid_list = ibs.get_valid_imgsetids()
    # Check to make sure imagesets are ok:
    for imgsetid in imgsetid_list:
        # First, check if imageset can be pushed
        aid_list = ibs.get_imageset_aids(imgsetid)
        assert len(aid_list) > 0, (
            'ImageSet imgsetid=%r cannot be shipped with0 annots' % (imgsetid,))
        unknown_flags = ibs.is_aid_unknown(aid_list)
        unnamed_aid_list = ut.compress(aid_list, unknown_flags)
        assert len(unnamed_aid_list) == 0, (
            ('ImageSet imgsetid=%r cannot be shipped becuase '
             'annotation(s) %r have not been named') % (imgsetid, unnamed_aid_list, ))

    ## Call Wildbook url to signal update
    print('[ibs.wildbook_signal_imgsetid_list] ship imgsetid_list = %r to wildbook' % (
        imgsetid_list, ))

    # With a lock file, modify the configuration with the new settings
    lock_fpath = join(ibs.get_ibeis_resource_dir(), 'wildbook.lock')
    with lockfile.LockFile(lock_fpath):
        # Update the Wildbook configuration to see *THIS* ibeis database
        if use_config_file:
            update_wildbook_config(ibs, wildbook_tomcat_path, dryrun)

        # Check and push 'done' imagesets
        status_list = []
        for imgsetid in imgsetid_list:
            #Check for nones
            #status = _send(imgsetid, use_config_file=use_config_file, dryrun=dryrun)
            imageset_uuid = ibs.get_imageset_uuid(imgsetid)
            #url = submit_imgsetid_url_fmtstr.format(imageset_uuid=imageset_uuid)
            url = wildbook_base_url + '/ia'
            print('[_send] URL=%r' % (url, ))
            payload = {
                'resolver':
                {
                    'fromIAImageSet': str(imageset_uuid)
                }
            }
            if not dryrun:
                response = requests.post(url, json=payload)
                print('response = %r' % (response,))

            status = response.status_code == 200
            if set_shipped_flag and not dryrun:
                if status:
                    ibs.set_imageset_shipped_flags([imgsetid], [1])
                    if open_url_on_complete:
                        #status, response = submit_wildbook_url(url, payload, dryrun=dryrun)
                        view_occur_url = wildbook_base_url + '/occurrence.jsp?number=%s' % (imageset_uuid,)
                        _browser = ut.get_prefered_browser(PREFERED_BROWSER)
                        _browser.open_new_tab(view_occur_url)
                else:
                    ibs.set_imageset_shipped_flags([imgsetid], [0])
        return status_list


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs
        python -m ibeis.control.manual_wildbook_funcs --allexamples
        python -m ibeis.control.manual_wildbook_funcs --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
