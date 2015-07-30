# -*- coding: utf-8 -*-
"""
CommandLine;
    # Reset IBEIS database (can skip if done)
    python -m ibeis.tests.reset_testdbs --reset_mtest

CommandLine;
    # Reset Wildbook database
    python -m ibeis.control.manual_wildbook_funcs --exec-reset_local_wildbook

    # Install Wildbook
    python -m ibeis.control.manual_wildbook_funcs --exec-install_wildbook

    # Startup Wildbook
    python -m ibeis.control.manual_wildbook_funcs --exec-startup_wildbook_server

    # Login to wildbook (can skip)
    python -m ibeis.control.manual_wildbook_funcs --exec-test_wildbook_login

    # Ship Encounters to wildbook
    python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_eid_list

    # Change annotations names to a single name
    python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_annot_name_changes:1

    # Change annotations names back to normal
    python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_annot_name_changes:2

"""
from __future__ import absolute_import, division, print_function
import six  # NOQA
import re
import utool as ut
import lockfile
import os
import requests
from os.path import join
from ibeis.control import controller_inject
from ibeis.control.controller_inject import make_ibs_register_decorator
print, rrr, profile = ut.inject2(__name__, '[manual_wildbook]')

CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)

register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)

#PREFERED_BROWSER = 'chrome'
PREFERED_BROWSER = None
if ut.get_computer_name() == 'hyrule':
    PREFERED_BROWSER = 'firefox'
#webbrowser._tryorder


def get_tomcat_startup_tmpdir():
    dpath_list = [
        #os.environ.get('CATALINA_TMPDIR', None),
        ut.ensure_app_resource_dir('ibeis', 'tomcat', 'ibeis_startup_tmpdir'),
    ]
    return_path = ut.search_candidate_paths(dpath_list, verbose=True)
    return return_path


def find_tomcat(verbose=ut.NOT_QUIET):
    r"""
    Returns:
        str: tomcat_dpath

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --test-find_tomcat

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> tomcat_dpath = find_tomcat()
        >>> result = ('tomcat_dpath = %s' % (str(tomcat_dpath),))
        >>> print(result)
    """
    import utool as ut
    import os
    fname_list = ['Tomcat', 'tomcat']
    # Places for system install of tomcat
    if ut.WIN32:
        dpath_list = ['C:/Program Files (x86)', 'C:/Program Files']
    else:
        dpath_list = ['/var/lib', '/usr/share', '/opt', '/lib']
    if ut.DARWIN:
        dpath_list = ['/Library'] + dpath_list

    priority_paths = [
        # Numberone preference is the CATALINA_HOME directory
        os.environ.get('CATALINA_HOME', None),
        # We put tomcat here if we can't find it
        ut.get_app_resource_dir('ibeis', 'tomcat')
    ]

    required_subpaths = [
        'webapps',
        'bin',
        'bin/catalina.sh',
    ]

    return_path = ut.search_candidate_paths(dpath_list, fname_list, priority_paths, required_subpaths, verbose=verbose)
    tomcat_dpath = return_path
    print('tomcat_dpath = %r ' % (tomcat_dpath,))
    return tomcat_dpath


def download_tomcat():
    """
    Put tomcat into a directory controlled by ibeis

    CommandLine:
        # Reset
        python -c "import utool as ut; ut.delete(ut.unixjoin(ut.get_app_resource_dir('ibeis'), 'tomcat'))"
    """
    from os.path import splitext, dirname
    if ut.WIN32:
        tomcat_binary_url = 'http://mirrors.advancedhosters.com/apache/tomcat/tomcat-8/v8.0.24/bin/apache-tomcat-8.0.24-windows-x86.zip'
    else:
        tomcat_binary_url = 'http://mirrors.advancedhosters.com/apache/tomcat/tomcat-8/v8.0.24/bin/apache-tomcat-8.0.24.zip'
    zip_fpath = ut.grab_file_url(tomcat_binary_url, appname='ibeis')
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


def find_installed_tomcat(check_unpacked=True):
    """
    Asserts that tomcat was properly installed
    """
    tomcat_dpath = find_tomcat()
    if tomcat_dpath is None:
        raise ImportError('Cannot find tomcat')
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
        python -m ibeis.control.manual_wildbook_funcs --test-reset_local_wildbook
        python -m ibeis.control.manual_wildbook_funcs --test-find_or_download_tomcat

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


def find_or_download_wilbook_warfile():
    #scp jonc@pachy.cs.uic.edu:/var/lib/tomcat/webapps/ibeis.war ~/Downloads/pachy_ibeis.war
    #wget http://dev.wildme.org/ibeis_data_dir/ibeis.war

    war_url = 'http://dev.wildme.org/ibeis_data_dir/ibeis.war'
    war_fpath = ut.grab_file_url(war_url, appname='ibeis')
    return war_fpath


def reset_local_wildbook():
    r"""
    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --test-reset_local_wildbook

    Example:
        >>> # SCRIPT
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> reset_local_wildbook()
    """
    import utool as ut
    try:
        shutdown_wildbook_server()
    except ImportError:
        pass
    ut.delete(ut.unixjoin(ut.get_app_resource_dir('ibeis'), 'tomcat'))


def install_wildbook(verbose=ut.NOT_QUIET):
    """
    Script to setup wildbook on a unix based system
    (hopefully eventually this will generalize to win32)

    CommandLine:
        # Reset
        python -m ibeis.control.manual_wildbook_funcs --test-reset_local_wildbook
        # Setup
        python -m ibeis.control.manual_wildbook_funcs --test-install_wildbook
        # Startup
        python -m ibeis.control.manual_wildbook_funcs --test-startup_wildbook_server --show --exec-mode


    Example:
        >>> # SCRIPT
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> verbose = True
        >>> result = install_wildbook()
        >>> print(result)
    """
    # TODO: allow custom specified tomcat directory
    from os.path import basename, splitext, join
    import time
    import re
    import subprocess
    try:
        output = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
        java_version = output.split('\n')[0].replace('java version ', '').replace('"', '')
        print('java_version = %r' % (java_version,))
        if not java_version.startswith('1.7'):
            print('Warning wildbook is only supported for java 1.7')
    except OSError:
        output = None
    if output is None:
        raise ImportError('Cannot find java on this machine. Please install java: http://www.java.com/en/download/')

    tomcat_dpath = find_or_download_tomcat()
    assert tomcat_dpath is not None, 'Could not find tomcat'
    war_fpath = find_or_download_wilbook_warfile()
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
    if not ut.checkpath(unpacked_war_dpath, verbose=verbose):
        # Need to make sure you start catalina in the same directory otherwise
        # the derby databsae gets put in in cwd
        with ut.ChdirContext(get_tomcat_startup_tmpdir()):
            # Starting and stoping catalina should be sufficient to unpack the war
            startup_fpath  = join(tomcat_dpath, 'bin', 'startup.sh')
            shutdown_fpath = join(tomcat_dpath, 'bin', 'shutdown.sh')
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
            import requests
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
            ut.cmd(ut.quote_single_command(shutdown_fpath))
            print('It is ok if the shutdown.sh fails')
            time.sleep(.5)

    # Make sure permissions are correctly set in wildbook
    permission_fpath = join(unpacked_war_dpath, 'WEB-INF/web.xml')
    ut.assertpath(permission_fpath)
    permission_text = ut.readfrom(permission_fpath)
    lines_to_remove = [
        '/EncounterSetMarkedIndividual = authc, roles[admin]'
    ]
    new_permission_text = permission_text[:]
    for line in lines_to_remove:
        re.search(re.escape(line), permission_text)
        pattern = '^' + ut.named_field('prefix', '\\s*') + re.escape(line) + ut.named_field('suffix', '\\s*\n')
        match = re.search(pattern, permission_text, flags=re.MULTILINE | re.DOTALL)
        if match is None:
            continue
        repl = ut.bref_field('prefix') + '<!--' + line + ' -->' + ut.bref_field('suffix')
        new_permission_text = re.sub(pattern, repl, permission_text, flags=re.MULTILINE | re.DOTALL)
        assert new_permission_text != permission_text, 'text should have changed'
    if new_permission_text != permission_text:
        print('Need to write new permission texts')
        ut.writeto(permission_fpath, new_permission_text)
    else:
        print('Permission file seems to be ok')

    print('Wildbook is installed and waiting to be started')


def startup_wildbook_server(verbose=ut.NOT_QUIET):
    r"""
    Args:
        verbose (bool):  verbosity flag(default = True)

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --test-startup_wildbook_server --show --exec-mode

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> verbose = True
        >>> wb_url = startup_wildbook_server()
        >>> ut.quit_if_noshow()
        >>> ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(wb_url)
    """
    # TODO: allow custom specified tomcat directory
    from os.path import join
    import time
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


def shutdown_wildbook_server(verbose=ut.NOT_QUIET):
    r"""
    Args:
        verbose (bool):  verbosity flag(default = True)

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --test-shutdown_wildbook_server --exec-mode

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> verbose = True
        >>> wb_url = shutdown_wildbook_server()
        >>> ut.quit_if_noshow()
        >>> ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(wb_url)
    """
    # TODO: allow custom specified tomcat directory
    from os.path import join
    import time
    tomcat_dpath = find_installed_tomcat(check_unpacked=False)

    # Ensure environment variables
    #os.environ['JAVA_HOME'] = find_java_jvm()
    #os.environ['TOMCAT_HOME'] = tomcat_dpath
    #os.environ['CATALINA_HOME'] = tomcat_dpath

    with ut.ChdirContext(get_tomcat_startup_tmpdir()):
        shutdown_fpath = join(tomcat_dpath, 'bin', 'shutdown.sh')
        #ut.cmd(shutdown_fpath)
        ut.cmd(ut.quote_single_command(shutdown_fpath))
        time.sleep(.5)


def test_wildbook_login():
    r"""
    Returns:
        tuple: (wb_target, tomcat_dpath)

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --exec-test_wildbook_login

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
        driver = ut.grab_selenium_driver(PREFERED_BROWSER)
        driver.get(wb_url)
        login_button = driver.find_element_by_partial_link_text('Log in')
        login_button.click()
        # Find login elements
        username_field = driver.find_element_by_name('username')
        password_field = driver.find_element_by_name('password')
        submit_login_button = driver.find_element_by_name('submit')
        rememberMe_button = driver.find_element_by_name('rememberMe')
        # Execute elements
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
    #    tomcat_dpath = join(os.environ['CODE_DIR'], 'Wildbook/tmp/apache-tomcat-8.0.24')
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
    ut.assert_exists(wildbook_tomcat_path, 'Wildbook is not installed on this machine', info=True)
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


def submit_wildbook_url(url, payload=None, browse_on_error=True, dryrun=False, timeout=2):
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
            ut.printex(ex, 'Could not connect to Wildbook server at url=%r' % url)
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


def update_wildbook_config(ibs, wildbook_tomcat_path, dryrun=False):
    wildbook_properteis_dpath = join(wildbook_tomcat_path, 'WEB-INF/classes/bundles/')
    print('[ibs.wildbook_signal_eid_list()] Wildbook properties=%r' % (wildbook_properteis_dpath, ))
    # The src file is non-standard. It should be remove here as well
    wildbook_config_fpath_dst = join(wildbook_properteis_dpath, 'commonConfiguration.properties')
    ut.assert_exists(wildbook_properteis_dpath)
    # for come reason the .default file is not there, that should be ok though
    orig_content = ut.read_from(wildbook_config_fpath_dst)
    content = orig_content
    content = re.sub('IBEIS_DB_path = .*', 'IBEIS_DB_path = ' + ibs.get_db_core_path(), content)
    content = re.sub('IBEIS_image_path = .*', 'IBEIS_image_path = ' + ibs.get_imgdir(), content)

    # Write to the configuration if it is different
    if orig_content != content:
        need_sudo = not ut.is_file_writable(wildbook_config_fpath_dst)
        if need_sudo:
            quoted_content = '"%s"' % (content, )
            print('[ibs.wildbook_signal_eid_list()] To update the Wildbook configuration, we need sudo privaleges')
            command = ['sudo', 'sh', '-c', '\'', 'echo', quoted_content, '>', wildbook_config_fpath_dst, '\'']
            # ut.cmd(command, sudo=True)
            command = ' '.join(command)
            if not dryrun:
                os.system(command)
        else:
            ut.write_to(wildbook_config_fpath_dst, content)


# @default_decorator
def export_to_wildbook(ibs):
    """
        Exports identified chips to wildbook

        Legacy:
            import ibeis.dbio.export_wb as wb
            print('[ibs] exporting to wildbook')
            eid_list = ibs.get_valid_eids()
            addr = 'http://127.0.0.1:8080/wildbook-4.1.0-RELEASE'
            #addr = 'http://tomcat:tomcat123@127.0.0.1:8080/wildbook-5.0.0-EXPERIMENTAL'
            ibs._send_wildbook_request(addr)
            wb.export_ibeis_to_wildbook(ibs, eid_list)

            # compute encounters
            # get encounters by id
            # get ANNOTATIONs by encounter id
            # submit requests to wildbook
            return None
    """
    raise NotImplementedError()


@register_ibs_method
@register_api('/api/core/wildbook_signal_annot_name_changes/', methods=['PUT'])
def wildbook_signal_annot_name_changes(ibs, aid_list=None, tomcat_dpath=None, wb_target=None, dryrun=False):
    r"""
    Args:
        aid_list (int):  list of annotation ids(default = None)
        tomcat_dpath (None): (default = None)
        wb_target (None): (default = None)
        dryrun (bool): (default = False)

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_annot_name_changes:0 --dryrun
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_annot_name_changes:1 --dryrun
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_annot_name_changes:1
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_annot_name_changes:2

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
    print('[ibs.wildbook_signal_eid_list()] signaling any annotation name changes to wildbook')

    wildbook_base_url, wildbook_tomcat_path = ibs.get_wildbook_info(tomcat_dpath, wb_target)
    url_command = 'EncounterSetMarkedIndividual'
    BASIC_AUTH = False
    if BASIC_AUTH:
        #url_command += '=authcBasicWildbook'
        username = 'tomcat'
        password = 'tomcat123'
        wildbook_base_url = 'http://' + username + ':' + password + '@' + wildbook_base_url.replace('http://', '')
    url_args_fmtstr = '&'.join([
        'encounterID={annot_uuid!s}',
        'individualID={name_text!s}',
    ])
    submit_namchange_url_fmtstr = wildbook_base_url + '/' + url_command + '?' + url_args_fmtstr

    if aid_list is None:
        aid_list = ibs.get_valid_aids(is_known=True)
    # Build URLs to submit
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    annot_name_text_list = ibs.get_annot_name_texts(aid_list)
    submit_url_list = [
        submit_namchange_url_fmtstr.format(annot_uuid=str(annot_uuid), name_text=str(name_text))
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
            # encounter in this message is a wb-encounter not our ia-encounter
            #if ut.VERBOSE:
            print(response_json['message'])
            message_list.append(str(response_json['message']))
        except Exception as ex:
            print(ut.indentjoin(message_list))
            ut.printex(ex, 'Failed getting json from responce. This probably means there is an authentication issue')
            raise
        assert response_json['success']
    print(ut.indentjoin(message_list))
    return status_list


@register_ibs_method
@register_api('/api/core/wildbook_signal_eid_list/', methods=['PUT'])
def wildbook_signal_eid_list(ibs, eid_list=None, set_shipped_flag=True, open_url_on_complete=True, tomcat_dpath=None, wb_target=None, dryrun=False):
    """
    Exports specified encounters to wildbook. This is a synchronous call.

    Args:
        eid_list (list): (default = None)
        set_shipped_flag (bool): (default = True)
        open_url_on_complete (bool): (default = True)

    RESTful:
        Method: PUT
        URL:    /api/core/wildbook_signal_eid_list/

    Ignore:
        cd $CODE_DIR/Wildbook/tmp

        # Reset IBEIS database
        python -m ibeis.tests.reset_testdbs --reset_mtest

        # Reset Wildbook database
        python -m ibeis.control.manual_wildbook_funcs --exec-reset_local_wildbook

        # Install Wildbook
        python -m ibeis.control.manual_wildbook_funcs --test-install_wildbook

        # Startup Wildbook
        python -m ibeis.control.manual_wildbook_funcs --test-startup_wildbook_server

        # Login to wildbook
        python -m ibeis.control.manual_wildbook_funcs --exec-test_wildbook_login

        # Ship Encounters to wildbook
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_eid_list

        # Change annotations names to a single name
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_annot_name_changes:1

        # Change annotations names back to normal
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_annot_name_changes:2

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_eid_list
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_eid_list --dryrun
        python -m ibeis.control.manual_wildbook_funcs --test-wildbook_signal_eid_list --break

    SeeAlso:
        ~/local/build_scripts/init_wildbook.sh

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> dryrun = ut.get_argflag('--dryrun')
        >>> wb_target, tomcat_dpath = testdata_wildbook_server()
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> #gid_list = ibs.get_valid_gids()[0:10]
        >>> gid_list = ibs.get_valid_gids()[3:5]
        >>> new_eid = ibs.create_new_encounter_from_images(gid_list)  # NOQA
        >>> print('new encounter uuid = %r' % (ibs.get_encounter_uuid(new_eid),))
        >>> print('new encounter text = %r' % (ibs.get_encounter_text(new_eid),))
        >>> eid_list = [new_eid]
        >>> ibs.set_encounter_processed_flags([new_eid], [1])
        >>> gid_list = ibs.get_encounter_gids(new_eid)
        >>> ibs.set_image_reviewed(gid_list, [1] * len(gid_list))
        >>> set_shipped_flag = True
        >>> open_url_on_complete = True
        >>> result = ibs.wildbook_signal_eid_list(eid_list, set_shipped_flag, open_url_on_complete, tomcat_dpath, wb_target, dryrun)
        >>> # cleanup
        >>> #ibs.delete_encounters(new_eid)
        >>> print(result)
    """

    def _send(eid, use_config_file=False, dryrun=dryrun):
        encounter_uuid = ibs.get_encounter_uuid(eid)
        url = submit_eid_url_fmtstr.format(encounter_uuid=encounter_uuid)
        print('[_send] URL=%r' % (url, ))
        smart_xml_fname = ibs.get_encounter_smart_xml_fnames([eid])[0]
        smart_waypoint_id = ibs.get_encounter_smart_waypoint_ids([eid])[0]
        if smart_xml_fname is not None and smart_waypoint_id is not None:
            # Send smart data if availabel
            print(smart_xml_fname, smart_waypoint_id)
            smart_xml_fpath = join(ibs.get_smart_patrol_dir(), smart_xml_fname)
            smart_xml_content_list = open(smart_xml_fpath).readlines()
            print('[_send] Sending with SMART payload - patrol: %r (%d lines) waypoint_id: %r' %
                  (smart_xml_fpath, len(smart_xml_content_list), smart_waypoint_id))
            smart_xml_content = ''.join(smart_xml_content_list)
            payload = {
                'smart_xml_content': smart_xml_content,
                'smart_waypoint_id': smart_waypoint_id,
            }
            if not use_config_file:
                payload.update({
                    'IBEIS_DB_path'    : ibs.get_db_core_path(),
                    'IBEIS_image_path' : ibs.get_imgdir(),
                })
        else:
            payload = None
        status, response = submit_wildbook_url(url, payload, dryrun=dryrun)
        return status

    def _complete(eid):
        encounter_uuid = ibs.get_encounter_uuid(eid)
        complete_url_ = complete_url_fmtstr.format(encounter_uuid=encounter_uuid)
        print('[_complete] URL=%r' % (complete_url_, ))
        if open_url_on_complete and not dryrun:
            ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(complete_url_)

    if eid_list is None:
        eid_list = ibs.get_valid_eids()
    # Check to make sure encounters are ok:
    for eid in eid_list:
        # First, check if encounter can be pushed
        aid_list = ibs.get_encounter_aids(eid)
        assert len(aid_list) > 0, 'Encounter eid=%r cannot be shipped becuase there are no annotations' % (eid,)
        unknown_flags = ibs.is_aid_unknown(aid_list)
        unnamed_aid_list = ut.list_compress(aid_list, unknown_flags)
        assert len(unnamed_aid_list) == 0, 'Encounter eid=%r cannot be shipped becuase annotation(s) %r are not named' % (eid, unnamed_aid_list, )

    # Configuration
    use_config_file = True
    wildbook_base_url, wildbook_tomcat_path = ibs.get_wildbook_info(tomcat_dpath, wb_target)
    submit_eid_url_fmtstr   = wildbook_base_url + '/OccurrenceCreateIBEIS?ibeis_encounter_id={encounter_uuid!s}'
    complete_url_fmtstr = wildbook_base_url + '/occurrence.jsp?number={encounter_uuid!s}'
    # Call Wildbook url to signal update
    print('[ibs.wildbook_signal_eid_list()] shipping eid_list = %r to wildbook' % (eid_list, ))

    # With a lock file, modify the configuration with the new settings
    lock_fpath = join(ibs.get_ibeis_resource_dir(), 'wildbook.lock')
    with lockfile.LockFile(lock_fpath):
        # Update the Wildbook configuration to see *THIS* ibeis database
        if use_config_file:
            update_wildbook_config(ibs, wildbook_tomcat_path, dryrun)

        # Check and push 'done' encounters
        status_list = []
        for eid in eid_list:
            #Check for nones
            status = _send(eid, use_config_file=use_config_file, dryrun=dryrun)
            status_list.append(status)
            if set_shipped_flag and not dryrun:
                if status:
                    ibs.set_encounter_shipped_flags([eid], [1])
                    _complete(eid)
                else:
                    ibs.set_encounter_shipped_flags([eid], [0])
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
