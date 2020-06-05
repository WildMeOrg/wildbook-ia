# -*- coding: utf-8 -*-
"""
Manages local wildbook installations.

CommandLine:
    python -m utool.util_inspect check_module_usage --pat="wildbook_manager.py"

Utils:
    # TODO go to http://localhost:8080/wbia/createAssetStore.jsp
    tail -f ~/.config/wbia/tomcat/logs/catalina.out
    cat ~/.config/wbia/tomcat/logs/catalina.out
    python -m wbia shutdown_wildbook_server
    python -m wbia update_wildbook_install_config

"""
from __future__ import absolute_import, division, print_function
import utool as ut
import subprocess
import re
import time
import os
from os.path import dirname, join, basename, splitext

print, rrr, profile = ut.inject2(__name__)


# PREFERED_BROWSER = 'chrome'
# webbrowser._tryorder
PREFERED_BROWSER = None
if ut.get_computer_name() == 'hyrule':
    PREFERED_BROWSER = 'firefox'


# FIXME add as controller config
# ALLOW_SYSTEM_TOMCAT = ut.get_argflag('--allow-system-tomcat')


def get_tomcat_startup_tmpdir():
    dpath_list = [
        # os.environ.get('CATALINA_TMPDIR', None),
        ut.ensure_app_resource_dir('wbia', 'tomcat', 'wbia_startup_tmpdir'),
    ]
    tomcat_startup_dir = ut.search_candidate_paths(dpath_list, verbose=True)
    return tomcat_startup_dir


@ut.tracefunc_xml
def find_tomcat(verbose=ut.NOT_QUIET):
    r"""
    Searches likely places for tomcat to be installed

    Returns:
        str: tomcat_dpath

    Ignore:
        locate --regex "tomcat/webapps$"

    CommandLine:
        python -m wbia find_tomcat

    Example:
        >>> # SCRIPT
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> tomcat_dpath = find_tomcat()
        >>> result = ('tomcat_dpath = %s' % (str(tomcat_dpath),))
        >>> print(result)
    """
    # Tomcat folder must be named one of these and contain specific files
    fname_list = ['Tomcat', 'tomcat']
    # required_subpaths = ['webapps', 'bin', 'bin/catalina.sh']
    required_subpaths = ['webapps']

    # Places for local install of tomcat
    priority_paths = [
        # Number one preference is the CATALINA_HOME directory
        os.environ.get('CATALINA_HOME', None),
        # We put tomcat here if we can't find it
        ut.get_app_resource_dir('wbia', 'tomcat'),
    ]
    if ut.is_developer():
        # For my machine to use local catilina
        dpath_list = []
    else:
        # Places for system install of tomcat
        if ut.WIN32:
            dpath_list = ['C:/Program Files (x86)', 'C:/Program Files']
        elif ut.DARWIN:
            dpath_list = ['/Library']  # + dpath_list
        else:
            dpath_list = ['/var/lib', '/usr/share', '/opt', '/lib']
    return_path = ut.search_candidate_paths(
        dpath_list, fname_list, priority_paths, required_subpaths, verbose=verbose
    )
    tomcat_dpath = return_path
    print('tomcat_dpath = %r ' % (tomcat_dpath,))
    return tomcat_dpath


@ut.tracefunc_xml
def download_tomcat():
    """
    Put tomcat into a directory controlled by wbia

    CommandLine:
        # Reset
        python -c "import utool as ut; ut.delete(ut.unixjoin(ut.get_app_resource_dir('wbia'), 'tomcat'))"
    """
    print('Grabbing tomcat')
    # FIXME: need to make a stable link
    if ut.WIN32:
        tomcat_binary_url = 'http://mirrors.advancedhosters.com/apache/tomcat/tomcat-8/v8.0.36/bin/apache-tomcat-8.0.36-windows-x86.zip'
    else:
        tomcat_binary_url = 'http://mirrors.advancedhosters.com/apache/tomcat/tomcat-8/v8.0.36/bin/apache-tomcat-8.0.36.zip'
    zip_fpath = ut.grab_file_url(tomcat_binary_url, appname='wbia')
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
        python -m wbia find_installed_tomcat

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> check_unpacked = False
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
        import wbia

        # Check that webapps was unpacked
        wb_target = wbia.const.WILDBOOK_TARGET
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
        python -m purge_local_wildbook

        python -m wbia --tf purge_local_wildbook
        python -m wbia --tf find_or_download_tomcat

    Example:
        >>> # SCRIPT
        >>> from wbia.control.wildbook_manager import *  # NOQA
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
        python -m wbia find_java_jvm

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> jvm_fpath = find_java_jvm()
        >>> result = ('jvm_fpath = %r' % (jvm_fpath,))
        >>> print(result)
    """
    candidate_path_list = [
        os.environ.get('JAVA_HOME', None),
        '/usr/lib/jvm/java-7-openjdk-amd64',
    ]
    jvm_fpath = ut.search_candidate_paths(candidate_path_list, verbose=True)
    ut.assertpath(jvm_fpath, 'IBEIS cannot find Java Runtime Environment')
    return jvm_fpath


def find_or_download_wilbook_warfile(ensure=True, redownload=False):
    r"""
    scp jonc@pachy.cs.uic.edu:/var/lib/tomcat/webapps/wbia.war \
            ~/Downloads/pachy_wbia.war wget
    http://dev.wildme.org/wbia_data_dir/wbia.war
    """
    # war_url = 'http://dev.wildme.org/wbia_data_dir/wbia.war'
    war_url = 'http://springbreak.wildbook.org/tools/latest.war'
    war_fpath = ut.grab_file_url(
        war_url, appname='wbia', ensure=ensure, redownload=redownload, fname='wbia.war'
    )
    return war_fpath


def purge_local_wildbook():
    r"""
    Shuts down the server and then purges the server on disk

    CommandLine:
        python -m wbia purge_local_wildbook
        python -m wbia purge_local_wildbook --purge-war

    Example:
        >>> # SCRIPT
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> purge_local_wildbook()
    """
    try:
        shutdown_wildbook_server()
    except ImportError:
        pass
    ut.delete(ut.unixjoin(ut.get_app_resource_dir('wbia'), 'tomcat'))
    if ut.get_argflag('--purge-war'):
        war_fpath = find_or_download_wilbook_warfile(ensure=False)
        ut.delete(war_fpath)


def ensure_wb_mysql():
    r"""
    CommandLine:
        python -m wbia ensure_wb_mysql

    Example:
        >>> # SCRIPT
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> result = ensure_wb_mysql()
    """
    print('Execute the following code to install mysql')
    print(
        ut.codeblock(
            r"""
        # STARTBLOCK bash
        # Install
        sudo apt-get install mysql-server-5.6 -y
        sudo apt-get install mysql-common-5.6 -y
        sudo apt-get install mysql-client-5.6 -y

        mysql_config_editor set --login-path=local --host=localhost --user=root --password

        # Initialize
        mysql --login-path=local -e "create user 'wbiawb'@'localhost' identified by 'somepassword';"
        mysql --login-path=local -e "create database wbiawbtestdb;"
        mysql --login-path=local -e "grant all privileges on wbiawbtestdb.* to 'wbiawb'@'localhost';"

        # Reset
        mysql --login-path=local -e "drop database wbiawbtestdb"
        mysql --login-path=local -e "create database wbiawbtestdb"
        mysql --login-path=local -e "grant all privileges on wbiawbtestdb.* to 'wbiawb'@'localhost'"

        # Check if running
        mysqladmin --login-path=local status

        # mysql --login-path=local -e "status"
        # mysql -u root -proot status
        # mysql -u root -proot
        # ENDBLOCK bash
        """
        )
    )


def ensure_local_war(verbose=ut.NOT_QUIET):
    """
    Ensures tomcat has been unpacked and the war is localized

    CommandLine:
        wbia ensure_local_war

    Example:
        >>> # SCRIPT
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> result = ensure_local_war()
        >>> print(result)
    """
    # TODO: allow custom specified tomcat directory
    try:
        output = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
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
            'Please install java: http://www.java.com/en/download/'
        )

    tomcat_dpath = find_or_download_tomcat()
    assert tomcat_dpath is not None, 'Could not find tomcat'
    redownload = ut.get_argflag('--redownload-war')
    war_fpath = find_or_download_wilbook_warfile(redownload=redownload)
    war_fname = basename(war_fpath)

    # Move the war file to tomcat webapps if not there
    webapps_dpath = join(tomcat_dpath, 'webapps')
    deploy_war_fpath = join(webapps_dpath, war_fname)
    if not ut.checkpath(deploy_war_fpath, verbose=verbose):
        ut.copy(war_fpath, deploy_war_fpath)

    wb_target = splitext(war_fname)[0]
    return tomcat_dpath, webapps_dpath, wb_target


@ut.tracefunc_xml
def install_wildbook(verbose=ut.NOT_QUIET):
    """
    Script to setup wildbook on a unix based system
    (hopefully eventually this will generalize to win32)

    CommandLine:
        # Reset
        wbia purge_local_wildbook
        wbia ensure_wb_mysql
        wbia ensure_local_war
        # Setup
        wbia install_wildbook
        # wbia install_wildbook --nomysql
        # Startup
        wbia startup_wildbook_server --show

        Alternates:
            wbia install_wildbook --redownload-war
            wbia install_wildbook --assets
            wbia startup_wildbook_server --show

    Example:
        >>> # SCRIPT
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> verbose = True
        >>> result = install_wildbook()
        >>> print(result)
    """
    import requests

    # Ensure that the war file has been unpacked
    tomcat_dpath, webapps_dpath, wb_target = ensure_local_war()

    unpacked_war_dpath = join(webapps_dpath, wb_target)
    tomcat_startup_dir = get_tomcat_startup_tmpdir()
    fresh_install = not ut.checkpath(unpacked_war_dpath, verbose=verbose)
    if fresh_install:
        # Need to make sure you start catalina in the same directory otherwise
        # the derby databsae gets put in in cwd
        with ut.ChdirContext(tomcat_startup_dir):
            # Starting and stoping catalina should be sufficient to unpack the
            # war
            startup_fpath = join(tomcat_dpath, 'bin', 'startup.sh')
            # shutdown_fpath = join(tomcat_dpath, 'bin', 'shutdown.sh')
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
            ut.assertpath(
                unpacked_war_dpath,
                (
                    'Wildbook war might have not unpacked correctly.  This may '
                    'be ok. Try again. If it fails a second time, then there is a '
                    'problem.'
                ),
                verbose=True,
            )

            # Don't shutdown just yet. Need to create assets

    update_wildbook_install_config(webapps_dpath, unpacked_war_dpath)
    asset_flag_fpath = join(tomcat_startup_dir, 'made_assets.flag')

    # Pinging the server to create asset store
    # Ensureing that createAssetStore exists
    if not ut.checkpath(asset_flag_fpath):
        if not fresh_install:
            startup_wildbook_server()
        # web_url = startup_wildbook_server(verbose=False)
        print('Creating asset store')
        wb_url = 'http://localhost:8080/' + wb_target
        response = requests.get(wb_url + '/createAssetStore.jsp')
        if response is None or response.status_code != 200:
            print('There may be an error starting the server')
            # if response.status_code == 500:
            print(response.text)
            assert False, 'response error'
        else:
            print('Created asset store')
            # Create file signaling we did this
            ut.writeto(asset_flag_fpath, 'True')
        shutdown_wildbook_server(verbose=False)
        print('It is ok if the shutdown fails')
    elif fresh_install:
        shutdown_wildbook_server(verbose=False)

    # 127.0.0.1:8080/wildbook_data_dir/test.txt
    print('Wildbook is installed and waiting to be started')


@ut.tracefunc_xml
def update_wildbook_install_config(webapps_dpath, unpacked_war_dpath):
    """
    CommandLine:
        python -m wbia ensure_local_war
        python -m wbia update_wildbook_install_config
        python -m wbia update_wildbook_install_config --show

    Example:
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> import wbia
        >>> tomcat_dpath = find_installed_tomcat()
        >>> webapps_dpath = join(tomcat_dpath, 'webapps')
        >>> wb_target = wbia.const.WILDBOOK_TARGET
        >>> unpacked_war_dpath = join(webapps_dpath, wb_target)
        >>> locals_ = ut.exec_func_src(update_wildbook_install_config, globals())
        >>> #update_wildbook_install_config(webapps_dpath, unpacked_war_dpath)
        >>> ut.quit_if_noshow()
        >>> ut.vd(unpacked_war_dpath)
        >>> ut.editfile(locals_['permission_fpath'])
        >>> ut.editfile(locals_['jdoconfig_fpath'])
        >>> ut.editfile(locals_['asset_store_fpath'])
    """
    mysql_mode = not ut.get_argflag('--nomysql')

    # if ut.get_argflag('--vd'):
    #    ut.vd(unpacked_war_dpath)
    # find_installed_tomcat
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
        pattern = '^' + prefix + re.escape(line) + suffix
        match = re.search(pattern, permission_text, flags=re.MULTILINE | re.DOTALL)
        if match is None:
            continue
        newline = '<!--%s -->' % (line,)
        repl = ut.bref_field('prefix') + newline + ut.bref_field('suffix')
        new_permission_text = re.sub(
            pattern, repl, permission_text, flags=re.MULTILINE | re.DOTALL
        )
        assert new_permission_text != permission_text, 'text should have changed'
    if new_permission_text != permission_text:
        print('Need to write new permission texts')
        ut.writeto(permission_fpath, new_permission_text)
    else:
        print('Permission file seems to be ok')

    # Make sure we are using a non-process based database
    jdoconfig_fpath = join(
        unpacked_war_dpath, 'WEB-INF/classes/bundles/jdoconfig.properties'
    )
    print('Fixing backend database config')
    print('jdoconfig_fpath = %r' % (jdoconfig_fpath,))
    ut.assertpath(jdoconfig_fpath)
    jdoconfig_text = ut.readfrom(jdoconfig_fpath)
    # ut.vd(dirname(jdoconfig_fpath))
    # ut.editfile(jdoconfig_fpath)

    if mysql_mode:
        jdoconfig_text = ut.toggle_comment_lines(jdoconfig_text, 'mysql', False)
        jdoconfig_text = ut.toggle_comment_lines(jdoconfig_text, 'derby', 1)
        jdoconfig_text = ut.toggle_comment_lines(jdoconfig_text, 'sqlite', 1)
        mysql_user = 'wbiawb'
        mysql_passwd = 'somepassword'
        mysql_dbname = 'wbiawbtestdb'
        # Use mysql
        jdoconfig_text = re.sub(
            'datanucleus.ConnectionUserName = .*$',
            'datanucleus.ConnectionUserName = ' + mysql_user,
            jdoconfig_text,
            flags=re.MULTILINE,
        )
        jdoconfig_text = re.sub(
            'datanucleus.ConnectionPassword = .*$',
            'datanucleus.ConnectionPassword = ' + mysql_passwd,
            jdoconfig_text,
            flags=re.MULTILINE,
        )
        jdoconfig_text = re.sub(
            'datanucleus.ConnectionURL *= *jdbc:mysql:.*$',
            'datanucleus.ConnectionURL = jdbc:mysql://localhost:3306/' + mysql_dbname,
            jdoconfig_text,
            flags=re.MULTILINE,
        )
        jdoconfig_text = re.sub(
            '^.*jdbc:mysql://localhost:3306/shepherd.*$',
            '',
            jdoconfig_text,
            flags=re.MULTILINE,
        )
    else:
        # Use SQLIIte
        jdoconfig_text = ut.toggle_comment_lines(jdoconfig_text, 'derby', 1)
        jdoconfig_text = ut.toggle_comment_lines(jdoconfig_text, 'mysql', 1)
        jdoconfig_text = ut.toggle_comment_lines(jdoconfig_text, 'sqlite', False)
    ut.writeto(jdoconfig_fpath, jdoconfig_text)

    # Need to make sure wildbook can store information in a reasonalbe place
    # tomcat_data_dir = join(tomcat_startup_dir, 'webapps', 'wildbook_data_dir')
    tomcat_data_dir = join(webapps_dpath, 'wildbook_data_dir')
    ut.ensuredir(tomcat_data_dir)
    ut.writeto(join(tomcat_data_dir, 'test.txt'), 'A hosted test file')
    asset_store_fpath = join(unpacked_war_dpath, 'createAssetStore.jsp')
    asset_store_text = ut.read_from(asset_store_fpath)
    # data_path_pat = ut.named_field('data_path', 'new File(".*?").toPath')
    new_line = (
        'LocalAssetStore as = new LocalAssetStore("example Local AssetStore", new File("%s").toPath(), "%s", true);'
        % (tomcat_data_dir, 'http://localhost:8080/' + basename(tomcat_data_dir))
    )
    # HACKY
    asset_store_text2 = re.sub(
        '^LocalAssetStore as = .*$', new_line, asset_store_text, flags=re.MULTILINE
    )
    ut.writeto(asset_store_fpath, asset_store_text2)
    # ut.editfile(asset_store_fpath)


@ut.tracefunc_xml
def update_wildbook_ia_config(ibs, wildbook_tomcat_path, dryrun=False):
    """
    #if use_config_file and wildbook_tomcat_path:
    #    # Update the Wildbook configuration to see *THIS* wbia database
    #    with lockfile.LockFile(lock_fpath):
    #        update_wildbook_ia_config(ibs, wildbook_tomcat_path, dryrun)
    """
    wildbook_properteis_dpath = join(wildbook_tomcat_path, 'WEB-INF/classes/bundles/')
    print(
        '[ibs.update_wildbook_ia_config()] Wildbook properties=%r'
        % (wildbook_properteis_dpath,)
    )
    # The src file is non-standard. It should be remove here as well
    wildbook_config_fpath_dst = join(
        wildbook_properteis_dpath, 'commonConfiguration.properties'
    )
    ut.assert_exists(wildbook_properteis_dpath)
    # for come reason the .default file is not there, that should be ok though
    orig_content = ut.read_from(wildbook_config_fpath_dst)
    content = orig_content
    # Make sure wildbook knows where to find us
    if False:
        # Old way of telling WB where to find IA
        content = re.sub(
            'IBEIS_DB_path = .*', 'IBEIS_DB_path = ' + ibs.get_db_core_path(), content
        )
        content = re.sub(
            'IBEIS_image_path = .*', 'IBEIS_image_path = ' + ibs.get_imgdir(), content
        )

    web_port = ibs.get_web_port_via_scan()
    if web_port is None:
        raise ValueError('IA web server is not running on any expected port')
    ia_hostport = 'http://localhost:%s' % (web_port,)
    ia_rest_prefix = ut.named_field('prefix', 'IBEISIARestUrl.*')
    host_port = ut.named_field('host_port', 'http://.*?:[0-9]+')
    content = re.sub(
        ia_rest_prefix + host_port, ut.bref_field('prefix') + ia_hostport, content
    )

    # Write to the configuration if it is different
    if orig_content != content:
        need_sudo = not ut.is_file_writable(wildbook_config_fpath_dst)
        if need_sudo:
            quoted_content = '"%s"' % (content,)
            print('Attempting to gain sudo access to update wildbook config')
            command = [
                'sudo',
                'sh',
                '-c',
                "'",
                'echo',
                quoted_content,
                '>',
                wildbook_config_fpath_dst,
                "'",
            ]
            # ut.cmd(command, sudo=True)
            command = ' '.join(command)
            if not dryrun:
                os.system(command)
        else:
            ut.write_to(wildbook_config_fpath_dst, content)


@ut.tracefunc_xml
def startup_wildbook_server(verbose=ut.NOT_QUIET):
    r"""
    Args:
        verbose (bool):  verbosity flag(default = True)

    CommandLine:
        python -m wbia startup_wildbook_server
        python -m wbia startup_wildbook_server  --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> verbose = True
        >>> wb_url = startup_wildbook_server()
        >>> ut.quit_if_noshow()
        >>> ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(wb_url)
    """
    # TODO: allow custom specified tomcat directory
    import wbia

    tomcat_dpath = find_installed_tomcat()

    with ut.ChdirContext(get_tomcat_startup_tmpdir()):
        startup_fpath = join(tomcat_dpath, 'bin', 'startup.sh')
        ut.cmd(ut.quote_single_command(startup_fpath))
        time.sleep(1)
    wb_url = 'http://localhost:8080/' + wbia.const.WILDBOOK_TARGET
    # TODO go to http://localhost:8080/wbia/createAssetStore.jsp
    return wb_url


def shutdown_wildbook_server(verbose=ut.NOT_QUIET):
    r"""

    Args:
        verbose (bool):  verbosity flag(default = True)

    Ignore:
        tail -f ~/.config/wbia/tomcat/logs/catalina.out

    CommandLine:
        python -m wbia shutdown_wildbook_server

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> verbose = True
        >>> wb_url = shutdown_wildbook_server()
        >>> ut.quit_if_noshow()
        >>> ut.get_prefered_browser(PREFERED_BROWSER).open_new_tab(wb_url)
    """
    # TODO: allow custom specified tomcat directory
    tomcat_dpath = find_installed_tomcat(check_unpacked=False, strict=False)
    # TODO: allow custom specified tomcat directory
    # tomcat_dpath = find_installed_tomcat(check_unpacked=False)
    # catalina_out_fpath = join(tomcat_dpath, 'logs', 'catalina.out')
    if tomcat_dpath is not None:
        with ut.ChdirContext(get_tomcat_startup_tmpdir()):
            shutdown_fpath = join(tomcat_dpath, 'bin', 'shutdown.sh')
            # ut.cmd(shutdown_fpath)
            ut.cmd(ut.quote_single_command(shutdown_fpath))
            time.sleep(0.5)


def monitor_wildbook_logs(verbose=ut.NOT_QUIET):
    r"""
    Args:
        verbose (bool):  verbosity flag(default = True)

    CommandLine:
        python -m wbia monitor_wildbook_logs  --show

    Example:
        >>> # SCRIPT
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> monitor_wildbook_logs()
    """
    # TODO: allow custom specified tomcat directory
    import wbia

    tomcat_dpath = find_installed_tomcat()

    with ut.ChdirContext(get_tomcat_startup_tmpdir()):
        startup_fpath = join(tomcat_dpath, 'bin', 'startup.sh')
        ut.cmd(ut.quote_single_command(startup_fpath))
        time.sleep(1)
    wb_url = 'http://localhost:8080/' + wbia.const.WILDBOOK_TARGET
    return wb_url


def tryout_wildbook_login():
    r"""
    Helper function to test wildbook login automagically

    Returns:
        tuple: (wb_target, tomcat_dpath)

    CommandLine:
        python -m wbia tryout_wildbook_login

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.wildbook_manager import *  # NOQA
        >>> tryout_wildbook_login()
    """
    # Use selenimum to login to wildbook
    import wbia

    manaul_login = False
    wb_target = wbia.const.WILDBOOK_TARGET
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
            # login_button = driver.find_element_by_partial_link_text('Log in')
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


def get_wildbook_tomcat_path(ibs, tomcat_dpath=None, wb_target=None):
    DEFAULT_TOMCAT_PATH = find_installed_tomcat()
    tomcat_dpath = DEFAULT_TOMCAT_PATH if tomcat_dpath is None else tomcat_dpath
    wb_target = ibs.const.WILDBOOK_TARGET if wb_target is None else wb_target
    wildbook_tomcat_path = join(tomcat_dpath, 'webapps', wb_target)
    return wildbook_tomcat_path


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.control.wildbook_manager
        python -m wbia.control.wildbook_manager --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
