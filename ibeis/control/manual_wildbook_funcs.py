# -*- coding: utf-8 -*-
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
#PREFERED_BROWSER = 'firefox'
PREFERED_BROWSER = None
#webbrowser._tryorder


def hyrule_reset_wildbook():
    r"""
    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --exec-hyrule_reset_wildbook

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> result = hyrule_reset_wildbook()
        >>> print(result)
    """

    presetup_part = """
    scp jonc@pachy.cs.uic.edu:/var/lib/tomcat/webapps/ibeis.war ~/Downloads/pachy_ibeis.war
    # Slightly less volitile location
    rm ~/Downloads/ibeis.war
    wget  -o ~/Downloads/ibeis.war http://dev.wildme.org/ibeis_data_dir/ibeis.war
    wget http://dev.wildme.org/ibeis_data_dir/ibeis.war
    """
    assert ut.is_developer()
    delete_part = ut.codeblock(
        """
        #!/bin/bash

        # --- FRESHSTART ---
        # Make sure that tomcat vars are set
        export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64

        export WILDBOOK_TESTDIR=$CODE_DIR/Wildbook/tmp
        export TOMCAT_DIR=$WILDBOOK_TESTDIR/apache-tomcat-8.0.24
        export TOMCAT_HOME=$TOMCAT_DIR
        export CATALINA_HOME=$TOMCAT_DIR

        export WB_TARGET=ibeis

        # ensure everything is shutdown before we start
        $CATALINA_HOME/bin/shutdown.sh

        # Clean up old tomcat
        rm -rf $WILDBOOK_TESTDIR/apache-tomcat-8.0.24
        # REMOVE EVERYTHING.
        rm -rf $WILDBOOK_TESTDIR
        mkdir $WILDBOOK_TESTDIR
        cd $WILDBOOK_TESTDIR
        """)

    ignore_part = """
    cd $CODE_DIR/Wildbook
    """
    ignore_part

    create_part = ut.codeblock(
        """
        #!/bin/bash
        cd $WILDBOOK_TESTDIR
        unzip -q $WILDBOOK_TESTDIR/../apache-tomcat-8.0.24.zip -d $WILDBOOK_TESTDIR

        # make catalina runnable
        chmod +x $CATALINA_HOME/bin/catalina.sh
        chmod +x $CATALINA_HOME/bin/startup.sh
        chmod +x $CATALINA_HOME/bin/shutdown.sh

        # Install a wildbook .war file into tomcat webapps
        # assumes we've already downloaded the war file
        cp ~/Downloads/$WB_TARGET.war $CATALINA_HOME/webapps/

        # RUN TOMCAT SERVER (WE MUST BE IN THE TESTDIR ON STARTUP)
        $CATALINA_HOME/bin/startup.sh
        sleep .5
        $CATALINA_HOME/bin/shutdown.sh
        sleep .5
        sed -i 's/\/EncounterSetMarkedIndividual = authc, roles\[admin\]/<!--\/EncounterSetMarkedIndividual = authc, roles\[admin\]-->/' $TOMCAT_DIR/webapps/$WB_TARGET/WEB-INF/web.xml
        sleep .5
        $CATALINA_HOME/bin/startup.sh

        # google-chrome --new-window http://localhost:8080/$WB_TARGET
        # firefox http://localhost:8080/$WB_TARGET
        """
    )

    fix_settings = """
    # ensure everything is shutdown before we start
    $CATALINA_HOME/bin/shutdown.sh

    # Fix authentication
    gvim $TOMCAT_DIR/webapps/$WB_TARGET/WEB-INF/web.xml

    sed 's/\/EncounterSetMarkedIndividual = authc, roles\[admin\]/foo/' $TOMCAT_DIR/webapps/$WB_TARGET/WEB-INF/web.xml
    sed -i 's/^\(^ *\)\/EncounterSetMarkedIndividual = authc, roles\[admin\]/\1<!--\/EncounterSetMarkedIndividual = authc, roles\[admin\]-->/' $TOMCAT_DIR/webapps/$WB_TARGET/WEB-INF/web.xml  | grep EncounterSetMarkedIndividual
    sed 's/^\(^ *\)\/EncounterSetMarkedIndividual = authc, roles\[admin\]/\1<!--\/EncounterSetMarkedIndividual = authc, roles\[admin\]-->/' $TOMCAT_DIR/webapps/$WB_TARGET/WEB-INF/web.xml  | grep EncounterSetMarkedIndividual

    # Replace
    '/EncounterSetMarkedIndividual = authc, roles[admin]'
    # WITH
    '<!--/EncounterSetMarkedIndividual = authc, roles[admin]-->'
    """
    fix_settings

    bash_script = delete_part + '\n' + create_part
    if True:
        print('+-------')
        print('Executing bash script: ')
        ut.print_code(bash_script, 'bash')
        #print(bash_script)
        print('L______')
        os.system(bash_script)
    else:
        def write_script(fpath_, text):
            ut.writeto(fpath_, text)
            ut.chmod_add_executable(fpath_)

        fpath = 'hyrule_reset_wildbook.sh'
        fpath1 = 'hyrule_delete_wildbook.sh'
        fpath2 = 'hyrule_create_wildbook.sh'
        write_script(fpath, bash_script)
        write_script(fpath1, delete_part)
        write_script(fpath2, create_part)
        return fpath


def hyrule_wildbook_login():
    r"""
    Returns:
        tuple: (wb_target, tomcat_dpath)

    CommandLine:
        python -m ibeis.control.manual_wildbook_funcs --exec-hyrule_wildbook_login

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> hyrule_wildbook_login()
    """
    # Use selenimum to login to wildbook
    manaul_login = False
    wb_target = 'ibeis'
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
        accept_aggrement_button = driver.find_element_by_name('acceptUserAgreement')
        accept_aggrement_button.click()

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
    if ut.is_developer():
        tomcat_dpath = join(os.environ['CODE_DIR'], 'Wildbook/tmp/apache-tomcat-8.0.24')
    else:
        tomcat_dpath = '/var/lib/tomcat'
    wb_target = 'ibeis'
    return wb_target, tomcat_dpath


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
    if ut.is_developer():
        DEFAULT_TOMCAT_PATH = join(os.environ['CODE_DIR'], 'Wildbook/tmp/apache-tomcat-8.0.24')
    else:
        DEFAULT_TOMCAT_PATH = '/var/lib/tomcat'
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


def submit_wildbook_url(url, payload=None, browse_on_error=True, dryrun=False):
    """
    mirroring the one in IBEISController.py, but with changed functionality
    """
    if dryrun:
        print('[DRYrun_submit] URL=%r' % (url, ))
        response = None
        status = True
    else:
        print('[submit] URL=%r' % (url, ))
        try:
            if payload is None:
                response = requests.get(url)
                response = requests.get(url, auth=('tomcat', 'tomcat123'))
            else:
                response = requests.post(url, data=payload)
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
    for url in submit_url_list:
        status, response = submit_wildbook_url(url, payload, dryrun=dryrun)
        #print(ut.dict_str(response.__dict__, truncate=0))
        status_list.append(status)
        try:
            response_json = response.json()
            # encounter in this message is a wb-encounter not our ia-encounter
            print(response_json['message'])
        except Exception as ex:
            ut.printex(ex, 'Failed getting json from responce. This probably means there is an authentication issue')
            raise
        assert response_json['success']
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
        python -m ibeis.control.manual_wildbook_funcs --exec-hyrule_reset_wildbook

        # Login to wildbook
        python -m ibeis.control.manual_wildbook_funcs --exec-hyrule_wildbook_login

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
