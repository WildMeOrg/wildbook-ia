# -*- coding: utf-8 -*-
"""
CommandLine;
    # Reset IBEIS database (can skip if done)
    python -m wbia.tests.reset_testdbs --reset_mtest
    python -m wbia --tf reset_mtest

Notes:
    Moving components: java, tomcat, wildbook.war.

    python -m utool.util_inspect check_module_usage --pat="manual_wildbook_funcs.py"


CommandLine;
    # Start IA server
    python -m wbia --web --db PZ_MTEST

    # Reset Wildbook database
    python -m wbia purge_local_wildbook

    # Install Wildbook
    python -m wbia install_wildbook

    # Startup Wildbook
    python -m wbia startup_wildbook_server
    --show

    # Poll wildbook info
    python -m wbia get_wildbook_ia_url

    # Login to wildbook (can skip)
    python -m wbia test_wildbook_login

    # Ship ImageSets to wildbook
    python -m wbia wildbook_signal_imgsetid_list

    # Change annotations names to a single name
    python -m wbia wildbook_signal_annot_name_changes:1

    # Change annotations names back to normal
    python -m wbia wildbook_signal_annot_name_changes:2
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import requests
from wbia.control import controller_inject
from wbia.control import wildbook_manager as wb_man  # NOQA
from wbia.control.controller_inject import make_ibs_register_decorator

print, rrr, profile = ut.inject2(__name__)


DISABLE_WILDBOOK_SIGNAL = ut.get_argflag('--no-wb-signal')


CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


# PREFERED_BROWSER = 'chrome'
# webbrowser._tryorder
PREFERED_BROWSER = None
if ut.get_computer_name() == 'hyrule':
    PREFERED_BROWSER = 'firefox'


@register_ibs_method
def get_wildbook_base_url(ibs, wb_target=None):
    if DISABLE_WILDBOOK_SIGNAL:
        message = 'Wildbook signals are turned off via the command line'
        print(message)
        raise IOError(message)

    wb_port = 8080
    wb_target = ibs.const.WILDBOOK_TARGET if wb_target is None else wb_target

    if ibs.containerized:
        wb_hostname = 'nginx'
        wb_target = ''
    else:
        wb_hostname = '127.0.0.1'

    wb_hostname = str(wb_hostname)
    wb_port = str(wb_port)
    wb_target = str(wb_target)

    wildbook_base_url = 'http://%s:%s/%s/' % (wb_hostname, wb_port, wb_target,)
    wildbook_base_url = wildbook_base_url.strip('/')

    print('USING WB BASEURL: %r' % (wildbook_base_url,))
    return wildbook_base_url


@register_ibs_method
def assert_ia_available_for_wb(ibs, wb_target=None):
    # Test if we have a server alive
    try:
        ia_url = ibs.get_wildbook_ia_url(wb_target)

        if ia_url is None:
            message = 'Wildbook signals are turned off via the command line'
            print(message)
            raise IOError(message)
    except IOError:
        print('[ibs.assert_ia_available_for_wb] Caught IOError, returning None')
        return None
    except Exception as ex:
        ut.printex(ex, 'Could not get IA url. BLINDLY CHARCHING FORWARD!', iswarning=True)
    else:
        have_server = False
        for count in ut.delayed_retry_gen([1], timeout=3, raise_=False):
            try:
                rsp = requests.get(ia_url + '/api/test/heartbeat/', timeout=3)
                have_server = rsp.status_code == 200
                break
            except requests.ConnectionError:
                pass
        if not have_server:
            raise Exception('The image analysis server is not started.')
        return have_server


@register_ibs_method
def get_wildbook_ia_url(ibs, wb_target=None):
    """
    Where does wildbook expect us to be?

    CommandLine:
        python -m wbia get_wildbook_ia_url

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_wildbook_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> ia_url = ibs.get_wildbook_ia_url()
        >>> print('ia_url = %r' % (ia_url,))
    """
    import requests

    try:
        wb_url = ibs.get_wildbook_base_url(wb_target)
    except IOError:
        print('[ibs.get_wildbook_ia_url] Caught IOError, returning None')
        return None

    response = requests.get(wb_url + '/ia?status')
    status = response.status_code == 200
    if not status:
        raise Exception('Could not get IA status from wildbook')
    json_response = response.json()
    ia_url = json_response.get('iaURL')
    # print('response = %r' % (response,))
    return ia_url


@register_ibs_method
@register_api('/api/wildbook/signal/annot/name/', methods=['PUT'])
def wildbook_signal_annot_name_changes(ibs, aid_list=None, wb_target=None, dryrun=False):
    r"""
    Args:
        aid_list (int):  list of annotation ids(default = None)
        tomcat_dpath (None): (default = None)
        wb_target (None): (default = None)
        dryrun (bool): (default = False)

    CommandLine:
        python -m wbia wildbook_signal_annot_name_changes:0 --dryrun
        python -m wbia wildbook_signal_annot_name_changes:1 --dryrun
        python -m wbia wildbook_signal_annot_name_changes:1
        python -m wbia wildbook_signal_annot_name_changes:2

    Setup:
        >>> wb_target = None
        >>> dryrun = ut.get_argflag('--dryrun')

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_wildbook_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> #gid_list = ibs.get_valid_gids()[0:10]
        >>> gid_list = ibs.get_valid_gids()[3:5]
        >>> aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        >>> # Test case where some names change, some do not. There are no new names.
        >>> old_nid_list = ibs.get_annot_name_rowids(aid_list)
        >>> new_nid_list = ut.list_roll(old_nid_list, 1)
        >>> ibs.set_annot_name_rowids(aid_list, new_nid_list)
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, wb_target, dryrun)
        >>> ibs.set_annot_name_rowids(aid_list, old_nid_list)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_wildbook_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
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
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, wb_target, dryrun)
        >>> # Undo changes here (not undone in wildbook)
        >>> #ibs.set_annot_name_rowids(aid_list, old_nid_list)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_wildbook_funcs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> gid_list = ibs.get_valid_gids()[3:5]
        >>> aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        >>> old_nid_list = [1, 2]
        >>> ibs.set_annot_name_rowids(aid_list, old_nid_list)
        >>> # Signal what currently exists (should put them back to normal)
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, wb_target, dryrun)
    """
    print(
        '[ibs.wildbook_signal_annot_name_changes] signaling annot name changes to wildbook'
    )
    try:
        wb_url = ibs.get_wildbook_base_url(wb_target)
    except IOError:
        print('[ibs.wildbook_signal_annot_name_changes] Caught IOError, returning None')
        return None

    try:
        ibs.assert_ia_available_for_wb(wb_target)
    except Exception:
        pass
    if aid_list is None:
        aid_list = ibs.get_valid_aids(is_known=True)

    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    annot_name_text_list = ibs.get_annot_name_texts(aid_list)
    grouped_uuids = ut.group_items(annot_uuid_list, annot_name_text_list)
    url = wb_url + '/ia'
    payloads = [
        {
            'resolver': {
                'assignNameToAnnotations': {
                    'name': new_name,
                    'annotationIds': ut.lmap(str, annot_uuids),
                }
            }
        }
        for new_name, annot_uuids in grouped_uuids.items()
    ]
    status_list = []
    for json_payload in ut.ProgressIter(payloads, lbl='submitting URL', freq=1):
        print('[_send] URL=%r with json_payload=%r' % (url, json_payload))
        if dryrun:
            status = False
        else:
            response = requests.post(url, json=json_payload)
            status = response.status_code == 200
            if not status:
                print('Failed to push new names')
                # print(response.text)
        status_list.append(status)
    return status_list


@register_ibs_method
@register_api('/api/wildbook/signal/name/', methods=['PUT'])
def wildbook_signal_name_changes(
    ibs, nid_list, new_name_list, wb_target=None, dryrun=False
):
    r"""
    Args:
        nid_list (int):  list of name ids
        new_name_list (str):  list of corresponding names
        wb_target (None): (default = None)
        dryrun (bool): (default = False)

    CommandLine:
        python -m wbia wildbook_signal_name_changes:0 --dryrun
        python -m wbia wildbook_signal_name_changes:1 --dryrun
        python -m wbia wildbook_signal_name_changes:1
        python -m wbia wildbook_signal_name_changes:2

    Setup:
        >>> wb_target = None
        >>> dryrun = ut.get_argflag('--dryrun')
    """
    print('[ibs.wildbook_signal_name_changes] signaling name changes to wildbook')
    try:
        wb_url = ibs.get_wildbook_base_url(wb_target)
    except IOError:
        print('[ibs.wildbook_signal_name_changes] Caught IOError, returning None')
        return None

    try:
        ibs.assert_ia_available_for_wb(wb_target)
    except Exception:
        pass
    current_name_list = ibs.get_name_texts(nid_list)
    combined_list = sorted(list(zip(new_name_list, current_name_list)), reverse=True)
    url = wb_url + '/ia'
    json_payload = {
        'resolver': {
            'renameIndividuals': {
                'new': [_[0] for _ in combined_list],
                'old': [_[1] for _ in combined_list],
            }
        }
    }
    status_list = []
    print('[_send] URL=%r with json_payload=%r' % (url, json_payload))
    if dryrun:
        status = False
    else:
        response = requests.post(url, json=json_payload)
        response_json = response.json()
        status = response.status_code == 200 and response_json['success']
        if not status:
            status_list = False
            print('Failed to update names')
            # print(response.text)
        else:
            for name_response in response_json['results']:
                status = name_response['success']
                error = name_response.get('error', '')
                status = status or 'unknown MarkedIndividual' in error
                status_list.append(status)
        # ut.embed()
    return status_list


@register_ibs_method
def wildbook_get_existing_names(ibs, wb_target=None):
    print('[ibs.wildbook_get_existing_names] getting existing names out of wildbook')
    try:
        wb_url = ibs.get_wildbook_base_url(wb_target)
    except IOError:
        print('[ibs.wildbook_get_existing_names] Caught IOError, returning None')
        return None

    try:
        ibs.assert_ia_available_for_wb(wb_target)
    except Exception:
        pass
    url = wb_url + '/rest/org.ecocean.MarkedIndividual'
    response = requests.get(url)
    response_json = response.json()
    try:
        wildbook_existing_name_list = [_['individualID'] for _ in response_json]
        wildbook_existing_name_list = list(set(wildbook_existing_name_list))
    except Exception:
        wildbook_existing_name_list = []
    return wildbook_existing_name_list


@register_ibs_method
@register_api('/api/wildbook/signal/imageset/', methods=['PUT'])
def wildbook_signal_imgsetid_list(
    ibs,
    imgsetid_list=None,
    set_shipped_flag=True,
    open_url_on_complete=True,
    wb_target=None,
    dryrun=False,
):
    """
    Exports specified imagesets to wildbook. This is a synchronous call.

    Args:
        imgsetid_list (list): (default = None)
        set_shipped_flag (bool): (default = True)
        open_url_on_complete (bool): (default = True)

    RESTful:
        Method: PUT
        URL:    /api/wildbook/signal/imageset/

    Ignore:
        cd $CODE_DIR/Wildbook/tmp

        # Ensure IA server is up
        python -m wbia --web --db PZ_MTEST

        # Reset IBEIS database
        python -m wbia.tests.reset_testdbs --reset_mtest
        python -m wbia  reset_mtest

        # Completely remove Wildbook database
        python -m wbia  purge_local_wildbook

        # Install Wildbook
        python -m wbia  install_wildbook

        # Startup Wildbook
        python -m wbia  startup_wildbook_server

        # Login to wildbook
        python -m wbia  test_wildbook_login

        # Ship ImageSets to wildbook
        python -m wbia  wildbook_signal_imgsetid_list

        # Change annotations names to a single name
        python -m wbia  wildbook_signal_annot_name_changes:1

        # Change annotations names back to normal
        python -m wbia  wildbook_signal_annot_name_changes:2

    CommandLine:
        python -m wbia wildbook_signal_imgsetid_list
        python -m wbia wildbook_signal_imgsetid_list --dryrun
        python -m wbia wildbook_signal_imgsetid_list --break

    SeeAlso:
        ~/local/build_scripts/init_wildbook.sh

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.control.manual_wildbook_funcs import *  # NOQA
        >>> dryrun = ut.get_argflag('--dryrun')
        >>> wb_target = None
        >>> import wbia
        >>> # Need to start a web server for wildbook to hook into
        >>> defaultdb = 'PZ_MTEST'
        >>> ibs = wbia.opendb(defaultdb=defaultdb)
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
        >>>     with wbia.opendb_bg_web(defaultdb, managed=True) as web_ibs:
        ...         result = web_ibs.wildbook_signal_imgsetid_list(imgsetid_list, set_shipped_flag, open_url_on_complete, wb_target, dryrun)
        >>> else:
        ...     result = ibs.wildbook_signal_imgsetid_list(imgsetid_list, set_shipped_flag, open_url_on_complete, wb_target, dryrun)
        >>> # cleanup
        >>> #ibs.delete_imagesets(new_imgsetid)
        >>> print(result)

    """
    try:
        wb_url = ibs.get_wildbook_base_url(wb_target)
    except IOError:
        print('[ibs.wildbook_signal_imgsetid_list] Caught IOError, returning None')
        return None

    try:
        ibs.assert_ia_available_for_wb(wb_target)
    except Exception:
        pass

    if imgsetid_list is None:
        imgsetid_list = ibs.get_valid_imgsetids()

    # Check to make sure imagesets are ok:
    for imgsetid in imgsetid_list:
        # First, check if imageset can be pushed
        aid_list = ibs.get_imageset_aids(imgsetid)
        assert (
            len(aid_list) > 0
        ), 'ImageSet imgsetid=%r cannot be shipped with0 annots' % (imgsetid,)
        unknown_flags = ibs.is_aid_unknown(aid_list)
        unnamed_aid_list = ut.compress(aid_list, unknown_flags)
        unnamed_ok_aid_list = ibs.filter_annots_general(unnamed_aid_list, minqual='ok',)
        nUnnamedOk = sum(unnamed_ok_aid_list)
        assert nUnnamedOk == 0, (
            'ImageSet imgsetid=%r1 cannot be shipped becuase '
            'annotation(s) %r with an identifiable quality have '
            'not been named'
        ) % (imgsetid, unnamed_ok_aid_list,)

    # Call Wildbook url to signal update
    print(
        '[ibs.wildbook_signal_imgsetid_list] ship imgsetid_list = %r to wildbook'
        % (imgsetid_list,)
    )
    imageset_uuid_list = ibs.get_imageset_uuid(imgsetid_list)
    print(
        '[ibs.wildbook_signal_imgsetid_list] ship imgset_uuid_list = %r to wildbook'
        % (imageset_uuid_list,)
    )

    url = wb_url + '/ia'
    dbname = ibs.db.get_db_init_uuid()
    occur_url_fmt = wb_url + '/occurrence.jsp?number={uuid}&dbname={dbname}'
    # enc_url_fmt = (wb_url + '/encounters/encounter.jsp?number={uuid}')

    # Check and push 'done' imagesets
    status_list = []
    for imgsetid, imageset_uuid in zip(imgsetid_list, imageset_uuid_list):
        print('[_send] URL=%r' % (url,))
        json_payload = {'resolver': {'fromIAImageSet': str(imageset_uuid)}}
        if dryrun:
            status = False
        else:
            response = requests.post(url, json=json_payload)
            status = response.status_code == 200
            print('response = %r' % (response,))
            if set_shipped_flag:
                ibs.set_imageset_shipped_flags([imgsetid], [status])
                if status and open_url_on_complete:
                    view_occur_url = occur_url_fmt.format(
                        uuid=imageset_uuid, dbname=dbname
                    )
                    _browser = ut.get_prefered_browser(PREFERED_BROWSER)
                    _browser.open_new_tab(view_occur_url)
        status_list.append(status)

    try:
        ibs.update_special_imagesets()
        ibs.notify_observers()
    except Exception:
        pass

    return status_list


@register_ibs_method
def get_flukebook_image_uuids(ibs):
    from datetime import datetime
    import uuid
    import pytz

    PST = pytz.timezone('US/Pacific')

    url = 'https://www.flukebook.org/acmIdSync.jsp'

    now = datetime.now(tz=PST)
    timestamp = now.strftime('%Y-%m-%d-%H-00-00')
    filename = 'flukebook.image.admid.%s.json' % (timestamp,)
    filepath = ut.grab_file_url(url, appname='wbia', fname=filename)

    with open(filepath, 'r') as file:
        file_content = file.read()
        file_json = ut.from_json(file_content)
    print('Loaded %d Image ACM string UUIDs from Flukebook' % (len(file_json),))

    uuid_list = []
    for uuid_str in file_json:
        try:
            uuid_ = uuid.UUID(uuid_str)
            uuid_list.append(uuid_)
        except ValueError:
            continue

    print('Validated %d Image UUIDs from Flukebook' % (len(uuid_list),))
    flukebook_image_uuid_list = list(set(uuid_list))
    print('Validated %d de-duplicated Image UUIDs from Flukebook' % (len(uuid_list),))

    return flukebook_image_uuid_list


@register_ibs_method
def delete_flukebook_orphaned_image_uuids(ibs, auto_delete=True):
    flukebook_image_uuid_list = ibs.get_flukebook_image_uuids()

    gid_list = ibs.get_valid_gids()
    local_image_uuid_list = ibs.get_image_uuids(gid_list)

    unknown_uuid_list = list(set(flukebook_image_uuid_list) - set(local_image_uuid_list))
    candidate_uuid_list = list(
        set(local_image_uuid_list) - set(flukebook_image_uuid_list)
    )

    print(
        'There are %d Image UUIDs in Flukebook that are not here'
        % (len(unknown_uuid_list),)
    )
    print(
        'There are %d Image UUIDs in here that are not in Flukebook'
        % (len(candidate_uuid_list),)
    )

    if auto_delete and len(candidate_uuid_list) > 0:
        candidate_gid_list = ibs.get_image_gids_from_uuid(candidate_uuid_list)
        assert None not in candidate_gid_list
        ibs.delete_images(candidate_gid_list)

    return candidate_uuid_list


@register_ibs_method
def get_flukebook_annot_uuids(ibs, filter_match_against_on=True):
    from datetime import datetime
    import uuid
    import pytz

    PST = pytz.timezone('US/Pacific')

    url = 'https://www.flukebook.org/acmIdSync.jsp?annotations'

    now = datetime.now(tz=PST)
    timestamp = now.strftime('%Y-%m-%d-%H-00-00')
    filename = 'flukebook.annot.admid.%s.json' % (timestamp,)
    filepath = ut.grab_file_url(url, appname='wbia', fname=filename)

    with open(filepath, 'r') as file:
        file_content = file.read()
        file_json = ut.from_json(file_content)
    print('Loaded %d Annot ACM string UUIDs from Flukebook' % (len(file_json),))

    uuid_list = []
    species_list = []
    for uuid_str in file_json:
        content = file_json[uuid_str]
        try:
            uuid_ = uuid.UUID(uuid_str)
        except ValueError:
            continue

        match = content.get('match', None)
        species = content.get('species', None)
        assert None not in [match, species]
        assert len(match) == len(species)

        if len(match) > 1:
            match_ = set(match)
            species_ = set(species)
            if len(match_) == 1 and len(species_) == 1:
                match = list(match_)
                species = list(species_)
                assert len(match) == len(species)

        if len(match) > 1:
            match_ = set(match)
            species_ = set(species)
            if len(species_) == 1:
                match = [any(match_)]
                species = list(species_)

        if len(match) == 1 and len(species) == 1:
            match = match[0]
            species = species[0]

            if filter_match_against_on is None or match == filter_match_against_on:
                uuid_list.append(uuid_)
                species_list.append(species)
        else:
            print(match, species)

    assert len(uuid_list) == len(species_list)
    assert len(uuid_list) == len(set(uuid_list))
    print('Validated %d Annotation UUIDs from Flukebook' % (len(uuid_list),))

    flukebook_annot_uuid_list = uuid_list
    flukebook_annot_species_list = species_list

    return flukebook_annot_uuid_list, flukebook_annot_species_list


@register_ibs_method
def delete_flukebook_orphaned_annot_uuids(ibs, auto_delete=True):
    from wbia import constants as const

    (
        flukebook_annot_uuid_list,
        flukebook_annot_species_list,
    ) = ibs.get_flukebook_annot_uuids()

    aid_list = ibs.get_valid_aids()
    local_annot_uuid_list = ibs.get_annot_uuids(aid_list)

    unknown_uuid_list = list(set(flukebook_annot_uuid_list) - set(local_annot_uuid_list))
    candidate_uuid_list = list(
        set(local_annot_uuid_list) - set(flukebook_annot_uuid_list)
    )

    print(
        'There are %d Annot UUIDs in Flukebook that are not here'
        % (len(unknown_uuid_list),)
    )
    print(
        'There are %d Annot UUIDs in here that are not in Flukebook'
        % (len(candidate_uuid_list),)
    )

    if auto_delete and len(candidate_uuid_list) > 0:
        candidate_aid_list = ibs.get_annot_aids_from_uuid(candidate_uuid_list)
        assert None not in candidate_aid_list
        ibs.delete_annots(candidate_aid_list)

    # Update species
    aid_list = ibs.get_valid_aids()
    local_annot_uuid_list = ibs.get_annot_uuids(aid_list)

    known_uuid_list = list(set(flukebook_annot_uuid_list) & set(local_annot_uuid_list))
    known_aid_list = ibs.get_annot_aids_from_uuid(known_uuid_list)
    assert None not in known_aid_list

    flukebook_species_dict = dict(
        zip(flukebook_annot_uuid_list, flukebook_annot_species_list)
    )
    known_species_list = ibs.get_annot_species(known_aid_list)

    flukebook_species_mapping = {
        'humpback_whale': 'megaptera_novaeangliae',
        'tursiops_sp.': 'tursiops_sp',
    }

    update_dict = {}
    update_aid_list = []
    update_species_list = []
    for known_aid, known_uuid, known_species in zip(
        known_aid_list, known_uuid_list, known_species_list
    ):
        flukebook_species = flukebook_species_dict.get(known_uuid, None)
        if flukebook_species is None:
            flukebook_species = const.UNKNOWN
        assert flukebook_species is not None
        flukebook_species = flukebook_species.lower()
        flukebook_species = flukebook_species.replace(' ', '_')
        flukebook_species = flukebook_species_mapping.get(
            flukebook_species, flukebook_species
        )

        if known_species != flukebook_species:

            if flukebook_species == const.UNKNOWN and known_species != const.UNKNOWN:
                continue

            update_aid_list.append(known_aid)
            update_species_list.append(flukebook_species)

            if known_species not in update_dict:
                update_dict[known_species] = {}
            if flukebook_species not in update_dict[known_species]:
                update_dict[known_species][flukebook_species] = 0
            update_dict[known_species][flukebook_species] += 1

    assert len(update_aid_list) == len(update_species_list)
    print(ut.repr3(update_dict))

    ibs.set_annot_species(update_aid_list, update_species_list)

    return candidate_uuid_list, update_dict


@register_ibs_method
@register_api('/api/flukebook/sync/', methods=['GET', 'POST'])
def flukebook_sync(ibs, **kwargs):
    candidate_image_uuid_list = ibs.delete_flukebook_orphaned_image_uuids()
    candidate_annot_uuid_list, update_dict = ibs.delete_flukebook_orphaned_annot_uuids()

    result_dict = {
        'deleted_images': len(candidate_image_uuid_list),
        'deleted_annots': len(candidate_annot_uuid_list),
        'species_update': update_dict,
    }
    return result_dict


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.control.manual_wildbook_funcs
        python -m wbia.control.manual_wildbook_funcs --allexamples
        python -m wbia.control.manual_wildbook_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
