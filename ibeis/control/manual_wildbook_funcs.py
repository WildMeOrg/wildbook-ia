# -*- coding: utf-8 -*-
"""
CommandLine;
    # Reset IBEIS database (can skip if done)
    python -m ibeis.tests.reset_testdbs --reset_mtest
    python -m ibeis --tf reset_mtest

Notes:
    Moving compoments: java, tomcat, wildbook.war.

    python -m utool.util_inspect check_module_usage --pat="manual_wildbook_funcs.py"


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

    # Poll wildbook info
    python -m ibeis get_wildbook_ia_url

    # Login to wildbook (can skip)
    python -m ibeis test_wildbook_login

    # Ship ImageSets to wildbook
    python -m ibeis wildbook_signal_imgsetid_list

    # Change annotations names to a single name
    python -m ibeis wildbook_signal_annot_name_changes:1

    # Change annotations names back to normal
    python -m ibeis wildbook_signal_annot_name_changes:2
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import requests
from ibeis.control import controller_inject
from ibeis.control import wildbook_manager as wb_man  # NOQA
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


@register_ibs_method
def get_wildbook_base_url(ibs, wb_target=None):
    wb_target = ibs.const.WILDBOOK_TARGET if wb_target is None else wb_target
    hostname = '127.0.0.1'
    wb_port = 8080
    wildbook_base_url = 'http://' + str(hostname) + ':' + str(wb_port) + '/' + wb_target
    return wildbook_base_url


@register_ibs_method
def assert_ia_available_for_wb(ibs, wb_target=None):
    # Test if we have a server alive
    try:
        ia_url = ibs.get_wildbook_ia_url(wb_target)
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
        python -m ibeis get_wildbook_ia_url

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.control.manual_wildbook_funcs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> ia_url = ibs.get_wildbook_ia_url()
        >>> print('ia_url = %r' % (ia_url,))
    """
    import requests
    wb_url = ibs.get_wildbook_base_url(wb_target)
    response = requests.get(wb_url + '/ia?status')
    status = response.status_code == 200
    #result_json = {"settings":{"IBEISIARestUrlAddAnnotations":"http://52.37.240.178:5000/api/annot/json/"},"iaURL":"http://52.37.240.178:5000/","iaEnabled":true,"timestamp":1466534267714}
    if not status:
        raise Exception('Cou.ld not get IA status from wildbook')
    json_response = response.json()
    ia_url = json_response.get('iaURL')
    #print('response = %r' % (response,))
    return ia_url


@register_ibs_method
@register_api('/api/wildbook/signal/annot/name/', methods=['PUT'])
def wildbook_signal_annot_name_changes(ibs, aid_list=None, wb_target=None,
                                       dryrun=False):
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

    Setup:
        >>> wb_target = None
        >>> dryrun = ut.get_argflag('--dryrun')

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
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, wb_target, dryrun)
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
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, wb_target, dryrun)
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
        >>> result = ibs.wildbook_signal_annot_name_changes(aid_list, wb_target, dryrun)
    """
    print('[ibs.wildbook_signal_imgsetid_list] signaling annot name changes to wildbook')
    wb_url = ibs.get_wildbook_base_url(wb_target)
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
        {'resolver': {'assignNameToAnnotations': {
            'name': new_name,
            'annotationIds' : ut.lmap(str, annot_uuids),
        }}}
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
                print(response.text)
        status_list.append(status)
    return status_list


@register_ibs_method
@register_api('/api/wildbook/signal/imageset/', methods=['PUT'])
def wildbook_signal_imgsetid_list(ibs, imgsetid_list=None,
                                  set_shipped_flag=True,
                                  open_url_on_complete=True,
                                  wb_target=None, dryrun=False):
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
        python -m ibeis --web --db PZ_MTEST

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
        >>> wb_target = None
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
        >>> result = ibs.wildbook_signal_imgsetid_list(imgsetid_list, set_shipped_flag, open_url_on_complete, wb_target, dryrun)
        >>> # cleanup
        >>> #ibs.delete_imagesets(new_imgsetid)
        >>> print(result)
        >>> if ut.get_argflag('--bg'):
        >>>     web_ibs.terminate2()

    """
    wb_url = ibs.get_wildbook_base_url(wb_target)
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
        assert len(aid_list) > 0, (
            'ImageSet imgsetid=%r cannot be shipped with0 annots' % (imgsetid,))
        unknown_flags = ibs.is_aid_unknown(aid_list)
        unnamed_aid_list = ut.compress(aid_list, unknown_flags)
        unnamed_ok_aid_list = ibs.filter_annots_general(
            unnamed_aid_list,
            minqual='ok',
        )
        nUnnamedOk = sum(unnamed_ok_aid_list)
        assert nUnnamedOk == 0, (
            ('ImageSet imgsetid=%r1 cannot be shipped becuase '
             'annotation(s) %r with an identifiable quality have '
             'not been named') % (imgsetid, unnamed_ok_aid_list, ))

    # Call Wildbook url to signal update
    print('[ibs.wildbook_signal_imgsetid_list] ship imgsetid_list = %r to wildbook' % (
        imgsetid_list, ))

    url = wb_url + '/ia'
    occur_url_fmt = (wb_url + '/occurrence.jsp?number={uuid}')
    #enc_url_fmt = (wb_url + '/encounters/encounter.jsp?number={uuid}')

    # Check and push 'done' imagesets
    status_list = []
    for imgsetid in imgsetid_list:
        imageset_uuid = ibs.get_imageset_uuid(imgsetid)
        print('[_send] URL=%r' % (url, ))
        json_payload = {'resolver': {'fromIAImageSet': str(imageset_uuid) }}
        if dryrun:
            status = False
        else:
            response = requests.post(url, json=json_payload)
            status = response.status_code == 200
            print('response = %r' % (response,))
            if set_shipped_flag:
                ibs.set_imageset_shipped_flags([imgsetid], [status])
                if status and open_url_on_complete:
                    view_occur_url = occur_url_fmt.format(uuid=imageset_uuid,)
                    _browser = ut.get_prefered_browser(PREFERED_BROWSER)
                    _browser.open_new_tab(view_occur_url)
        status_list.append(status)
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
