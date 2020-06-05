# -*- coding: utf-8 -*-
"""
python -c "import utool as ut; ut.write_modscript_alias('Tgen.sh', 'wbia.templates.template_generator')"
sh Tgen.sh --key test --invert --Tcfg with_getters=True with_setters=False --modfname manual_test_funcs

# TODO: Fix this name it is too special case
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
from six.moves import zip
import ubelt as ub  # NOQA
from wbia import constants as const
from wbia.control import accessor_decors, controller_inject  # NOQA
import utool as ut
import uuid
from wbia.control.controller_inject import make_ibs_register_decorator

print, rrr, profile = ut.inject2(__name__)


VERBOSE_SQL = ut.get_argflag(('--print-sql', '--verbose-sql', '--verb-sql', '--verbsql'))
CLASS_INJECT_KEY, register_ibs_method = make_ibs_register_decorator(__name__)


register_api = controller_inject.get_wbia_flask_api(__name__)


TEST_ROWID = 'test_rowid'
TEST_UUID = 'test_uuid'
TEST_USER_IDENTITY = 'test_user_identity'
TEST_CHALLENGE = 'test_challenge_json'
TEST_RESPONSE = 'test_response_json'
TEST_RESULT = 'test_result'
TEST_TIME = 'test_time_posix'


@register_ibs_method
@accessor_decors.ider
@register_api('/api/test/', methods=['GET'])
def _get_all_test_rowids(ibs):
    r"""
    Returns:
        list_ (list): all nids of known animals
        (does not include unknown names)
    """
    all_known_test_rowids = ibs.staging.get_all_rowids(const.TEST_TABLE)
    return all_known_test_rowids


@register_ibs_method
@accessor_decors.adder
@register_api('/api/test/', methods=['POST'])
def add_test(
    ibs,
    test_challenge_list,
    test_response_list,
    test_result_list=None,
    test_uuid_list=None,
    test_user_identity_list=None,
):
    assert len(test_challenge_list) == len(test_response_list)
    n_input = len(test_challenge_list)

    test_challenge_list = [
        ut.to_json(test_challenge) for test_challenge in test_challenge_list
    ]

    test_response_list = [
        ut.to_json(test_response) for test_response in test_response_list
    ]

    if test_uuid_list is None:
        test_uuid_list = [uuid.uuid4() for _ in range(n_input)]
    if test_result_list is None:
        test_result_list = [None] * n_input
    if test_user_identity_list is None:
        test_user_identity_list = [None] * n_input

    superkey_paramx = (0,)
    colnames = [
        TEST_UUID,
        TEST_CHALLENGE,
        TEST_RESPONSE,
        TEST_RESULT,
        TEST_USER_IDENTITY,
    ]
    params_iter = list(
        zip(
            test_uuid_list,
            test_challenge_list,
            test_response_list,
            test_result_list,
            test_user_identity_list,
        )
    )
    test_rowid_list = ibs.staging.add_cleanly(
        const.TEST_TABLE,
        colnames,
        params_iter,
        ibs.get_test_rowids_from_uuid,
        superkey_paramx,
    )
    return test_rowid_list


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/test/rowid/uuid/', methods=['GET'])
def get_test_rowids_from_uuid(ibs, uuid_list):
    test_rowid_list = ibs.staging.get(
        const.TEST_TABLE, (TEST_ROWID,), uuid_list, id_colname=TEST_UUID
    )
    return test_rowid_list


@register_ibs_method
@accessor_decors.deleter
@register_api('/api/test/', methods=['DELETE'])
def delete_test(ibs, test_rowid_list):
    r"""
    deletes tests from the database

    RESTful:
        Method: DELETE
        URL:    /api/test/
    """
    if ut.VERBOSE:
        print('[ibs] deleting %d tests' % len(test_rowid_list))
    ibs.staging.delete_rowids(const.TEST_TABLE, test_rowid_list)


@register_ibs_method
@accessor_decors.getter_1to1
@register_api('/api/test/uuid/', methods=['GET'])
def get_test_uuid(ibs, test_rowid_list):
    test_uuid_list = ibs.staging.get(const.TEST_TABLE, (TEST_UUID,), test_rowid_list)
    return test_uuid_list


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.control.manual_test_funcs
        python -m wbia.control.manual_test_funcs --allexamples
        python -m wbia.control.manual_test_funcs --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
