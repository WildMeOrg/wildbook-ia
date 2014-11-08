"""
TemplateInfo:
    autogen_time = 22:12:50 2014/11/07

ToRegenerate:
    python ibeis/control/templates.py --dump-autogen-controller
"""
from __future__ import absolute_import, division, print_function
import functools  # NOQA
import six  # NOQA
from six.moves import map, range  # NOQA
from ibeis import constants
from ibeis.control.IBEISControl import IBEISController
import utool  # NOQA
import utool as ut  # NOQA
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[autogen_ibsfuncs]')

# Create dectorator to inject these functions into the IBEISController
register_ibs_aliased_method   = ut.make_class_method_decorator((IBEISController, 'autogen'))
register_ibs_unaliased_method = ut.make_class_method_decorator((IBEISController, 'autogen'))


def register_ibs_method(func):
    aliastup = (func, 'autogen_' + ut.get_funcname(func))
    register_ibs_unaliased_method(func)
    register_ibs_aliased_method(aliastup)
    return func

# AUTOGENED CONSTANTS:
ANNOT_ROWID                 = 'annot_rowid'
CHIP_ROWID                  = 'chip_rowid'
CONFIG_ROWID                = 'config_rowid'
FEATWEIGHT_FORGROUND_WEIGHT = 'featweight_forground_weight'
FEATWEIGHT_ROWID            = 'featweight_rowid'
FEAT_ROWID                  = 'feature_rowid'

# =========================
# NATIVE.TDELETER METHODS
# =========================


@register_ibs_method
#@deleter
#@cache_invalidator(constants.FEATURE_WEIGHT_TABLE)
def delete_featweight(ibs, featweight_rowid_list):
    """
    featweight.delete(featweight_rowid_list)

    delete featweight rows

    Args:
        featweight_rowid_list

    TemplateInfo:
        Tdeleter_native_tbl
        tbl = featweight

    Tdeleter_native_tbl
    """
    from ibeis.model.preproc import preproc_featweight
    if utool.VERBOSE:
        print('[ibs] deleting %d featweight rows' % len(featweight_rowid_list))
    # Prepare: Delete externally stored data (if any)
    preproc_featweight.on_delete(ibs, featweight_rowid_list)
    # Finalize: Delete self
    ibs.dbcache.delete_rowids(
        constants.FEATURE_WEIGHT_TABLE, featweight_rowid_list)


# =========================
# NATIVE.TGETTER METHODS
# =========================


@register_ibs_method
#@getter
def get_featweight_rowid_from_superkey(ibs, feature_rowid_list, config_rowid_list):
    """
    featweight_rowid_list <- featweight[feature_rowid_list, config_rowid_list]

    Args:
        superkey lists: feature_rowid_list, config_rowid_list

    Returns:
        featweight_rowid_list

    TemplateInfo:
        Tgetter_native_rowid_from_superkey
        tbl = featweight
    """
    colnames = (FEATWEIGHT_ROWID),
    # FIXME: col_rowid is not correct
    params_iter = zip(feature_rowid_list, config_rowid_list)
    andwhere_colnames = [feature_rowid_list, config_rowid_list]
    featweight_rowid_list = ibs.dbcache.get_where2(
        constants.FEATURE_WEIGHT_TABLE, colnames, params_iter, andwhere_colnames)
    return featweight_rowid_list


# =========================
# NATIVE.TGETTER_DEPENDANT METHODS
# =========================


@register_ibs_method
#@getter
def get_annot_fgweights(ibs, aid_list, qreq_=None, ensure=False):
    """
    featweight_rowid_list <- annot.featweight.rowids[aid_list]

    get fgweight data of the annot table using the dependant featweight table

    Args:
        aid_list (list):

    Returns:
        list: fgweight_list

    TemplateInfo:
        Tgetter_rl_pclines_dependant_column
        root = annot
        col  = fgweight
        leaf = featweight

    """
    # Get leaf rowids
    cid_list = ibs.get_annot_cids(aid_list, qreq_=qreq_, ensure=ensure)
    fid_list = ibs.get_chip_fids(cid_list, qreq_=qreq_, ensure=ensure)
    featweight_rowid_list = ibs.get_feat_featweight_rowids(
        fid_list, qreq_=qreq_, ensure=ensure)
    # Get col values
    fgweight_list = ibs.get_featweight_fgweight(featweight_rowid_list)
    return fgweight_list


# =========================
# NATIVE.TGETTER_NATIVE METHODS
# =========================


@register_ibs_method
#@getter
def get_featweight_fgweight(ibs, featweight_rowid_list):
    """
    fgweight_list <- featweight.fgweight[featweight_rowid_list]

    gets data from the "native" column "fgweight" in the "featweight" table

    Args:
        featweight_rowid_list (list):

    Returns:
        list: fgweight_list

    TemplateInfo:
        Tgetter_table_column
        col = fgweight
        tbl = featweight
    """
    id_iter = featweight_rowid_list
    colnames = (FEATWEIGHT_FORGROUND_WEIGHT,)
    fgweight_list = ibs.dbcache.get(
        constants.FEATURE_WEIGHT_TABLE, colnames, id_iter, id_colname='rowid')
    return fgweight_list


# =========================
# NATIVE.TIDER METHODS
# =========================


@register_ibs_method
#@ider
def _get_all_featweight_rowids(ibs):
    """
    all_featweight_rowids <- featweight.get_all_rowids()

    Returns:
        list_ (list): unfiltered featweight_rowids

    TemplateInfo:
        Tider_all_rowids
        tbl = featweight
    """
    all_featweight_rowids = ibs.dbcache.get_all_rowids(
        constants.FEATURE_WEIGHT_TABLE)
    return all_featweight_rowids


# =========================
# PL.TADDER METHODS
# =========================


@register_ibs_method
#@adder
def add_feat_featweights(ibs, fid_list, qreq_=None):
    """
    feat.featweight.add(fid_list)

    Adds / ensures / computes a dependant property

    Tadder_pl_dependant -- CRITICAL FUNCTION MUST EXIST FOR ALL DEPENDANTS

    parent=feat
    leaf=featweight

    returns config_rowid of the current configuration

    Example:
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> fid_list = ibs.get_valid_fids()
        >>> qreq_ = None
        >>> featweight_rowid_list = ibs.add_feat_featweights(fid_list, qreq_=qreq_)
    """
    from ibeis.model.preproc import preproc_featweight
    # Get requested configuration id
    config_rowid = ibs.get_featweight_config_rowid(qreq_=qreq_)
    # Find leaf rowids that need to be computed
    featweight_rowid_list = ibs.get_feat_featweight_rowids(
        fid_list, qreq_=qreq_, ensure=False)
    # Get corresponding "dirty" parent rowids
    dirty_fid_list = utool.get_dirty_items(fid_list, featweight_rowid_list)
    if len(dirty_fid_list) > 0:
        if utool.VERBOSE:
            print('[ibs] adding %d / %d featweight' %
                  (len(dirty_fid_list), len(fid_list)))

        # Dependant columns do not need true from_superkey getters.
        # We can use the  Tgetter_rl_dependant_rowids instead
        get_rowid_from_superkey = functools.partial(
            ibs.get_feat_featweight_rowids, qreq_=qreq_, ensure=False)
        fgweight_list = preproc_featweight.add_featweight_params_gen(
            ibs, fid_list)
        params_iter = ((fid, config_rowid, fgweight)
                       for fid, fgweight in
                       zip(fid_list, fgweight_list))
        colnames = [
            'feature_rowid', 'config_rowid', 'featweight_forground_weight']
        featweight_rowid_list = ibs.dbcache.add_cleanly(
            constants.FEATURE_WEIGHT_TABLE, colnames, params_iter, get_rowid_from_superkey)
    return featweight_rowid_list


# =========================
# PL.TCFG METHODS
# =========================


@register_ibs_method
#@ider
def get_featweight_config_rowid(ibs, qreq_=None):
    """
    featweight_cfg_rowid = featweight.config_rowid()

    returns config_rowid of the current configuration
    Config rowids are always ensured


    TemplateInfo:
        Tcfg_leaf_config_rowid_getter
        leaf = featweight

    Example:
        >>> import ibeis; ibs = ibeis.opendb('testdb1')
        >>> featweight_cfg_rowid = ibs.get_featweight_config_rowid()
    """
    if qreq_ is not None:
        featweight_cfg_suffix = qreq_.qparams.featweight_cfgstr
        # TODO store config_rowid in qparams
    else:
        featweight_cfg_suffix = ibs.cfg.featweight_cfg.get_cfgstr()
    featweight_cfg_rowid = ibs.add_config(featweight_cfg_suffix)
    return featweight_cfg_rowid


# =========================
# PL.TGETTER METHODS
# =========================


@register_ibs_method
#@getter
def get_feat_featweight_rowids(ibs, fid_list, qreq_=None, ensure=False):
    """
    featweight_rowid_list <- feat.featweight.rowids[fid_list]

    get featweight rowids of feat under the current state configuration

    Args:
        fid_list (list):

    Returns:
        list: featweight_rowid_list

    TemplateInfo:
        Tgetter_pl_dependant_rowids
        parent = feat
        leaf = featweight

    Example:
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> fid_list = ibs.get_valid_fids()
        >>> qreq_ = None
        >>> ensure = False
        >>> featweight_rowid_list = ibs.get_feat_featweight_rowids(fid_list, qreq_, ensure)
    """
    if ensure:
        featweight_rowid_list = ibs.add_feat_featweights(fid_list, qreq_=qreq_)
        return featweight_rowid_list
    else:
        colnames = (FEATWEIGHT_ROWID,)
        config_rowid = ibs.get_featweight_config_rowid(qreq_=qreq_)
        andwhere_colnames = (FEAT_ROWID, CONFIG_ROWID,)
        params_iter = ((fid, config_rowid,) for fid in fid_list)
        featweight_rowid_list = ibs.dbcache.get_where2(
            constants.FEATURE_WEIGHT_TABLE, colnames, params_iter, andwhere_colnames)
        return featweight_rowid_list


# =========================
# RL.TADDER METHODS
# =========================


@register_ibs_method
#@adder
def add_annot_featweights(ibs, aid_list, qreq_=None):
    """
    featweight_rowid_list <- annot.featweight.ensure(aid_list)

    Adds / ensures / computes a dependant property

    returns config_rowid of the current configuration

    CONVINIENCE FUNCTION

    TemplateInfo:
        Tadder_rl_dependant
        root = annot
        leaf = featweight

    Example:
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> qreq_ = None
        >>> featweight_rowid_list = ibs.add_annot_featweights(aid_list, qreq_=qreq_)
    """
    fid_list = ibs.get_annot_fids(aid_list, qreq_=qreq_, ensure=True)
    featweight_rowid_list = ibs.add_feat_featweights(fid_list, qreq_=qreq_)
    return featweight_rowid_list


# =========================
# RL.TDELETER METHODS
# =========================


@register_ibs_method
#@deleter
#@cache_invalidator(constants.ANNOTATION_TABLE)
def delete_annot_featweight(ibs, aid_list, qreq_=None):
    """
    annot.featweight.delete(aid_list)

    Args:
        aid_list

    TemplateInfo:
        Tdeleter_rl_depenant
        root = annot
        leaf = featweight
    """
    if utool.VERBOSE:
        print('[ibs] deleting %d annots leaf nodes' % len(aid_list))
    # Delete any dependants
    _featweight_rowid_list = ibs.get_annot_featweight_rowids(
        aid_list, qreq_=qreq_, ensure=False)
    featweight_rowid_list = ut.filter_Nones(_featweight_rowid_list)
    ibs.delete_featweight(featweight_rowid_list)


# =========================
# RL.TGETTER METHODS
# =========================


@register_ibs_method
#@getter
def get_annot_featweight_all_rowids(ibs, aid_list):
    """
    featweight_rowid_list <- annot.featweight.all_rowids([aid_list])

    get featweight rowids of annot under the current state configuration

    Args:
        aid_list (list):

    Returns:
        list: featweight_rowid_list

    TemplateInfo:
        Tgetter_rl_dependant_all_rowids
        root = annot
        leaf = featweight
    """
    colnames = (FEAT_ROWID,)
    featweight_rowid_list = ibs.dbcache.get(
        constants.FEATURE_WEIGHT_TABLE, colnames, aid_list,
        id_colname=ANNOT_ROWID)
    return featweight_rowid_list


@register_ibs_method
#@getter
def get_annot_featweight_rowids(ibs, aid_list, qreq_=None, ensure=False):
    """
    featweight_rowid_list = annot.featweight.rowids[aid_list]

    get featweight rowids of annot under the current state configuration

    Args:
        aid_list (list):

    Returns:
        list: featweight_rowid_list

    TemplateInfo:
        Tgetter_rl_dependant_rowids
        root        = annot
        leaf_parent = feat
        leaf        = featweight

    Example:
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> aid_list = ibs.get_valid_aids()
        >>> qreq_ = None
        >>> ensure = False
        >>> featweight_rowid_list1 = ibs.get_annot_featweight_rowids(aid_list, qreq_, ensure)
        >>> print(featweight_rowid_list1)
        >>> ensure = True
        >>> featweight_rowid_list2 = ibs.get_annot_featweight_rowids(aid_list, qreq_, ensure)
        >>> print(featweight_rowid_list2)
        >>> ensure = False
        >>> featweight_rowid_list3 = ibs.get_annot_featweight_rowids(aid_list, qreq_, ensure)
        >>> print(featweight_rowid_list3)
    """
    if ensure:
        # Ensuring dependant columns is equivilant to adding cleanly
        return ibs.add_annot_featweights(aid_list, qreq_=qreq_)
    else:
        # Get leaf_parent rowids
        fid_list = ibs.get_annot_fids(
            aid_list, qreq_=qreq_, ensure=False)
        colnames = (FEATWEIGHT_ROWID,)
        config_rowid = ibs.get_featweight_config_rowid(qreq_=qreq_)
        andwhere_colnames = (FEAT_ROWID, CONFIG_ROWID,)
        params_iter = [(fid, config_rowid,) for fid in fid_list]
        featweight_rowid_list = ibs.dbcache.get_where2(
            constants.FEATURE_WEIGHT_TABLE, colnames, params_iter, andwhere_colnames)
        return featweight_rowid_list

if __name__ == '__main__':
    """
    CommandLine:
        python ibeis\control\_autogen_ibeiscontrol_funcs.py
        python ibeis\control\_autogen_ibeiscontrol_funcs.py --test-get_annot_featweight_rowids
    """
    import utool as ut
    testable_list = [
        get_annot_featweight_rowids
    ]
    ut.doctest_funcs(testable_list)
