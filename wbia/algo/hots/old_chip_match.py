# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six

print, rrr, profile = ut.inject2(__name__)


class AlignedListDictProxy(ut.DictLike_old):
    """
    simulates a dict when using parallel lists the point of this class is that
    when there are many instances of this class, then key2_idx can be shared between
    them. Ideally this class wont be used and will disappear when the parallel
    lists are being used properly.

    DEPCIRATE AlignedListDictProxy's defaultdict behavior is weird
    """

    def __init__(self, key2_idx, key_list, val_list):
        # if isinstance(key_list, np.ndarray):
        #    key_list = key_list.tolist()
        self.key_list = key_list
        self.val_list = val_list
        self.key2_idx = key2_idx
        self.default_function = None

    def __eq__(self, key):
        raise NotImplementedError()

    def pop(self, key):
        raise NotImplementedError()

    def __getitem__(self, key):
        try:
            idx = self.key2_idx[key]
        except (KeyError, IndexError):
            if self.default_function is not None:
                # behave like a default dict here
                self[key] = self.default_function()
                return self[key]
            else:
                raise
        return self.val_list[idx]
        # return ut.take(self.val_list, ut.dict_take(self.key2_idx, key))

    def __setitem__(self, key, val):
        try:
            idx = self.key2_idx[key]
        except KeyError:
            idx = len(self.key_list)
            self.key_list.append(key)
            self.key2_idx[key] = idx
        try:
            self.val_list[idx] = val
        except IndexError:
            if idx == len(self.val_list):
                self.val_list.append(val)
            else:
                raise
            # else:
            #    offset = idx - len(self.val_list)
            #    self.val_list.extend(([None] * offset) + [val])

    def iteritems(self):
        for key, val in zip(self.key_list, self.val_list):
            yield key, val

    def iterkeys(self):
        return iter(self.key_list)

    def itervalues(self):
        return iter(self.val_list)


class _OldStyleChipMatchSimulator(object):
    # SIMULATE OLD CHIPMATCHES UNTIL TRANSFER IS COMPLETE
    # TRY NOT TO USE THESE AS THEY WILL BE MUCH SLOWER THAN
    # NORMAL.
    _oldfields = ('aid2_fm', 'aid2_fsv', 'aid2_fk', 'aid2_score', 'aid2_H')

    def __iter__(cm):
        for field in cm._oldfields:
            yield getattr(cm, field)

    def __getitem__(cm, index):
        if isinstance(index, six.string_types):
            return cm.__dict__[index]
        else:
            return getattr(cm, cm._oldfields[index])

    @property
    def aid2_fm(cm):
        return AlignedListDictProxy(cm.daid2_idx, cm.daid_list, cm.fm_list)

    @property
    def aid2_fsv(cm):
        return AlignedListDictProxy(cm.daid2_idx, cm.daid_list, cm.fsv_list)

    @property
    def aid2_fk(cm):
        return AlignedListDictProxy(cm.daid2_idx, cm.daid_list, cm.fk_list)

    @property
    def aid2_H(cm):
        return (
            None
            if cm.H_list is None
            else AlignedListDictProxy(cm.daid2_idx, cm.daid_list, cm.H_list)
        )

    @property
    def aid2_score(cm):
        return (
            {}
            if cm.score_list is None
            else AlignedListDictProxy(cm.daid2_idx, cm.daid_list, cm.score_list)
        )

    @property
    def aid2_fs(cm):
        if cm.fs_list is None:
            fs_list = cm.get_fsv_prod_list()
        else:
            fs_list = cm.fs_list
        return AlignedListDictProxy(cm.daid2_idx, cm.daid_list, fs_list)

    # qres compatibility

    # @property
    # def filtkey_list(cm):
    #     """ for compatibility with qres """
    #     return cm.fsv_col_lbls

    # @property
    # def nid2_name_score(cm):
    #     """
    #     DEPCIRATE AlignedListDictProxy's defaultdict behavior is weird

    #     """
    #     return ({} if cm.score_list is None else
    #             AlignedListDictProxy(cm.nid2_nidx, cm.unique_nids, cm.name_score_list))

    # @property
    # def aid2_annot_score(cm):
    #     """ DEPCIRATE AlignedListDictProxy's defaultdict behavior is weird

    #     ute cm.get_annot_scores instead

    #     """
    #     return ({} if cm.annot_score_list is None else
    #             AlignedListDictProxy(cm.daid2_idx, cm.daid_list, cm.annot_score_list))

    # def get_nscoretup(cm):
    #     return cm.get_ranked_nids_and_aids()

    # def tokwargs(cm):
    #     """
    #     Can be unpacked and passed as kwargs
    #     **cm.tokwargs()
    #     """
    #     return ut.KwargsWrapper(cm)
