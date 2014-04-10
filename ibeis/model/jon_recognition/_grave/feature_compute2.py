''' Computes feature representations '''
from __future__ import absolute_import, division, print_function
import utool
(print, print_,
 printDBG, rrr, profile) = utool.inject(__name__, '[fc2]', DEBUG=False)
# scientific
import numpy as np
# python
from os.path import join
# IBEIS
from ibeis.dev import params
import pyhesaff


# =======================================
# Main Script
# =======================================
@profile
def bigcache_feat_save(cache_dir, uid, ext, kpts_list, desc_list):
    print('[fc2] Caching desc_list and kpts_list')
    utool.smart_save(kpts_list, cache_dir, 'kpts_list', uid, ext)
    utool.smart_save(desc_list, cache_dir, 'desc_list', uid, ext)


@profile
def bigcache_feat_load(cache_dir, uid, ext):
    #io.debug_smart_load(cache_dir, fname='*', uid=uid, ext='.*')
    kpts_list = utool.smart_load(cache_dir, 'kpts_list', uid, ext, can_fail=True)
    desc_list = utool.smart_load(cache_dir, 'desc_list', uid, ext, can_fail=True)
    if desc_list is None or kpts_list is None:
        return None
    desc_list = desc_list.tolist()
    kpts_list = kpts_list.tolist()
    print('[fc2]  Loaded kpts_list and desc_list from big cache')
    return kpts_list, desc_list


@profile
def sequential_feat_load(feat_cfg, feat_fpath_list):
    kpts_list = []
    desc_list = []
    # Debug loading (seems to use lots of memory)
    print('\n')
    try:
        nFeats = len(feat_fpath_list)
        prog_label = '[fc2] Loading feature: '
        mark_progress, end_progress = utool.progress_func(nFeats, prog_label)
        for count, feat_path in enumerate(feat_fpath_list):
            try:
                npz = np.load(feat_path, mmap_mode=None)
            except IOError:
                print('\n')
                utool.checkpath(feat_path, verbose=True)
                print('IOError on feat_path=%r' % feat_path)
                raise
            kpts = npz['arr_0']
            desc = npz['arr_1']
            npz.close()
            kpts_list.append(kpts)
            desc_list.append(desc)
            mark_progress(count)
        end_progress()
        print('[fc2] Finished load of individual kpts and desc')
    except MemoryError:
        print('\n------------')
        print('[fc2] Out of memory')
        print('[fc2] Trying to read: %r' % feat_path)
        print('[fc2] len(kpts_list) = %d' % len(kpts_list))
        print('[fc2] len(desc_list) = %d' % len(desc_list))
        raise
    return kpts_list, desc_list


def precompute_hesaff(rchip_fpath, feat_fpath, dict_args):
    # Calls the function which reads the chip and computes features
    kpts, desc = pyhesaff.detect_kpts(rchip_fpath, dict_args)
    # Saves the features to the feature cache dir
    np.savez(feat_fpath, kpts, desc)
    return None  # kpts, desc


def _cx2_feat_fpaths(ibs, cid_list):
    feat_dir = ibs.dirs.feat_dir
    feat_cfg = ibs.prefs.feat_cfg
    feat_uid = feat_cfg.get_uid()
    cid_list = ibs.tables.cid2_cid[cid_list]
    feat_fname_fmt = ''.join(('cid%d', feat_uid, '.npz'))
    feat_fpath_fmt = join(feat_dir, feat_fname_fmt)
    feat_fpath_list = [feat_fpath_fmt % cid for cid in cid_list]
    return feat_fpath_list


@profile
def _load_features_individualy(ibs, cid_list):
    use_cache = not params.args.nocache_feats
    feat_cfg = ibs.prefs.feat_cfg
    feat_uid = feat_cfg.get_uid()
    print('[fc2]  Loading ' + feat_uid + ' individually')
    # Build feature paths
    rchip_fpath_list = [ibs.cpaths.cid2_rchip_path[cid] for cid in iter(cid_list)]
    feat_fpath_list = _cx2_feat_fpaths(ibs, cid_list)
    #feat_fname_list = [feat_fname_fmt % cid for cid in cid_list]
    # Compute features in parallel, saving them to disk
    kwargs_list = [feat_cfg.get_dict_args()] * len(rchip_fpath_list)
    pfc_kwargs = {
        'func': precompute_hesaff,
        'arg_list': [rchip_fpath_list, feat_fpath_list, kwargs_list],
        'num_procs': params.args.num_procs,
        'lazy': use_cache,
    }
    utool.parallel_compute(**pfc_kwargs)
    # Load precomputed features sequentially
    kpts_list, desc_list = sequential_feat_load(feat_cfg, feat_fpath_list)
    return kpts_list, desc_list


@profile
def _load_features_bigcache(ibs, cid_list):
    # args for smart load/save
    feat_cfg = ibs.prefs.feat_cfg
    feat_uid = feat_cfg.get_uid()
    cache_dir  = ibs.dirs.cache_dir
    sample_uid = utool.hashstr_arr(cid_list, 'cids')
    bigcache_uid = '_'.join((feat_uid, sample_uid))
    ext = '.npy'
    loaded = bigcache_feat_load(cache_dir, bigcache_uid, ext)
    if loaded is not None:  # Cache Hit
        kpts_list, desc_list = loaded
    else:  # Cache Miss
        kpts_list, desc_list = _load_features_individualy(ibs, cid_list)
        # Cache all the features
        bigcache_feat_save(cache_dir, bigcache_uid, ext, kpts_list, desc_list)
    return kpts_list, desc_list


@profile
@utool.indent_decor('[fc2]')
def load_features(ibs, cid_list=None, **kwargs):
    # TODO: There needs to be a fast way to ensure that everything is
    # already loaded. Same for cc2.
    print('=============================')
    print('[fc2] Precomputing and loading features: %r' % ibs.get_db_name())
    #----------------
    # COMPUTE SETUP
    #----------------
    use_cache = not params.args.nocache_feats
    use_big_cache = use_cache and cid_list is None
    feat_cfg = ibs.prefs.feat_cfg
    feat_uid = feat_cfg.get_uid()
    if ibs.feats.feat_uid != '' and ibs.feats.feat_uid != feat_uid:
        print('[fc2] Disagreement: OLD_feat_uid = %r' % ibs.feats.feat_uid)
        print('[fc2] Disagreement: NEW_feat_uid = %r' % feat_uid)
        print('[fc2] Unloading all chip information')
        ibs.unload_all()
        ibs.load_chips(cid_list=cid_list)
    print('[fc2] feat_uid = %r' % feat_uid)
    # Get the list of chip features to load
    cid_list = ibs.get_valid_cxs() if cid_list is None else cid_list
    if not np.iterable(cid_list):
        cid_list = [cid_list]
    print('[cc2] len(cid_list) = %r' % len(cid_list))
    if len(cid_list) == 0:
        return  # HACK
    cid_list = np.array(cid_list)  # HACK
    if use_big_cache:  # use only if all descriptors requested
        kpts_list, desc_list = _load_features_bigcache(ibs, cid_list)
    else:
        kpts_list, desc_list = _load_features_individualy(ibs, cid_list)
    # Extend the datastructure if needed
    list_size = max(cid_list) + 1
    utool.ensure_list_size(ibs.feats.cid2_kpts, list_size)
    utool.ensure_list_size(ibs.feats.cid2_desc, list_size)
    # Copy the values into the ChipPaths object
    for lx, cid in enumerate(cid_list):
        ibs.feats.cid2_kpts[cid] = kpts_list[lx]
    for lx, cid in enumerate(cid_list):
        ibs.feats.cid2_desc[cid] = desc_list[lx]
    ibs.feats.feat_uid = feat_uid
    print('[fc2]=============================')


def clear_feature_cache(ibs):
    feat_cfg = ibs.prefs.feat_cfg
    feat_dir = ibs.dirs.feat_dir
    cache_dir = ibs.dirs.cache_dir
    feat_uid = feat_cfg.get_uid()
    print('[fc2] clearing feature cache: %r' % feat_dir)
    utool.remove_files_in_dir(feat_dir, '*' + feat_uid + '*', verbose=True, dryrun=False)
    utool.remove_files_in_dir(cache_dir, '*' + feat_uid + '*', verbose=True, dryrun=False)
    pass
