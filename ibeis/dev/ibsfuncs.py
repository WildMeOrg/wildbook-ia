# developer convenience functions for ibs
from __future__ import absolute_import, division, print_function
import types
from itertools import izip
from os.path import relpath, split, join, exists
import utool
from ibeis import constants
from ibeis import sysres
from ibeis.export import export_hsdb
from ibeis import constants
from ibeis.control.accessor_decors import getter_1to1

# Inject utool functions
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[ibsfuncs]', DEBUG=False)


__INJECTABLE_FUNCS__ = []


def __injectable(func):
    global __INJECTABLE_FUNCS__
    func_ = utool.indent_func(func)
    __INJECTABLE_FUNCS__.append(func_)
    return func_


@__injectable
def refresh(ibs):
    from ibeis.dev import ibsfuncs
    from ibeis.dev import all_imports
    ibsfuncs.rrr()
    all_imports.reload_all()
    ibsfuncs.inject_ibeis(ibs)


@__injectable
def export_to_hotspotter(ibs):
    export_hsdb.export_ibeis_to_hotspotter(ibs)


@__injectable
def get_image_roi_bboxes(ibs, gid_list):
    rids_list = ibs.get_image_rids(gid_list)
    bboxes_list = ibs.get_unflat_roi_bboxes(rids_list)
    return bboxes_list


@__injectable
def get_image_roi_thetas(ibs, gid_list):
    rids_list = ibs.get_image_rids(gid_list)
    thetas_list = ibs.get_unflat_roi_thetas(rids_list)
    return thetas_list


@__injectable
def compute_all_chips(ibs):
    print('[ibs] compute_all_chips')
    rid_list = ibs.get_valid_rids()
    cid_list = ibs.add_chips(rid_list)
    return cid_list


@__injectable
def compute_all_features(ibs, **kwargs):
    print('[ibs] compute_all_features')
    rid_list = ibs.get_valid_rids(**kwargs)
    cid_list = ibs.get_roi_cids(rid_list, ensure=True)
    fid_list = ibs.add_feats(cid_list)
    return fid_list


@__injectable
def ensure_roi_data(ibs, rid_list, chips=True, feats=True):
    if chips or feats:
        cid_list = ibs.add_chips(rid_list)
    if feats:
        ibs.add_feats(cid_list)


@__injectable
def get_empty_gids(ibs, eid=None):
    """ returns gid list without any chips """
    gid_list = ibs.get_valid_gids(eid=eid)
    nRois_list = ibs.get_image_num_rois(gid_list)
    empty_gids = [gid for gid, nRois in izip(gid_list, nRois_list) if nRois == 0]
    return empty_gids


@__injectable
def convert_empty_images_to_rois(ibs):
    """ images without chips are given an ROI over the entire image """
    gid_list = ibs.get_empty_gids()
    rid_list = ibs.use_images_as_rois(gid_list)
    return rid_list


@__injectable
def use_images_as_rois(ibs, gid_list, name_list=None, nid_list=None,
                       notes_list=None, adjust_percent=0.0):
    """ Adds an roi the size of the entire image to each image.
    adjust_percent - shrinks the ROI by percentage on each side
    """
    pct = adjust_percent  # Alias
    gsize_list = ibs.get_image_sizes(gid_list)
    # Build bounding boxes as images size minus padding
    bbox_list  = [(int( 0 + (gw * pct)),
                   int( 0 + (gh * pct)),
                   int(gw - (gw * pct * 2)),
                   int(gh - (gh * pct * 2)))
                  for (gw, gh) in gsize_list]
    theta_list = [0.0 for _ in xrange(len(gsize_list))]
    rid_list = ibs.add_rois(gid_list, bbox_list, theta_list,
                            name_list=name_list, nid_list=nid_list, notes_list=notes_list)
    return rid_list


@__injectable
def assert_valid_rids(ibs, rid_list):
    valid_rids = set(ibs.get_valid_rids())
    invalid_rids = [rid for rid in rid_list if rid not in valid_rids]
    assert len(invalid_rids) == 0, 'invalid rids: %r' % (invalid_rids,)


@__injectable
def delete_all_features(ibs):
    all_fids = ibs._get_all_fids()
    ibs.delete_features(all_fids)


@__injectable
def delete_all_chips(ibs):
    all_cids = ibs._get_all_cids()
    ibs.delete_chips(all_cids)


@__injectable
def delete_all_encounters(ibs):
    all_eids = ibs._get_all_eids()
    ibs.delete_encounters(all_eids)


@__injectable
def vd(ibs):
    utool.view_directory(ibs.get_dbdir())


@__injectable
def get_roi_desc_cache(ibs, rids):
    """ When you have a list with duplicates and you dont want to copy data
    creates a reference to each data object idnexed by a dict """
    unique_rids = list(set(rids))
    unique_desc = ibs.get_roi_desc(unique_rids)
    desc_cache = dict(list(izip(unique_rids, unique_desc)))
    return desc_cache


@__injectable
def get_roi_is_hard(ibs, rid_list):
    notes_list = ibs.get_roi_notes(rid_list)
    is_hard_list = ['hard' in notes.lower().split() for (notes)
                    in notes_list]
    return is_hard_list


@__injectable
def localize_images(ibs, gid_list=None):
    """
    Moves the images into the ibeis image cache.
    Images are renamed to img_uuid.ext
    """
    if gid_list is None:
        gid_list  = ibs.get_valid_gids()
    gpath_list = ibs.get_image_paths(gid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    gext_list  = ibs.get_image_exts(gid_list)
    # Build list of image names based on uuid in the ibeis imgdir
    guuid_strs = (str(guuid) for guuid in guuid_list)
    loc_gname_list = [guuid + ext for (guuid, ext) in izip(guuid_strs, gext_list)]
    loc_gpath_list = [join(ibs.imgdir, gname) for gname in loc_gname_list]
    utool.copy_list(gpath_list, loc_gpath_list, lbl='Localizing Images: ')
    ibs.set_image_uris(gid_list, loc_gname_list)
    assert all(map(exists, loc_gpath_list)), 'not all images copied'


@__injectable
def delete_invalid_nids(ibs):
    """ Removes names that have no Rois from the database """
    invalid_nids = ibs.get_invalid_nids()
    ibs.delete_names(invalid_nids)


@__injectable
def delete_invalid_eids(ibs):
    """ Removes encounters without images """
    eid_list = ibs.get_valid_eids(min_num_gids=0)
    nGids_list = ibs.get_encounter_num_gids(eid_list)
    is_invalid = [nGids == 0 for nGids in nGids_list]
    invalid_eids = utool.filter_items(eid_list, is_invalid)
    ibs.delete_encounters(invalid_eids)


@__injectable
@getter_1to1
def is_nid_unknown(ibs, nid_list):
    return [nid == ibs.UNKNOWN_NID or nid < 0 for nid in nid_list]


@__injectable
def get_match_truth(ibs, rid1, rid2):
    nid1, nid2 = ibs.get_roi_nids((rid1, rid2))
    isunknown_list = ibs.is_nid_unknown((nid1, nid2))
    if any(isunknown_list):
        truth = 2  # Unknown
    elif nid1 == nid2:
        truth = 1  # True
    elif nid1 != nid2:
        truth = 0  # False
    else:
        raise AssertionError('invalid_unknown_truth_state')
    return truth


def unflat_map(method, unflat_rowids, **kwargs):
    """
    Uses an ibeis lookup function with a non-flat rowid list.
    In essence this is equivilent to map(method, unflat_rowids).
    The utility of this function is that it only calls method once.
    This is more efficient for calls that can take a list of inputs
    """
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = utool.invertable_flatten(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals = method(flat_rowids, **kwargs)
    # Then unflatten the results to the original input dimensions
    unflat_vals = utool.util_list.unflatten(flat_vals, reverse_list)
    return unflat_vals


def unflat_multimap(method_list, unflat_rowids, **kwargs):
    """ unflat_map, but allows multiple methods
    """
    # First flatten the list, and remember the original dimensions
    flat_rowids, reverse_list = utool.invertable_flatten(unflat_rowids)
    # Then preform the lookup / implicit mapping
    flat_vals_list = [method(flat_rowids, **kwargs) for method in method_list]
    # Then unflatten the results to the original input dimensions
    unflat_vals_list = [utool.util_list.unflatten(flat_vals, reverse_list)
                        for flat_vals in flat_vals_list]
    return unflat_vals_list

# TODO: Depricate the lookup names
unflat_lookup = unflat_map


def _make_unflat_getter_func(flat_getter):
    if isinstance(flat_getter, types.MethodType):
        # Unwrap fmethods
        func = flat_getter.im_func
    else:
        func = flat_getter
    func_name = func.func_name
    assert func_name.startswith('get_'), 'only works on getters, not: ' + func_name
    # Create new function
    def unflat_getter(self, unflat_rowids, *args, **kwargs):
        # First flatten the list
        flat_rowids, reverse_list = utool.invertable_flatten(unflat_rowids)
        # Then preform the lookup
        flat_vals = func(self, flat_rowids, *args, **kwargs)
        # Then unflatten the list
        unflat_vals = utool.util_list.unflatten(flat_vals, reverse_list)
        return unflat_vals
    unflat_getter.func_name = func_name.replace('get_', 'get_unflat_')
    return unflat_getter


def inject_ibeis(ibs):
    """ Injects custom functions into an IBEISController """
    # Give the ibeis object the inject_func_as_method
    utool.inject_func_as_method(ibs, utool.inject_func_as_method)
    # List of getters to unflatten
    to_unflatten = [
        ibs.get_roi_uuids,
        ibs.get_image_uuids,
        ibs.get_names,
        ibs.get_image_unixtime,
        ibs.get_roi_bboxes,
        ibs.get_roi_thetas,
    ]
    for flat_getter in to_unflatten:
        unflat_getter = _make_unflat_getter_func(flat_getter)
        ibs.inject_func_as_method(unflat_getter)
    for func in __INJECTABLE_FUNCS__:
        ibs.inject_func_as_method(func)


def delete_ibeis_database(dbdir):
    _ibsdb = join(dbdir, constants.PATH_NAMES._ibsdb)
    print('Deleting _ibsdb=%r' % _ibsdb)
    if exists(_ibsdb):
        utool.delete(_ibsdb)


def assert_valid_names(name_list):
    """ Asserts that user specified names do not conflict with
    the standard unknown name """
    def isconflict(name, other):
        return name.startswith(other) and len(name) > len(other)
    valid_namecheck = [not isconflict(name, constants.UNKNOWN_NAME)
                       for name in name_list]
    assert all(valid_namecheck), ('A name conflicts with UKNONWN Name. -- '
                                  'cannot start a name with four underscores')


def assert_and_fix_gpath_slashes(gpath_list):
    """
    Asserts that all paths are given with forward slashes.
    If not it fixes them
    """
    try:
        msg = ('gpath_list must be in unix format (no backslashes).'
               'Failed on %d-th gpath=%r')
        for count, gpath in enumerate(gpath_list):
            assert gpath.find('\\') == -1, (msg % (count, gpath))
    except AssertionError as ex:
        utool.printex(ex, iswarning=True)
        gpath_list = map(utool.unixpath, gpath_list)
    return gpath_list


def ridstr(rid, ibs=None, notes=False):
    """ Helper to make a string from an RID """
    if not notes:
        return 'rid%d' % (rid,)
    else:
        assert ibs is not None
        notes = ibs.get_roi_notes(rid)
        name  = ibs.get_roi_names(rid)
        return 'rid%d-%r-%r' % (rid, str(name), str(notes))


def vsstr(qrid, rid, lite=False):
    if lite:
        return '%d-vs-%d' % (qrid, rid)
    else:
        return 'qrid%d-vs-rid%d' % (qrid, rid)


def list_images(img_dir, fullpath=True, recursive=True):
    """ lists images that are not in an internal cache """
    ignore_list = ['_hsdb', '.hs_internals', '_ibeis_cache', '_ibsdb']
    gpath_list = utool.list_images(img_dir,
                                   fullpath=fullpath,
                                   recursive=recursive,
                                   ignore_list=ignore_list)
    return gpath_list


def normalize_name(name):
    """
    Maps unknonwn names to the standard ____
    """
    if name in constants.ACCEPTED_UNKNOWN_NAMES:
        name = '____'
    return name


def normalize_names(name_list):
    """
    Maps unknonwn names to the standard ____
    """
    return map(normalize_name, name_list)


def get_names_from_parent_folder(gpath_list, img_dir, fmtkey='name'):
    """
    Input: gpath_list
    Output: names based on the parent folder of each image
    """
    relgpath_list = [relpath(gpath, img_dir) for gpath in gpath_list]
    _name_list  = [split(relgpath)[0] for relgpath in relgpath_list]
    name_list = normalize_names(_name_list)
    return name_list


class FMT_KEYS:
    name_fmt = '{name:*}[id:d].{ext}'
    snails_fmt  = '{name:*dd}{id:dd}.{ext}'


def get_names_from_gnames(gpath_list, img_dir, fmtkey='{name:*}[rid:d].{ext}'):
    """
    Input: gpath_list
    Output: names based on the parent folder of each image
    """
    INGEST_FORMATS = {
        FMT_KEYS.name_fmt: utool.named_field_regex([
            ('name', r'[a-zA-Z]+'),  # all alpha characters
            ('id',   r'\d*'),        # first numbers (if existant)
            ( None,  r'\.'),
            ('ext',  r'\w+'),
        ]),

        FMT_KEYS.snails_fmt: utool.named_field_regex([
            ('name', r'[a-zA-Z]+\d\d'),  # species and 2 numbers
            ('id',   r'\d\d'),  # 2 more numbers
            ( None,  r'\.'),
            ('ext',  r'\w+'),
        ]),
    }
    regex = INGEST_FORMATS.get(fmtkey, fmtkey)
    gname_list = utool.fpaths_to_fnames(gpath_list)
    parsed_list = [utool.regex_parse(regex, gname) for gname in gname_list]

    anyfailed = False
    for gpath, parsed in izip(gpath_list, parsed_list):
        if parsed is None:
            print('FAILED TO PARSE: %r' % gpath)
            anyfailed = True
    if anyfailed:
        msg = ('FAILED REGEX: %r' % regex)
        raise Exception(msg)

    _name_list = [parsed['name'] for parsed in parsed_list]
    name_list = normalize_names(_name_list)
    return name_list


def resolve_name_conflicts(gid_list, name_list):
    # Build conflict map
    conflict_gid_to_names = utool.build_conflict_dict(gid_list, name_list)

    # Check to see which gid has more than one name
    unique_gids = utool.unique_keep_order2(gid_list)
    unique_names = []
    unique_notes = []

    for gid in unique_gids:
        names = utool.unique_keep_order2(conflict_gid_to_names[gid])
        unique_name = names[0]
        unique_note = ''
        if len(names) > 1:
            if '____' in names:
                names.remove('____')
            if len(names) == 1:
                unique_name = names[0]
            else:
                unique_name = names[0]
                unique_note = 'aliases([' + ', '.join(map(repr, names[1:])) + '])'
        unique_names.append(unique_name)
        unique_notes.append(unique_note)

    return unique_gids, unique_names, unique_notes


def make_roi_uuids(image_uuid_list, bbox_list, theta_list, deterministic=True):
    augment_uuid = utool.util_hash.augment_uuid
    random_uuid = utool.util_hash.random_uuid
    try:
        # Check to make sure bbox input is a tuple-list, not a list-list
        if len(bbox_list) > 0:
            try:
                assert isinstance(bbox_list[0], tuple), 'Bounding boxes must be tuples of ints!'
                assert isinstance(bbox_list[0][0], int), 'Bounding boxes must be tuples of ints!'
            except AssertionError as ex:
                utool.printex(ex)
                print('bbox_list = %r' % (bbox_list,))
                raise
        roi_uuid_list = [augment_uuid(img_uuid, bbox, theta)
                         for img_uuid, bbox, theta
                         in izip(image_uuid_list, bbox_list, theta_list)]
        if not deterministic:
            # Augment determenistic uuid with a random uuid to ensure randomness
            # (this should be ensured in all hardward situations)
            roi_uuid_list = [augment_uuid(random_uuid(), _uuid)
                             for _uuid in roi_uuid_list]
    except Exception as ex:
        utool.printex(ex, 'Error building roi_uuids', '[add_roi]',
                      key_list=['image_uuid_list'])
        raise
    return roi_uuid_list


def get_species_dbs(species_prefix):
    ibs_dblist = sysres.get_ibsdb_list()
    isvalid_list = [split(path)[1].startswith(species_prefix) for path in ibs_dblist]
    return utool.filter_items(ibs_dblist, isvalid_list)


def merge_species_databases(species_prefix):
    """ Build a merged database """
    from ibeis.control import IBEISControl
    print('[ibsfuncs] Merging species with prefix: %r' % species_prefix)
    utool.util_parallel.ensure_pool(warn=False)
    with utool.Indenter('    '):
        # Build / get target database
        all_db = '__ALL_' + species_prefix + '_'
        all_dbdir = sysres.db_to_dbdir(all_db, allow_newdir=True)
        ibs_target = IBEISControl.IBEISController(all_dbdir)
        # Build list of databases to merge
        species_dbdir_list = get_species_dbs(species_prefix)
        ibs_source_list = []
        for dbdir in species_dbdir_list:
            ibs_source = IBEISControl.IBEISController(dbdir)
            ibs_source_list.append(ibs_source)
    print('[ibsfuncs] Destination database: %r' % all_db)
    print('[ibsfuncs] Source databases:' +
          utool.indentjoin(species_dbdir_list, '\n *   '))
    #Merge the databases into ibs_target
    merge_databases(ibs_target, ibs_source_list)
    return ibs_target


def merge_databases(ibs_target, ibs_source_list):
    """ Merges a list of databases into a target """

    def merge_images(ibs_target, ibs_source):
        """ merge image helper """
        gid_list1   = ibs_source.get_valid_gids()
        uuid_list1  = ibs_source.get_image_uuids(gid_list1)
        gpath_list1 = ibs_source.get_image_paths(gid_list1)
        aif_list1   = ibs_source.get_image_aifs(gid_list1)
        # Add images to target
        ibs_target.add_images(gpath_list1)
        # Merge properties
        gid_list2  = ibs_target.get_image_gids_from_uuid(uuid_list1)
        ibs_target.set_image_aifs(gid_list2, aif_list1)

    def merge_rois(ibs_target, ibs_source):
        """ merge rois helper """
        rid_list1   = ibs_source.get_valid_rids()
        uuid_list1  = ibs_source.get_roi_uuids(rid_list1)
        # Get the images in target_db
        gid_list1   = ibs_source.get_roi_gids(rid_list1)
        bbox_list1  = ibs_source.get_roi_bboxes(rid_list1)
        theta_list1 = ibs_source.get_roi_thetas(rid_list1)
        name_list1  = ibs_source.get_roi_names(rid_list1, distinguish_unknowns=False)
        notes_list1 = ibs_source.get_roi_notes(rid_list1)

        image_uuid_list1 = ibs_source.get_image_uuids(gid_list1)
        gid_list2  = ibs_target.get_image_gids_from_uuid(image_uuid_list1)
        image_uuid_list2 = ibs_target.get_image_uuids(gid_list2)
        # Assert that the image uuids have not changed
        assert image_uuid_list1 == image_uuid_list2, 'error merging roi image uuids'
        rid_list2 = ibs_target.add_rois(gid_list2,
                                        bbox_list1,
                                        theta_list=theta_list1,
                                        name_list=name_list1,
                                        notes_list=notes_list1)
        uuid_list2 = ibs_target.get_roi_uuids(rid_list2)
        assert uuid_list2 == uuid_list1, 'error merging roi uuids'

    # Do the merging
    for ibs_source in ibs_source_list:
        try:
            print('Merging ' + ibs_source.get_dbname() +
                  ' into ' + ibs_target.get_dbname())
            merge_images(ibs_target, ibs_source)
            merge_rois(ibs_target, ibs_source)
        except Exception as ex:
            utool.printex(ex, 'error merging ' + ibs_source.get_dbname() +
                          ' into ' + ibs_target.get_dbname())


def delete_non_exemplars(ibs):
    rid_list_ = ibs.get_valid_rids()
    examplar_flag_list = ibs.get_roi_exemplar_flag(rid_list_)
    nonexemplar_flag_list = [not is_exemplar for is_exemplar in exemplar_flags]
    rid_list = utool.filter_items(rid_list, nonexemplar_flag_list)
    nid_list = ibs.get_roi_nids(rid_list)
    gid_list = ibs.get_roi_gids(rid_list)
    fid_list = ibs.get_roi_fids(rid_list)
    ibs.delete_features(fid_list)
    ibs.delete_images(gid_list)
    ibs.delete_names(nid_list)
    all_eids = ibs.get_valid_eids()
    rids_list = ibs.get_encounter_rids(all_eids)
    eid_list_ = [eid if len(rids_list[x]) > 0 else None for x, eid in enumerate(rids_list)]
    eid_list = utool.filter_Nones(eid_list)
    ibs.delete_encounters(eid_list)


def get_title(ibs):
    if ibs is None:
        title = 'IBEIS - No Database Directory Open'
    elif ibs.dbdir is None:
        title = 'IBEIS - !! INVALID DATABASE !!'
    else:
        dbdir = ibs.get_dbdir()
        dbname = ibs.get_dbname()
        title = 'IBEIS - %r - Database Directory = %s' % (dbname, dbdir)
    return title


@__injectable
def print_stats(ibs):
    from ibeis.dev import dbinfo
    dbinfo.dbstats(ibs)


@__injectable
def get_infostr(ibs):
    """ Returns printable database information """
    dbname = ibs.get_dbname()
    workdir = utool.unixpath(ibs.get_workdir())
    num_images = ibs.get_num_images()
    num_rois = ibs.get_num_rois()
    num_names = ibs.get_num_names()
    infostr = '''
    workdir = %r
    dbname = %r
    num_images = %r
    num_rois = %r
    num_names = %r
    ''' % (workdir, dbname, num_images, num_rois, num_names)
    return infostr


@__injectable
def print_roi_table(ibs):
    """ Dumps roi table to stdout """
    print('\n')
    print(ibs.db.get_table_csv('rois', exclude_columns=['roi_uuid', 'roi_verts']))


@__injectable
def print_chip_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv('chips'))


@__injectable
def print_feat_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv('features', exclude_columns=[
        'feature_keypoints', 'feature_sifts']))


@__injectable
def print_image_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv('images'))
    #, exclude_columns=['image_rowid']))


@__injectable
def print_label_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv('labels'))


@__injectable
def print_rlr_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.AL_RELATION_TABLE))


@__injectable
def print_config_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.CONFIG_TABLE))


@__injectable
def print_encounter_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.ENCOUNTER_TABLE))


@__injectable
def print_egpairs_table(ibs):
    """ Dumps chip table to stdout """
    print('\n')
    print(ibs.db.get_table_csv(constants.EG_RELATION_TABLE))


@__injectable
def print_tables(ibs, exclude_columns=None, exclude_tables=None):
    if exclude_columns is None:
        exclude_columns = ['annot_uuid', 'label_uuid', 'annot_verts', 'feature_keypoints',
                           'feature_sifts', 'image_uuid', 'image_uri']
    if exclude_tables is None:
        exclude_tables = ['masks', 'recognitions', 'chips', 'features']
    for table_name in ibs.db.get_table_names():
        if table_name in exclude_tables:
            continue
        print('\n')
        print(ibs.db.get_table_csv(table_name, exclude_columns=exclude_columns))
    #ibs.print_image_table()
    #ibs.print_roi_table()
    #ibs.print_labels_table()
    #ibs.print_rlr_table()
    #ibs.print_config_table()
    #ibs.print_chip_table()
    #ibs.print_feat_table()
    print('\n')


def make_new_name(ibs):
    new_name = 'name_%d' % ibs.get_num_names()
    return new_name


#@getter_1to1
@__injectable
def is_rid_unknown(ibs, rid_list):
    nid_list = ibs.get_roi_nids(rid_list)
    return ibs.is_nid_unknown(nid_list)
