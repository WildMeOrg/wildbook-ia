# developer convenience functions for ibs
from __future__ import absolute_import, division, print_function
import types
from itertools import izip
from os.path import relpath, split, join, exists
import utool
from ibeis import constants
from ibeis import sysres
from ibeis.export import export_hsdb

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
def get_image_bboxes(ibs, gid_list):
    size_list = ibs.get_image_sizes(gid_list)
    bbox_list  = [(0, 0, w, h) for (w, h) in size_list]
    return bbox_list


@__injectable
def compute_all_chips(ibs):
    print('[ibs] compute_all_chips')
    rid_list = ibs.get_valid_rids()
    cid_list = ibs.add_chips(rid_list)
    return cid_list


@__injectable
def compute_all_features(ibs):
    print('[ibs] compute_all_features')
    rid_list = ibs.get_valid_rids()
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
def get_empty_gids(ibs):
    """ returns gid list without any chips """
    gid_list = ibs.get_valid_gids()
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
    if gid_list is None:
        gid_list  = ibs.get_valid_gids()
    gpath_list = ibs.get_image_paths(gid_list)
    guuid_list = ibs.get_image_uuids(gid_list)
    gext_list  = ibs.get_image_exts(gid_list)
    # Build list of image names based on uuid in the ibeis imgdir
    local_gname_list = [str(guuid) + ext for guuid, ext, in izip(guuid_list, gext_list)]
    local_gpath_list = [join(ibs.imgdir, gname) for gname in local_gname_list]
    utool.copy_list(gpath_list, local_gpath_list, lbl='Localizing Images: ')
    ibs.set_image_uris(gid_list, local_gname_list)

    assert all(map(exists, local_gpath_list)), 'not all images copied'


@__injectable
def delete_invalid_nids(ibs):
    """ Removes names that have no Rois from the database """
    invalid_nids = ibs.get_invalid_nids()
    ibs.delete_names(invalid_nids)


def unflat_lookup(method, unflat_uids, **kwargs):
    """ Uses an ibeis lookup function with a non-flat uid list.
    """
    # First flatten the list
    flat_uids, reverse_list = utool.invertable_flatten(unflat_uids)
    # Then preform the lookup
    flat_vals = method(flat_uids, **kwargs)
    # Then unflatten the list
    unflat_vals = utool.util_list.unflatten(flat_vals, reverse_list)
    return unflat_vals


def _make_unflat_getter_func(flat_getter):
    if isinstance(flat_getter, types.MethodType):
        # Unwrap fmethods
        func = flat_getter.im_func
    else:
        func = flat_getter
    func_name = func.func_name
    assert func_name.startswith('get_'), 'only works on getters, not: ' + func_name
    # Create new function
    def unflat_getter(self, unflat_uids, *args, **kwargs):
        # First flatten the list
        flat_uids, reverse_list = utool.invertable_flatten(unflat_uids)
        # Then preform the lookup
        flat_vals = func(self, flat_uids, *args, **kwargs)
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
        ibs.get_image_unixtime
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
    INJEST_FORMATS = {
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
    regex = INJEST_FORMATS.get(fmtkey, fmtkey)
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


def make_roi_uuids(image_uuid_list, bbox_list, theta_list):
    try:
        # Check to make sure bbox input is a tuple-list, not a list-list
        if len(bbox_list) > 0:
            assert isinstance(bbox_list[0], tuple), 'Bounding boxes must be tuples of ints!'
            assert isinstance(bbox_list[0][0], int), 'Bounding boxes must be tuples of ints!'
        roi_uuid_list = [utool.util_hash.augment_uuid(img_uuid, bbox, theta)
                            for img_uuid, bbox, theta
                            in izip(image_uuid_list, bbox_list, theta_list)]
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
