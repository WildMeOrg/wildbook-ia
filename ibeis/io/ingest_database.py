#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
"""
This module lists known raw databases and how to ingest them.
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, map, range
import ibeis
from os.path import relpath, split, exists, join
from ibeis import ibsfuncs
from ibeis import constants
import utool
import utool as ut  # NOQA:
import parse


def normalize_name(name):
    """
    Maps unknonwn names to the standard ____
    """
    if name in constants.ACCEPTED_UNKNOWN_NAMES:
        name = constants.INDIVIDUAL_KEY
    return name


def normalize_names(name_list):
    """
    Maps unknonwn names to the standard ____
    """
    return list(map(normalize_name, name_list))


def get_name_text_from_parent_folder(gpath_list, img_dir, fmtkey='name'):
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
    giraffe1_fmt = '{name:*}_{id:d}.{ext}'


def get_name_text_from_gnames(gpath_list, img_dir, fmtkey='{name:*}[aid:d].{ext}'):
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

        FMT_KEYS.giraffe1_fmt: utool.named_field_regex([
            ('name',  r'G\d+'),  # species and 2 numbers
            ('under', r'_'),     # 2 more numbers
            ('id',    r'\d+'),   # 2 more numbers
            ( None,   r'\.'),
            ('ext',   r'\w+'),
        ]),
    }
    regex = INGEST_FORMATS.get(fmtkey, fmtkey)
    gname_list = utool.fpaths_to_fnames(gpath_list)
    parsed_list = [utool.regex_parse(regex, gname) for gname in gname_list]

    anyfailed = False
    for gpath, parsed in zip(gpath_list, parsed_list):
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


#
#
### <STANDARD DATABASES> ###

STANDARD_INGEST_FUNCS = {}


def __standard(dbname):
    """  Decorates a function as a standard ingestable database """
    def __registerdb(func):
        STANDARD_INGEST_FUNCS[dbname] = func
        return func
    return __registerdb


@__standard('polar_bears')
def ingest_polar_bears(dbname):
    return Ingestable(dbname, ingest_type='named_folders',
                      adjust_percent=0.00,
                      fmtkey='name')


@__standard('testdb1')
def ingest_testdb1(dbname):
    """
    ingest_testdb1

    Example:
        >>> # DOCTEST = OFF
        >>> from ibeis.io.ingest_database import *  # NOQA
        >>> from
        >>> import utool
        >>> from vtool.tests import grabdata
        >>> grabdata.ensure_testdata()
        >>> # DELETE TESTDB1
        >>> TESTDB1 = join(ibeis.sysres.get_workdir(), 'testdb1')
        >>> utool.delete(TESTDB1, ignore_errors=False)
        >>>
        >>> result = ingest_testdb1(dbname)
    """
    from vtool.tests import grabdata
    def postingest_tesdb1_func(ibs):
        print('postingest_tesdb1_func')
        # Adjust data as we see fit
        import numpy as np
        gid_list = np.array(ibs.get_valid_gids())
        # Set image unixtimes
        unixtimes_even = (gid_list[0::2] + 100).tolist()
        unixtimes_odd  = (gid_list[1::2] + 9001).tolist()
        unixtime_list = unixtimes_even + unixtimes_odd
        ibs.set_image_unixtime(gid_list, unixtime_list)
        # Unname first aid in every name
        aid_list = ibs.get_valid_aids()
        nid_list = ibs.get_annot_nids(aid_list)
        nid_list = [ (nid if nid > 0 else None) for nid in nid_list]
        unique_flag = utool.flag_unique_items(nid_list)
        unique_nids = utool.filter_items(nid_list, unique_flag)
        none_nids = [ nid is not None for nid in nid_list]
        flagged_nids = [nid for nid in unique_nids if nid_list.count(nid) > 1]
        plural_flag = [nid in flagged_nids for nid in nid_list]
        flag_list = list(map(all, zip(plural_flag, unique_flag, none_nids)))
        flagged_aids = utool.filter_items(aid_list, flag_list)
        if utool.VERYVERBOSE:
            def print2(*args):
                print('[post_testdb1] ' + ', '.join(args))
            print2('aid_list=%r' % aid_list)
            print2('nid_list=%r' % nid_list)
            print2('unique_flag=%r' % unique_flag)
            print2('plural_flag=%r' % plural_flag)
            print2('unique_nids=%r' % unique_nids)
            print2('none_nids=%r' % none_nids)
            print2('flag_list=%r' % flag_list)
            print2('flagged_nids=%r' % flagged_nids)
            print2('flagged_aids=%r' % flagged_aids)
            # print2('new_nids=%r' % new_nids)
        # Unname, some annotations for testing
        delete_aids = utool.filter_items(aid_list, flag_list)
        ibs.delete_annot_nids(delete_aids)
        # Add all annotations with names as exemplars
        from ibeis.control.IBEISControl import IBEISController
        assert isinstance(ibs, IBEISController)
        unflagged_aids = utool.get_dirty_items(aid_list, flag_list)
        exemplar_flags = [True] * len(unflagged_aids)
        ibs.set_annot_exemplar_flag(unflagged_aids, exemplar_flags)
        # Set some test species labels
        from ibeis.constants import Species
        species_list = ibs.get_annot_species(aid_list)
        for ix in range(0, 6):
            species_list[ix] = Species.ZEB_PLAIN
        # These are actually plains zebras.
        for ix in range(8, 10):
            species_list[ix] = Species.ZEB_GREVY
        for ix in range(10, 12):
            species_list[ix] = Species.POLAR_BEAR
        ibs.set_annot_species(aid_list, species_list)
        ibs.set_annot_notes(aid_list[8:10], ['this is actually a plains zebra'] * 2)
        ibs.set_annot_notes(aid_list[0:1], ['aid 1 and 2 are correct matches'])
        ibs.set_annot_notes(aid_list[6:7], ['very simple image to debug feature detector'])
        ibs.set_annot_notes(aid_list[7:8], ['standard test image'])
        #ut.embed()
        return None
    return Ingestable(dbname, ingest_type='named_images',
                      fmtkey=FMT_KEYS.name_fmt,
                      img_dir=grabdata.get_testdata_dir(),
                      adjust_percent=0.00,
                      postingest_func=postingest_tesdb1_func)


@__standard('snails_drop1')
def ingest_snails_drop1(dbname):
    return Ingestable(dbname,
                      ingest_type='named_images',
                      fmtkey=FMT_KEYS.snails_fmt,
                      #img_dir='/raid/raw/snails_drop1_59MB',
                      adjust_percent=.20)


@__standard('JAG_Kieryn')
def ingest_JAG_Kieryn(dbname):
    return Ingestable(dbname,
                      ingest_type='unknown',
                      adjust_percent=0.00)


@__standard('Giraffes')
def ingest_Giraffes1(dbname):
    return Ingestable(dbname,
                      ingest_type='named_images',
                      fmtkey=FMT_KEYS.giraffe1_fmt,
                      adjust_percent=0.00)


def get_standard_ingestable(dbname):
    if dbname in STANDARD_INGEST_FUNCS:
        return STANDARD_INGEST_FUNCS[dbname](dbname)
    else:
        raise AssertionError('Unknown dbname=%r' % (dbname,))


def ingest_standard_database(dbname, force_delete=False):
    """
    ingest_standard_database

    Args:
        dbname (str): database name
        force_delete (bool):

    Example:
        >>> from ibeis.io.ingest_database import *  # NOQA
        >>> dbname = 'testdb1'
        >>> force_delete = False
        >>> result = ingest_standard_database(dbname, force_delete)
        >>> print(result)
    """
    from ibeis.control import IBEISControl
    print('[ingest] Ingest Standard Database: dbname=%r' % (dbname,))
    ingestable = get_standard_ingestable(dbname)
    dbdir = ibeis.sysres.db_to_dbdir(ingestable.dbname, allow_newdir=True, use_sync=False)
    utool.ensuredir(dbdir, verbose=True)
    if force_delete:
        ibsfuncs.delete_ibeis_database(dbdir)
    ibs = IBEISControl.IBEISController(dbdir)
    ingest_rawdata(ibs, ingestable)

### </STANDARD DATABASES> ###
#
#


class Ingestable(object):
    """
    Temporary structure representing how to ingest a databases
    """
    def __init__(self, dbname, img_dir=None, ingest_type=None, fmtkey=None,
                 adjust_percent=0.0, postingest_func=None):
        self.dbname          = dbname
        self.img_dir         = img_dir
        self.ingest_type     = ingest_type
        self.fmtkey          = fmtkey
        self.adjust_percent  = adjust_percent
        self.postingest_func = postingest_func
        self.ensure_feasibility()

    def ensure_feasibility(self):
        rawdir  = ibeis.sysres.get_rawdir()
        if self.img_dir is None:
            # Try to find data either the raw or work dir
            self.img_dir = ibeis.sysres.db_to_dbdir(self.dbname, extra_workdirs=[rawdir])
        msg = 'Cannot find img_dir for dbname=%r, img_dir=%r' % (self.dbname, self.img_dir)
        assert self.img_dir is not None, msg
        assert exists(self.img_dir), msg
        if self.ingest_type == 'named_folders':
            assert self.fmtkey == 'name'


def ingest_rawdata(ibs, ingestable, localize=False):
    """
    Ingests rawdata into an ibeis database.

    if ingest_type == 'named_folders':
        Converts folder structure where folders = name, to ibsdb
    if ingest_type == 'named_images':
        Converts imgname structure where imgnames = name_id.ext, to ibsdb
    """
    img_dir         = ingestable.img_dir
    ingest_type     = ingestable.ingest_type
    fmtkey          = ingestable.fmtkey
    adjust_percent  = ingestable.adjust_percent
    postingest_func = ingestable.postingest_func
    print('[ingest] ingesting rawdata: img_dir=%r, injest_type=%r' % (img_dir, ingest_type))
    # Get images in the image directory
    gpath_list  = ibsfuncs.list_images(img_dir)
    # Parse structure for image names
    if ingest_type == 'named_folders':
        name_list = get_name_text_from_parent_folder(gpath_list, img_dir, fmtkey)
        pass
    if ingest_type == 'named_images':
        name_list = get_name_text_from_gnames(gpath_list, img_dir, fmtkey)
    if ingest_type == 'unknown':
        name_list = [constants.INDIVIDUAL_KEY for _ in range(len(gpath_list))]

    # Add Images
    gid_list_ = ibs.add_images(gpath_list)
    # <DEBUG>
    #print('added: ' + utool.indentjoin(map(str, zip(gid_list_, gpath_list))))
    unique_gids = list(set(gid_list_))
    print("[ingest] Length gid list: %d" % len(gid_list_))
    print("[ingest] Length unique gid list: %d" % len(unique_gids))
    assert len(gid_list_) == len(gpath_list)
    for gid in gid_list_:
        if gid is None:
            print('[ingest] big fat warning')
    # </DEBUG>
    gid_list = utool.filter_Nones(gid_list_)
    unique_gids, unique_names, unique_notes = resolve_name_conflicts(
        gid_list, name_list)
    # Add ANNOTATIONs with names and notes
    aid_list = ibs.use_images_as_annotations(unique_gids,
                                             name_list=unique_names,
                                             notes_list=unique_notes,
                                             adjust_percent=adjust_percent)
    if localize:
        ibs.localize_images()
    if postingest_func is not None:
        postingest_func(ibs)
    # Print to show success
    #ibs.print_image_table()
    #ibs.print_tables()
    #ibs.print_annotation_table()
    #ibs.print_alr_table()
    #ibs.print_lblannot_table()
    #ibs.print_image_table()
    return aid_list


def ingest_oxford_style_db(dbdir):
    """

    >>> from ibeis.io.ingest_database import *  # NOQA
    >>> import ibeis
    >>> dbdir = '/raid/work/Oxford'
    >>> dbdir = '/raid/work/Paris'
    >>>
    #>>> ibeis.io.convert_db.ingest_oxford_style_db(dbdir)
    """
    from PIL import Image
    import os
    print('Loading Oxford Style Images from: ' + dbdir)

    def _parse_oxsty_gtfname(gt_fname):
        """ parse gtfname for: (gt_name, quality_lbl, num) """
        # num is an id, not a number of annots
        gt_format = '{}_{:d}_{:D}.txt'
        name, num, quality = parse.parse(gt_format, gt_fname)
        return (name, num, quality)

    def _read_oxsty_gtfile(gt_fpath, name, quality, img_dpath, ignore_list):
        oxsty_annot_info_list = []
        # read the individual ground truth file
        with open(gt_fpath, 'r') as file:
            line_list = file.read().splitlines()
            for line in line_list:
                if line == '':
                    continue
                fields = line.split(' ')
                gname = fields[0].replace('oxc1_', '') + '.jpg'
                # >:( Because PARIS just cant keep paths consistent
                if gname.find('paris_') >= 0:
                    paris_hack = gname[6:gname.rfind('_')]
                    gname = join(paris_hack, gname)
                if gname in ignore_list:
                    continue
                if len(fields) > 1:  # if has bbox
                    bbox =  [int(round(float(x))) for x in fields[1:]]
                else:
                    # Get annotation width / height
                    gpath = join(img_dpath, gname)
                    (w, h) = Image.open(gpath).size
                    bbox = [0, 0, w, h]
                oxsty_annot_info = (gname, bbox)
                oxsty_annot_info_list.append(oxsty_annot_info)
        return oxsty_annot_info_list

    gt_dpath = utool.existing_subpath(dbdir,
                                      ['oxford_style_gt',
                                       'gt_files_170407',
                                       'oxford_groundtruth'])

    img_dpath = utool.existing_subpath(dbdir,
                                       ['oxbuild_images',
                                        'images'])

    corrupted_file_fpath = join(gt_dpath, 'corrupted_files.txt')
    ignore_list = []
    # Check for corrupted files (Looking at your Paris Buildings Dataset)
    if utool.checkpath(corrupted_file_fpath):
        ignore_list = utool.read_from(corrupted_file_fpath).splitlines()

    #utool.rrrr()
    #utool.list_images = utool.util_path.list_images

    gname_list = utool.list_images(img_dpath, ignore_list=ignore_list,
                                   recursive=True, full=False)

    # just in case utool broke
    for ignore in ignore_list:
        assert ignore not in gname_list

    # Read the Oxford Style Groundtruth files
    print('Loading Oxford Style Names and Annots')
    gt_fname_list = os.listdir(gt_dpath)
    num_gt_files = len(gt_fname_list)
    query_annots  = []
    gname2_annots_raw = utool.ddict(list)
    name_set = set([])
    print(' * num_gt_files = %d ' % num_gt_files)
    #
    # Iterate over each groundtruth file
    mark_, end_ = utool.log_progress('parsed oxsty gtfile: ', num_gt_files)
    for gtx, gt_fname in enumerate(gt_fname_list):
        mark_(gtx)
        if gt_fname == 'corrupted_files.txt':
            continue
        #Get name, quality, and num from fname
        (name, num, quality) = _parse_oxsty_gtfname(gt_fname)
        gt_fpath = join(gt_dpath, gt_fname)
        name_set.add(name)
        oxsty_annot_info_sublist = _read_oxsty_gtfile(
            gt_fpath, name, quality, img_dpath, ignore_list)
        if quality == 'query':
            for (gname, bbox) in oxsty_annot_info_sublist:
                query_annots.append((gname, bbox, name, num))
        else:
            for (gname, bbox) in oxsty_annot_info_sublist:
                gname2_annots_raw[gname].append((name, bbox, quality))
    end_()
    print(' * num_query images = %d ' % len(query_annots))
    #
    # Remove duplicates img.jpg : (*1.txt, *2.txt, ...) -> (*.txt)
    gname2_annots     = utool.ddict(list)
    multinamed_gname_list = []
    for gname, val in gname2_annots_raw.iteritems():
        val_repr = list(map(repr, val))
        unique_reprs = set(val_repr)
        unique_indexes = [val_repr.index(urep) for urep in unique_reprs]
        for ux in unique_indexes:
            gname2_annots[gname].append(val[ux])
        if len(gname2_annots[gname]) > 1:
            multinamed_gname_list.append(gname)
    # print some statistics
    query_gname_list = [tup[0] for tup in query_annots]
    gname_with_groundtruth_list = gname2_annots.keys()
    gname_with_groundtruth_set = set(gname_with_groundtruth_list)
    gname_set = set(gname_list)
    query_gname_set = set(query_gname_list)
    gname_without_groundtruth_list = list(gname_set - gname_with_groundtruth_set)
    print(' * num_images = %d ' % len(gname_list))
    print(' * images with groundtruth    = %d ' % len(gname_with_groundtruth_list))
    print(' * images without groundtruth = %d ' % len(gname_without_groundtruth_list))
    print(' * images with multi-groundtruth = %d ' % len(multinamed_gname_list))
    #make sure all queries have ground truth and there are no duplicate queries
    #
    assert len(query_gname_list) == len(query_gname_set.intersection(gname_with_groundtruth_list))
    assert len(query_gname_list) == len(set(query_gname_list))
    #=======================================================
    # Build IBEIS database
    ibs = ibeis.opendb(dbdir, allow_newdir=True)
    ibs.cfg.other_cfg.auto_localize = False
    print('adding to table: ')
    # Add images to ibeis
    gpath_list = [join(img_dpath, gname) for gname in gname_list]
    gid_list = ibs.add_images(gpath_list)

    # 1) Add Query Annotations
    qgname_list, qbbox_list, qname_list, qid_list = zip(*query_annots)
    # get image ids of queries
    qgid_list = [gid_list[gname_list.index(gname)] for gname in qgname_list]
    qnote_list = ['query'] * len(qgid_list)
    # 2) Add nonquery database annots
    dgname_list = list(gname2_annots.keys())  # NOQA
    dgid_list = []
    dname_list = []
    dbbox_list = []
    dnote_list = []
    for gname in gname2_annots.keys():
        gid = gid_list[gname_list.index(gname)]
        annots = gname2_annots[gname]
        for name, bbox, quality in annots:
            dgid_list.append(gid)
            dbbox_list.append(bbox)
            dname_list.append(name)
            dnote_list.append(quality)
    # 3) Add distractors: TODO: 100k
    ugid_list = [gid_list[gname_list.index(gname)]
                 for gname in gname_without_groundtruth_list]
    ubbox_list = [[0, 0, w, h] for (w, h) in ibs.get_image_sizes(ugid_list)]
    unote_list = ['distractor'] * len(ugid_list)

    # TODO Annotation consistency in terms of duplicate bounding boxes
    qaid_list = ibs.add_annots(qgid_list, bbox_list=qbbox_list, name_list=qname_list, notes_list=qnote_list)
    daid_list = ibs.add_annots(dgid_list, bbox_list=dbbox_list, name_list=dname_list, notes_list=dnote_list)
    uaid_list = ibs.add_annots(ugid_list, bbox_list=ubbox_list, notes_list=unote_list)
    print('Added %d query annototations' % len(qaid_list))
    print('Added %d database annototations' % len(daid_list))
    print('Added %d distractor annototations' % len(uaid_list))


if __name__ == '__main__':
    """
    CommandLine:
        python ibeis/ingest/ingest_database.py --db JAG_Kieryn --force-delete
        python ibeis/ingest/ingest_database.py --db polar_bears --force_delete
        python ibeis/ingest/ingest_database.py --db snails_drop1
    """
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    print('__main__ = ingest_database.py')
    print(utool.unindent(
        '''
        usage:
        python ibeis/ingest/ingest_database.py --db [dbname]

        Valid dbnames:''') + utool.indentjoin(STANDARD_INGEST_FUNCS.keys(), '\n  * '))
    dbname = utool.get_argval('--db', str, None)
    force_delete = utool.get_argflag('--force_delete')
    ibs = ingest_standard_database(dbname, force_delete)
    #img_dir = join(ibeis.sysres.get_workdir(), 'polar_bears')
    #main_locals = ibeis.main(dbdir=img_dir, gui=False)
    #ibs = main_locals['ibs']
    #ingest_rawdata(ibs, img_dir)
