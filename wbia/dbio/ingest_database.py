#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module lists known raw databases and how to ingest them.

Specify arguments and run the following command to ingest a database

python -m wbia --tf ingest_rawdata --db seaturtles  --imgdir "~/turtles/Turtles from Jill" --ingest-type=named_folders --species=turtles
python -m wbia --tf ingest_rawdata --db PZ_OlPej2016 --imgdir /raid/raw/OlPejPZ_June_2016 --ingest-type=named_folders --species=zebra_plains

# --- GET DATA ---
rsync -avhzP <user>@<host>:<remotedir>  <path-to-raw-imgs>
# --- RUN INGEST SCRIPT ---
python -m wbia --tf ingest_rawdata --db <new-wbia-db-name> --imgdir <path-to-raw-imgs> --ingest-type=named_folders --species=<optional> --fmtkey=<optional>


Example:
    >>> # xdoctest: +SKIP
    >>> # The scripts in this file essentiall do this:
    >>> dbdir = '<your new database directory>'
    >>> gpath_list = '<path to your images>'
    >>> ibs = wbia.opendb(dbdir=dbdir, allow_newdir=True)
    >>> gid_list_ = ibs.add_images(gpath_list, auto_localize=False)  # NOQA
    >>> # use whole images as annotations
    >>> aid_list = ibs.use_images_as_annotations(gid_list_, adjust_percent=0)
    >>> # Extra stuff
    >>> name_list = '<names that correspond to your annots>'
    >>> ibs.set_annot_names(aid_list, name_list)
    >>> occur_text_list = '<occurrence that images belongs to>'
    >>> ibs.set_image_imagesettext(gid_list_, occur_text_list)
    >>> ibs.append_annot_case_tags(aid_list, '<annotation tags>')
"""
from __future__ import absolute_import, division, print_function
from six.moves import zip, map, range
import wbia
import os
from os.path import relpath, dirname, exists, join, realpath, basename, abspath
from wbia.other import ibsfuncs
from wbia import constants as const
import utool as ut
import vtool as vt
import parse

(print, rrr, profile) = ut.inject2(__name__)


"""
New Lynx

python -m wbia --tf ingest_rawdata --db lynx2  --imgdir "/media/raid/raw/WildME-WWF-lynx-Sept-2016/CARPETAS CATALOGO INDIVIDUOS" --ingest-type=named_folders --species=lynx --dry

"""


class Ingestable(object):
    """
    Temporary structure representing how to ingest a databases
    """

    def __init__(
        self,
        dbname,
        img_dir=None,
        ingest_type=None,
        fmtkey=None,
        adjust_percent=0.0,
        postingest_func=None,
        zipfile=None,
        species=None,
        images_as_annots=False,
    ):
        self.dbname = dbname
        self.img_dir = img_dir
        self.ingest_type = ingest_type
        self.fmtkey = fmtkey
        self.zipfile = zipfile
        self.adjust_percent = adjust_percent
        self.postingest_func = postingest_func
        self.species = species
        self.images_as_annots = images_as_annots
        self.ensure_feasibility()

    def __str__(self):
        return ut.repr2(self.__dict__)

    def ensure_feasibility(self):
        rawdir = wbia.sysres.get_rawdir()
        if self.img_dir is None:
            # Try to find data either the raw or work dir
            self.img_dir = wbia.sysres.db_to_dbdir(
                self.dbname, extra_workdirs=[rawdir], allow_newdir=True
            )
        msg = 'Cannot find img_dir for dbname=%r, img_dir=%r' % (
            self.dbname,
            self.img_dir,
        )
        assert self.img_dir is not None, msg
        # from os.path import isabs
        # if not isabs(self.img_dir):
        #    self.img_dir = join(dbdir, self.img_dir)
        self.img_dir = ut.truepath(self.img_dir)
        assert exists(self.img_dir), msg
        # if self.ingest_type == 'named_folders':
        #    assert self.fmtkey == 'name'


class Ingestable2(object):
    def __init__(
        self,
        dbdir,
        imgpath_list=None,
        imgdir_list=None,
        zipfile_list=None,
        postingest_func=None,
        ingest_config={},
        **kwargs,
    ):
        self.dbdir = dbdir
        self.zipfile_list = zipfile_list
        self.imgdir_list = imgdir_list
        self.imgpath_list = imgpath_list
        self.postingest_func = postingest_func

        from wbia import dtool

        # valid_species = None
        valid_species = ['____']

        class IngestConfig(dtool.Config):
            _param_info_list = [
                ut.ParamInfo('images_as_annots', False),
                ut.ParamInfo(
                    'ingest_type',
                    'unknown',
                    valid_values=['unknown', 'named_folders', 'named_images'],
                ),
                ut.ParamInfo(
                    'species',
                    '____',
                    hideif=lambda cfg: not cfg['images_as_annots'],
                    valid_values=valid_species,
                ),
                ut.ParamInfo(
                    'adjust_percent', 0.0, hideif=lambda cfg: not cfg['images_as_annots'],
                ),
            ]

        updatekw = kwargs.copy()
        updatekw.update(ingest_config)
        self.ingest_config = IngestConfig(**updatekw)

    def execute(self, ibs=None):
        print('[ingest_rawdata] Ingestable' + str(self))
        assert ibs is not None

        unzipped_file_base_dir = join(ibs.get_dbdir(), 'unzipped_files')

        def extract_from_zipfiles(zipfile_list):
            ut.ensuredir(unzipped_file_base_dir)
            gpath_list = []
            for zipfile in zipfile_list:
                img_dir = unzipped_file_base_dir
                unziped_file_relpath = dirname(
                    relpath(relpath(realpath(zipfile), realpath(img_dir)))
                )
                unzipped_file_dir = join(unzipped_file_base_dir, unziped_file_relpath)
                ut.ensuredir(unzipped_file_dir)
                ut.unzip_file(zipfile, output_dir=unzipped_file_dir, overwrite=False)
                gpaths = ut.list_images(unzipped_file_dir, fullpath=True, recursive=True)
                gpath_list.extend(gpaths)
            return gpath_list

        def list_images(img_dir):
            """ lists images that are not in an internal cache """
            import utool as ut  # NOQA

            ignore_list = [
                '_hsdb',
                '.hs_internals',
                const.PATH_NAMES.cache,
                const.PATH_NAMES._ibsdb,
            ]
            gpath_list = ut.list_images(
                img_dir, fullpath=True, recursive=True, ignore_list=ignore_list
            )
            return gpath_list

        # FIXME ensure python3 works with this
        gpath_list = []
        if self.imgpath_list is not None:
            gpath_list += self.imgpath_list

        if self.imgdir_list is not None:
            for img_dir in self.imgdir_list:
                gpath_list += ut.ensure_unicode_strlist(list_images(img_dir))

        if self.zipfile_list is not None:
            gpath_list += extract_from_zipfiles(self.zipfile_list)

        gpath_list = ut.ensure_unicode_strlist(gpath_list)

        # Parse structure for image names
        ingest_type = self.ingest_config.ingest_type
        if ingest_type == 'named_folders':
            name_list = get_name_texts_from_parent_folder(gpath_list, img_dir, None)
            pass
        elif ingest_type == 'named_images':
            name_list = get_name_texts_from_gnames(gpath_list, img_dir, None)
        elif ingest_type == 'unknown':
            name_list = [const.UNKNOWN for _ in range(len(gpath_list))]
        else:
            raise NotImplementedError('unknwon ingest_type=%r' % (ingest_type,))

        # Add Images
        gpath_list = [gpath.replace('\\', '/') for gpath in gpath_list]
        gid_list_ = ibs.add_images(gpath_list)

        # <DEBUG>
        # print('added: ' + ut.indentjoin(map(str, zip(gid_list_, gpath_list))))
        unique_gids = list(set(gid_list_))
        print('[ingest] Length gid list: %d' % len(gid_list_))
        print('[ingest] Length unique gid list: %d' % len(unique_gids))
        assert len(gid_list_) == len(gpath_list)
        for gid in gid_list_:
            if gid is None:
                print('[ingest] big fat warning')
        # </DEBUG>
        gid_list = ut.filter_Nones(gid_list_)
        unique_gids, unique_names, unique_notes = resolve_name_conflicts(
            gid_list, name_list
        )
        # Add ANNOTATIONs with names and notes
        if self.ingest_config.images_as_annots:
            aid_list = ibs.use_images_as_annotations(
                unique_gids, adjust_percent=self.ingest_config.adjust_percent
            )
            ibs.set_annot_names(aid_list, unique_names)
            ibs.set_annot_notes(aid_list, unique_notes)
            species_text = self.ingest_config.species
            if species_text is not None:
                ibs.set_annot_species(aid_list, [species_text] * len(aid_list))

        localize = False
        if localize:
            ibs.localize_images()

        if self.postingest_func is not None:
            self.postingest_func(ibs)
        return gid_list


def ingest_rawdata(ibs, ingestable, localize=False):
    """
    Ingests rawdata into an wbia database.

    Args:
        ibs (wbia.IBEISController):  wbia controller object
        ingestable (Ingestable):
        localize (bool): (default = False)

    Returns:
        list: aid_list -  list of annotation rowids

    Notes:
        if ingest_type == 'named_folders':
            Converts folder structure where folders = name, to ibsdb
        if ingest_type == 'named_images':
            Converts imgname structure where imgnames = name_id.ext, to ibsdb

    CommandLine:
        python wbia/dbio/ingest_database.py --db seals_drop2
        python -m wbia.dbio.ingest_database --exec-ingest_rawdata
        python -m wbia.dbio.ingest_database --exec-ingest_rawdata --db snow-leopards --imgdir /raid/raw_rsync/snow-leopards

        python -m wbia --tf ingest_rawdata --db wd_peter2 --imgdir /raid/raw_rsync/african-dogs --ingest-type=named_folders --species=wild_dog --fmtkey='African Wild Dog: {name}' --force-delete
        python -m wbia --tf ingest_rawdata --db <newdbname>  --imgdir <path-to-images> --ingest-type=named_folders --species=humpback

    Example:
        >>> # SCRIPT
        >>> # General ingest script
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> import wbia
        >>> dbname = ut.get_argval('--db', str, None)  # 'snow-leopards')
        >>> force_delete = ut.get_argflag(('--force_delete', '--force-delete'))
        >>> img_dir = ut.get_argval('--imgdir', type_=str, default=None)
        >>> ingest_type = ut.get_argval('--ingest-type', type_=str, default='unknown')
        >>> fmtkey = ut.get_argval('--fmtkey', type_=str, default=None)
        >>> species = ut.get_argval('--species', type_=str, default=None)
        >>> images_as_annots = ut.get_argval('--images-as-annots', type_=bool, default=None)
        >>> if images_as_annots is None:
        >>>     images_as_annots = ingest_type != 'unknown'
        >>> assert img_dir is not None, 'specify img dir'
        >>> assert dbname is not None, 'specify dbname'
        >>> ingestable = Ingestable(
        >>>     dbname, img_dir=img_dir, ingest_type=ingest_type,
        >>>     fmtkey=fmtkey, species=species, images_as_annots=images_as_annots,
        >>>     adjust_percent=0.00)
        >>> from wbia.control import IBEISControl
        >>> dbdir = wbia.sysres.db_to_dbdir(dbname, allow_newdir=True)
        >>> ut.ensuredir(dbdir, verbose=True)
        >>> if force_delete:
        >>>     ibsfuncs.delete_wbia_database(dbdir)
        >>> ibs = IBEISControl.request_IBEISController(dbdir)
        >>> localize = False
        >>> gid_list = ingest_rawdata(ibs, ingestable, localize)
        >>> result = ('gid_list = %s' % (str(gid_list),))
        >>> print(result)
    """
    print('[ingest_rawdata] Ingestable' + str(ingestable))

    if ingestable.zipfile is not None:
        zipfile_fpath = ut.truepath(join(wbia.sysres.get_workdir(), ingestable.zipfile))
        ingestable.img_dir = ut.unarchive_file(zipfile_fpath)

    img_dir = realpath(ingestable.img_dir)
    ingest_type = ingestable.ingest_type
    fmtkey = ingestable.fmtkey
    adjust_percent = ingestable.adjust_percent
    species_text = ingestable.species
    postingest_func = ingestable.postingest_func
    print(
        '[ingest] ingesting rawdata: img_dir=%r, injest_type=%r' % (img_dir, ingest_type)
    )
    # Get images in the image directory

    unzipped_file_base_dir = join(ibs.get_dbdir(), 'unzipped_files')

    def extract_zipfile_images(ibs, ingestable):
        import utool as ut  # NOQA

        zipfile_list = ut.glob(ingestable.img_dir, '*.zip', recursive=True)
        gpath_list = []
        if len(zipfile_list) > 0:
            print('Found zipfile_list = %r' % (zipfile_list,))
            ut.ensuredir(unzipped_file_base_dir)
            for zipfile in zipfile_list:
                unziped_file_relpath = dirname(
                    relpath(relpath(realpath(zipfile), realpath(ingestable.img_dir)))
                )
                unzipped_file_dir = join(unzipped_file_base_dir, unziped_file_relpath)
                ut.ensuredir(unzipped_file_dir)
                ut.unzip_file(zipfile, output_dir=unzipped_file_dir, overwrite=False)
                gpaths = ut.list_images(unzipped_file_dir, fullpath=True, recursive=True)
                gpath_list.extend(gpaths)
        return gpath_list

    def list_images(img_dir):
        """ lists images that are not in an internal cache """
        import utool as ut  # NOQA

        ignore_list = [
            '_hsdb',
            '.hs_internals',
            const.PATH_NAMES.cache,
            const.PATH_NAMES._ibsdb,
        ]
        gpath_list = ut.list_images(
            img_dir, fullpath=True, recursive=True, ignore_list=ignore_list
        )
        return gpath_list

    # FIXME ensure python3 works with this
    gpath_list1 = ut.ensure_unicode_strlist(list_images(img_dir))
    gpath_list2 = ut.ensure_unicode_strlist(extract_zipfile_images(ibs, ingestable))
    gpath_list = gpath_list1 + gpath_list2

    # Parse structure for image names
    if ingest_type == 'named_folders':
        name_list1 = get_name_texts_from_parent_folder(gpath_list1, img_dir, fmtkey)
        name_list2 = get_name_texts_from_parent_folder(
            gpath_list2, unzipped_file_base_dir, fmtkey
        )
        name_list = name_list1 + name_list2
        pass
    elif ingest_type == 'named_images':
        name_list = get_name_texts_from_gnames(gpath_list, img_dir, fmtkey)
    elif ingest_type == 'unknown':
        name_list = [const.UNKNOWN for _ in range(len(gpath_list))]
    else:
        raise NotImplementedError('unknwon ingest_type=%r' % (ingest_type,))

    # Find names likely to be the same?
    RECTIFY_NAMES_HUERISTIC = True
    if RECTIFY_NAMES_HUERISTIC:
        names = sorted(list(set(name_list)))
        splitchars = [' ', '/']

        def multisplit(str_, splitchars):
            import utool as ut

            n = [str_]
            for char in splitchars:
                n = ut.flatten([_.split(char) for _ in n])
            return n

        groupids = [multisplit(n1, splitchars)[0] for n1 in names]
        grouped_names = ut.group_items(names, groupids)
        fixed_names = {
            newkey: key
            for key, val in grouped_names.items()
            if len(val) > 1
            for newkey in val
        }
        name_list = [fixed_names.get(name, name) for name in name_list]

    # Add Images

    gpath_list = [gpath.replace('\\', '/') for gpath in gpath_list]

    if ut.get_argflag('--dry'):
        print('Found %d names' % (len(set(name_list))))
        print(set(name_list))
        print('Dry Run')
        return

    gid_list_ = ibs.add_images(gpath_list)

    # <DEBUG>
    # print('added: ' + ut.indentjoin(map(str, zip(gid_list_, gpath_list))))
    unique_gids = list(set(gid_list_))
    print('[ingest] Length gid list: %d' % len(gid_list_))
    print('[ingest] Length unique gid list: %d' % len(unique_gids))
    assert len(gid_list_) == len(gpath_list)
    for gid in gid_list_:
        if gid is None:
            print('[ingest] big fat warning')
    # </DEBUG>
    gid_list = ut.filter_Nones(gid_list_)
    unique_gids, unique_names, unique_notes = resolve_name_conflicts(gid_list, name_list)
    # Add ANNOTATIONs with names and notes
    if ingestable.images_as_annots:
        aid_list = ibs.use_images_as_annotations(
            unique_gids, adjust_percent=adjust_percent
        )
        ibs.set_annot_names(aid_list, unique_names)
        ibs.set_annot_notes(aid_list, unique_notes)
        if species_text is not None:
            ibs.set_annot_species(aid_list, [species_text] * len(aid_list))
    if localize:
        ibs.localize_images()

    TURTLE_HURISTIC = 'turtles' in img_dir
    if TURTLE_HURISTIC:
        """
        python -m wbia --tf ingest_rawdata --db seaturtles  --imgdir "~/turtles/Turtles from Jill" --ingest-type=named_folders --species=turtles
        """
        aid_list = ibs.get_valid_aids()
        parent_gids = ibs.get_annot_gids(aid_list)
        annot_orig_uris = ibs.get_image_uris_original(parent_gids)

        def parse_turtle_uri(uri):
            from os.path import splitext, dirname, basename

            info = {}
            uril = uri.lower()

            def findany(text, possible):
                return any([x in text for x in possible])

            if findany(uril, ['right']) or splitext(uril)[0].endswith('rs'):
                info['view'] = 'right'
            if findany(uril, ['left']) or splitext(uril)[0].endswith('ls'):
                info['view'] = 'left'
            if findany(uril, ['carapace', 'whole', 'carpace']) or splitext(uril)[
                0
            ].endswith('wb'):
                # info['view'] = 'top'
                info['view'] = 'up'
            occurrence_id = basename(dirname(uri))
            info['occurrence'] = 'occurrence' + occurrence_id
            return info

        turtle_info_list = [parse_turtle_uri(uri) for uri in annot_orig_uris]
        view_text_list = ut.take_column(turtle_info_list, 'view')
        occur_text_list = ut.take_column(turtle_info_list, 'occurrence')
        turtle_tag_list = list(zip(occur_text_list, view_text_list))

        # TODO: mark viewpoints using euler angles / quaternions
        ibs.set_image_imagesettext(parent_gids, occur_text_list)
        ibs.append_annot_case_tags(aid_list, ut.lmap(list, turtle_tag_list))

    if postingest_func is not None:
        postingest_func(ibs)
    # Print to show success
    # ibs.print_image_table()
    # ibs.print_tables()
    # ibs.print_annotation_table()
    # ibs.print_alr_table()
    # ibs.print_lblannot_table()
    # ibs.print_image_table()
    # return aid_list
    return gid_list


def normalize_name(name):
    """ Maps unknonwn names to the standard ____ """
    if name in const.ACCEPTED_UNKNOWN_NAMES:
        name = const.INDIVIDUAL_KEY
    return name


def get_name_texts_from_parent_folder(gpath_list, img_dir, fmtkey=None):
    """
    Input: gpath_list
    Output: names based on the parent folder of each image
    """
    # from os.path import commonprefix
    relgpath_list = [relpath(gpath, img_dir) for gpath in gpath_list]
    # _prefix = commonprefix(gpath_list)
    # relgpath_list = [relpath(gpath, _prefix) for gpath in gpath_list]
    _name_list = [dirname(relgpath) for relgpath in relgpath_list]

    if fmtkey is not None:
        # fmtkey = 'African Wild Dog: {name}'
        import parse

        parse_results = [parse.parse(fmtkey, name) for name in _name_list]
        _name_list = [
            res['name'] if res is not None else name
            for name, res in zip(_name_list, parse_results)
        ]

    name_list = list(map(normalize_name, _name_list))
    return name_list


class FMT_KEYS(object):  # NOQA
    name_fmt = '{name:*}[id:d].{ext}'
    snails_fmt = '{name:*dd}{id:dd}.{ext}'
    giraffe1_fmt = '{name:*}_{id:d}.{ext}'
    seal2_fmt = '{name:Phsd*}{id:[A-Z]}.{ext}'
    elephant_fmt = '{prefix?}{name}_{view}_{id?}.{ext}'


def get_name_texts_from_gnames(gpath_list, img_dir, fmtkey='{name:*}[aid:d].{ext}'):
    """
    Args:
        gpath_list (list): list of image paths
        img_dir (str): path to image directory
        fmtkey (str): pattern string to parse names from (default = '{name:*}[aid:d].{ext}')

    Returns:
        list: name_list - based on the parent folder of each image

    CommandLine:
        python -m wbia.dbio.ingest_database --test-get_name_texts_from_gnames

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> gpath_list = ['e_f0273_f.jpg', 'f0001_f.jpg', 'f0259_l_3.jpg', 'f0259_f_1.jpg',  'f0259_f (1).jpg', 'f0058_u16_f.jpg']
        >>> img_dir = ''
        >>> fmtkey = FMT_KEYS.elephant_fmt
        >>> result = get_name_texts_from_gnames(gpath_list, img_dir, fmtkey)
        >>> print(result)
    """
    # These define regexes that attempt to parse the insane and contradicting
    # naming schemes of the image sets that we get.
    INGEST_FORMATS = {
        FMT_KEYS.name_fmt: ut.named_field_regex(
            [
                ('name', r'[a-zA-Z]+'),  # all alpha characters
                ('id', r'\d*'),  # first numbers (if existant)
                (None, r'\.'),
                ('ext', r'\w+'),
            ]
        ),
        FMT_KEYS.snails_fmt: ut.named_field_regex(
            [
                ('name', r'[a-zA-Z]+\d\d'),  # species and 2 numbers
                ('id', r'\d\d'),  # 2 more numbers
                (None, r'\.'),
                ('ext', r'\w+'),
            ]
        ),
        FMT_KEYS.giraffe1_fmt: ut.named_field_regex(
            [
                ('name', r'G\d+'),
                ('under', r'_'),
                ('id', r'\d+'),
                (None, r'\.'),
                ('ext', r'\w+'),
            ]
        ),
        FMT_KEYS.seal2_fmt: ut.named_field_regex(
            [
                ('name', r'Phs\d+'),  # Phs and then numbers
                ('id', r'[A-Z]+'),  # 1 or more letters
                (None, r'\.'),
                ('ext', r'\w+'),
            ]
        ),
        # this one defines multiple possible regex types. yay standards
        FMT_KEYS.elephant_fmt: [
            ut.named_field_regex(
                [
                    ('prefix', r'(e_)?'),
                    ('name', r'[a-zA-Z0-9]+'),
                    ('view', r'_[rflo]'),
                    ('id', r'([ _][^.]+)?'),
                    (None, r'\.'),
                    ('ext', r'\w+'),
                ]
            ),
            ut.named_field_regex(
                [
                    ('prefix', r'(e_)?'),
                    ('name', r'[a-zA-Z0-9]+'),
                    ('id', r'([ _][^.]+)?'),
                    ('view', r'_[rflo]'),
                    (None, r'\.'),
                    ('ext', r'\w+'),
                ]
            ),
        ],
    }
    regex_list = INGEST_FORMATS.get(fmtkey, fmtkey)
    gname_list = ut.fpaths_to_fnames(gpath_list)

    def parse_format(regex_list, gname):
        if not isinstance(regex_list, list):
            regex_list = [regex_list]
        for regex in regex_list:
            result = ut.regex_parse(regex, gname)
            if result is not None:
                return result
        return None

    parsed_list = [parse_format(regex_list, gname) for gname in gname_list]

    anyfailed = False
    for gpath, parsed in zip(gpath_list, parsed_list):
        if parsed is None:
            print('FAILED TO PARSE: %r' % gpath)
            anyfailed = True
    if anyfailed:
        msg = 'FAILED REGEX: %r' % regex_list
        raise Exception(msg)

    _name_list = [parsed['name'] for parsed in parsed_list]
    name_list = list(map(normalize_name, _name_list))
    return name_list


def resolve_name_conflicts(gid_list, name_list):
    """ """
    # Build conflict map (values are lists of members)
    conflict_gid_to_names = ut.build_conflict_dict(gid_list, name_list)

    # Check to see which gid has more than one name
    unique_gids = ut.unique_ordered(gid_list)
    unique_names = []
    unique_notes = []

    for gid in unique_gids:
        names = ut.unique_ordered(conflict_gid_to_names[gid])
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
# ## <STANDARD DATABASES> ###

STANDARD_INGEST_FUNCS = {}


def __standard(dbname):
    """  Decorates a function as a standard ingestable database """

    def __registerdb(func):
        STANDARD_INGEST_FUNCS[dbname] = func
        return func

    return __registerdb


@__standard('testdb1')
def ingest_testdb1(dbname):
    """
    ingest_testdb1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> import utool as ut
        >>> from wbia import demodata
        >>> import wbia
        >>> demodata.ensure_testdata()
        >>> # DELETE TESTDB1
        >>> TESTDB1 = ut.unixjoin(wbia.sysres.get_workdir(), 'testdb1')
        >>> ut.delete(TESTDB1, ignore_errors=False)
        >>> result = ingest_testdb1(dbname)
    """
    from wbia import demodata  # TODO: remove and use utool appdir

    def postingest_tesdb1_func(ibs):
        import numpy as np
        from wbia import constants as const

        print('postingest_tesdb1_func')
        # Adjust data as we see fit

        # gid_list = np.array(ibs.images()._rowids)
        gid_list = np.array(ibs.get_valid_gids())
        # Set image unixtimes
        unixtimes_even = (gid_list[0::2] + 100).tolist()
        unixtimes_odd = (gid_list[1::2] + 9001).tolist()
        unixtime_list = unixtimes_even + unixtimes_odd
        ibs.set_image_unixtime(gid_list, unixtime_list)
        # Unname first aid in every name
        aid_list = ibs.get_valid_aids()
        nid_list = ibs.get_annot_name_rowids(aid_list)
        nid_list = [(nid if nid > 0 else None) for nid in nid_list]
        unique_flag = ut.flag_unique_items(nid_list)
        unique_nids = ut.compress(nid_list, unique_flag)
        none_nids = [nid is not None for nid in nid_list]
        flagged_nids = [nid for nid in unique_nids if nid_list.count(nid) > 1]
        plural_flag = [nid in flagged_nids for nid in nid_list]
        flag_list = list(map(all, zip(plural_flag, unique_flag, none_nids)))
        flagged_aids = ut.compress(aid_list, flag_list)
        if ut.VERYVERBOSE:

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
        unname_aids = ut.compress(aid_list, flag_list)
        ibs.delete_annot_nids(unname_aids)
        # Add all annotations with names as exemplars
        # from wbia.control.IBEISControl import IBEISController
        # assert isinstance(ibs, IBEISController)
        unflagged_aids = ut.get_dirty_items(aid_list, flag_list)
        exemplar_flags = [True] * len(unflagged_aids)
        ibs.set_annot_exemplar_flags(unflagged_aids, exemplar_flags)
        # Set some test species labels
        species_text_list = ibs.get_annot_species_texts(aid_list)
        for ix in range(0, 6):
            species_text_list[ix] = const.TEST_SPECIES.ZEB_PLAIN
        # These are actually plains zebras.
        for ix in range(8, 10):
            species_text_list[ix] = const.TEST_SPECIES.ZEB_GREVY
        for ix in range(10, 12):
            species_text_list[ix] = const.TEST_SPECIES.BEAR_POLAR

        ibs.set_annot_species(aid_list, species_text_list)
        ibs.set_annot_viewpoint_code(aid_list[0:6], ['left'] * 6)
        ibs.set_annot_viewpoint_code(aid_list[8:10], ['left'] * 2)
        ibs.set_annot_viewpoint_code(aid_list[10:12], ['right'] * 2)
        ibs.set_annot_notes(aid_list[8:10], ['this is actually a plains zebra'] * 2)
        ibs.set_annot_notes(aid_list[0:1], ['aid 1 and 2 are correct matches'])
        ibs.set_annot_notes(
            aid_list[6:7], ['very simple image to debug feature detector']
        )
        ibs.set_annot_notes(aid_list[7:8], ['standard test image'])
        ibs.set_annot_reviewed(aid_list[::2], [True] * len(aid_list[::2]))
        ibs.set_annot_multiple(aid_list[::2], [False] * len(aid_list[::2]))

        # Set some randomish gps flags that are within nnp
        unixtime_list = ibs.get_image_unixtime(gid_list)
        valid_lat_min = -1.4446
        valid_lat_max = -1.3271
        valid_lon_min = 36.7619
        valid_lon_max = 36.9622
        valid_lat_range = valid_lat_max - valid_lat_min
        valid_lon_range = valid_lon_max - valid_lon_min
        randstate = np.random.RandomState(unixtime_list)
        new_gps_list = randstate.rand(len(gid_list), 2)
        new_gps_list[:, 0] = (new_gps_list[:, 0] * valid_lat_range) + valid_lat_min
        new_gps_list[:, 1] = (new_gps_list[:, 1] * valid_lon_range) + valid_lon_min
        new_gps_list[8, :] = [-1, -1]
        ibs.set_image_gps(gid_list, new_gps_list)

        # TODO: add a nan timestamp
        ibs.append_annot_case_tags([2], ['error:bbox'])
        ibs.append_annot_case_tags([4], ['quality:washedout'])
        ibs.append_annot_case_tags([4], ['lighting'])

        aidgroups = ibs.group_annots_by_name(
            ibs.filter_annots_general(min_pername=2, verbose=True)
        )[0]
        aid1_list = ut.take_column(aidgroups, 0)
        aid2_list = ut.take_column(aidgroups, 1)
        annotmatch_rowids = ibs.add_annotmatch_undirected(aid1_list, aid2_list)

        ibs.set_annotmatch_evidence_decision(
            annotmatch_rowids, [True] * len(annotmatch_rowids)
        )
        ibs.set_annotmatch_prop(
            'photobomb', annotmatch_rowids, [True] * len(annotmatch_rowids)
        )

        for aids in aidgroups:
            pass

        print('finish postingest_tesdb1_func')
        return None

    return Ingestable(
        dbname,
        ingest_type='named_images',
        fmtkey=FMT_KEYS.name_fmt,
        img_dir=demodata.get_testdata_dir(),
        adjust_percent=0.00,
        images_as_annots=True,
        postingest_func=postingest_tesdb1_func,
    )


@__standard('humpbacks')
def ingest_humpbacks(dbname):
    # The original humpbacks data is ROI cropped images in the
    # named folder format
    return Ingestable(
        dbname,
        ingest_type='named_folders',
        adjust_percent=0.00,
        species='whale_humpback',
        # this zipfile is only on Zach's machine
        fmtkey='name',
    )


@__standard('polar_bears')
def ingest_polar_bears(dbname):
    return Ingestable(
        dbname, ingest_type='named_folders', adjust_percent=0.00, fmtkey='name'
    )


@__standard('wd_peter_blinston')
def ingest_wilddog_peter(dbname):
    """
    CommandLine:
        python -m wbia.dbio.ingest_database --exec-injest_main --db wd_peter_blinston
    """
    return Ingestable(
        dbname,
        ingest_type='unknown',
        img_dir='/raid/raw_rsync/african-dogs',
        adjust_percent=0.01,
        species=const.Species.WILDDOG,
    )


@__standard('lynx')
def ingest_lynx(dbname):
    """
    CommandLine:
        python -m wbia.dbio.ingest_database --exec-injest_main --db lynx
    """
    return Ingestable(
        dbname,
        ingest_type='named_folders',
        img_dir='/raid/raw_rsync/iberian-lynx/CARPETAS CATALOGO INDIVIDUOS/',
        adjust_percent=0.01,
        species='lynx',
        fmtkey='name',
    )


@__standard('WS_ALL')
def ingest_whale_sharks(dbname):
    """
    CommandLine:
        python -m wbia.dbio.ingest_database --exec-injest_main --db WS_ALL
    """
    return Ingestable(
        dbname,
        ingest_type='named_folders',
        img_dir='named-left-sharkimages',
        adjust_percent=0.01,
        species='whale_shark',
        fmtkey='name',
    )


@__standard('snails_drop1')
def ingest_snails_drop1(dbname):
    return Ingestable(
        dbname,
        ingest_type='named_images',
        fmtkey=FMT_KEYS.snails_fmt,
        species='snail',
        # img_dir='/raid/raw/snails_drop1_59MB',
        adjust_percent=0.20,
    )


@__standard('seals_drop2')
def ingest_seals_drop2(dbname):
    return Ingestable(
        dbname,
        zipfile='../raw/hiby_Phs_photos.zip',
        ingest_type='named_images',
        fmtkey=FMT_KEYS.seal2_fmt,
        # img_dir='/raid/raw/snails_drop1_59MB',
        adjust_percent=0.20,
        species='seal_saimma_ringed',
    )


@__standard('JAG_Kieryn')
def ingest_JAG_Kieryn(dbname):
    return Ingestable(
        dbname, ingest_type='unknown', species='jaguar', adjust_percent=0.00
    )


@__standard('Giraffes')
def ingest_Giraffes1(dbname):
    return Ingestable(
        dbname,
        ingest_type='named_images',
        fmtkey=FMT_KEYS.giraffe1_fmt,
        species='giraffe_reticulated',
        adjust_percent=0.00,
    )


@__standard('Elephants_drop1')
def ingest_Elephants_drop1(dbname):
    return Ingestable(
        dbname,
        zipfile='../raw_unprocessed/ID photo front_Elephants_4-29-2015-PeterGranli.zip',  # NOQA
        ingest_type='named_images',
        fmtkey=FMT_KEYS.elephant_fmt,
        species='elephant_savanna',
        adjust_percent=0.00,
    )


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

    Ignore:
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> dbname = 'testdb1'
        >>> force_delete = False
        >>> result = ingest_standard_database(dbname, force_delete)
        >>> print(result)
    """
    from wbia.control import IBEISControl

    print('[ingest] Ingest Standard Database: dbname=%r' % (dbname,))
    ingestable = get_standard_ingestable(dbname)
    dbdir = wbia.sysres.db_to_dbdir(ingestable.dbname, allow_newdir=True)
    ut.ensuredir(dbdir, verbose=True)
    if force_delete:
        ibsfuncs.delete_wbia_database(dbdir)
    ibs = IBEISControl.request_IBEISController(dbdir)
    ingest_rawdata(ibs, ingestable)


# ## </STANDARD DATABASES> ###
#
#


def ingest_oxford_style_db(dbdir, dryrun=False):
    """
    Ingest either oxford or paris

    Args:
        dbdir (str):

    CommandLine:
        python -m wbia.dbio.ingest_database --exec-ingest_oxford_style_db --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> dbdir = '/raid/work/Oxford'
        >>> dryrun = True
        >>> ingest_oxford_style_db(dbdir)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()

    Ignore:
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> import wbia
        >>> dbdir = '/raid/work/Oxford'
        >>> dbdir = '/raid/work/Paris'
        >>>
        #>>> wbia.dbio.convert_db.ingest_oxford_style_db(dbdir)
    """
    print('Loading Oxford Style Images from: ' + dbdir)

    def _parse_oxsty_gtfname(gt_fname):
        """ parse gtfname for: (gt_name, quality_lbl, num) """
        # num is an id, not a number of annots
        gt_format = '{}_{:d}_{:D}.txt'
        name, num, quality = parse.parse(gt_format, gt_fname)
        return (name, num, quality)

    @ut.memoize
    def _tmpread(gt_fpath):
        return ut.readfrom(gt_fpath)

    def _read_oxsty_gtfile(gt_fpath, name, quality, img_dpath, ignore_list):
        oxsty_annot_info_list = []
        # read the individual ground truth file
        line_list = _tmpread(gt_fpath).splitlines()
        # line_list = file.read().splitlines()
        for line in line_list:
            if line == '':
                continue
            fields = line.split(' ')
            gname = fields[0].replace('oxc1_', '') + '.jpg'
            # >:( Because PARIS just cant keep paths consistent
            if gname.find('paris_') >= 0:
                paris_hack = gname[6 : gname.rfind('_')]
                gname = join(paris_hack, gname)
            if gname in ignore_list:
                continue
            if len(fields) > 1:
                # if has bbox
                # x1, y1, w, h =  [int(round(float(x))) for x in fields[1:]]
                x1, y1, x2, y2 = [int(round(float(x))) for x in fields[1:]]
                w = x2 - x1
                h = y2 - y1
                bbox = [x1, y1, w, h]
            else:
                # Get annotation width / height
                gpath = join(img_dpath, gname)
                h, w, c = vt.imread(gpath, orient='auto').shape
                bbox = [0, 0, w, h]
            oxsty_annot_info = (gname, bbox)
            oxsty_annot_info_list.append(oxsty_annot_info)
        return oxsty_annot_info_list

    gt_dpath = ut.existing_subpath(
        dbdir, ['oxford_style_gt', 'gt_files_170407', 'oxford_groundtruth']
    )

    img_dpath = ut.existing_subpath(dbdir, ['oxbuild_images', 'images'])

    corrupted_file_fpath = join(gt_dpath, 'corrupted_files.txt')
    ignore_list = []
    # Check for corrupted files (Looking at your Paris Buildings Dataset)
    if ut.checkpath(corrupted_file_fpath):
        ignore_list = ut.read_from(corrupted_file_fpath).splitlines()

    gname_list = ut.list_images(
        img_dpath, ignore_list=ignore_list, recursive=True, full=False
    )

    # just in case utool broke
    for ignore in ignore_list:
        assert ignore not in gname_list

    # Read the Oxford Style Groundtruth files
    print('Loading Oxford Style Names and Annots')
    gt_fname_list = os.listdir(gt_dpath)
    num_gt_files = len(gt_fname_list)
    query_annots = []
    gname2_annots_raw = ut.ddict(list)
    name_set = set([])
    print(' * num_gt_files = %d ' % num_gt_files)
    #
    # Iterate over each groundtruth file
    for gtx, gt_fname in enumerate(ut.ProgIter(gt_fname_list, 'parsed oxsty gtfile: ')):
        if gt_fname == 'corrupted_files.txt':
            continue
        # Get name, quality, and num from fname
        (name, num, quality) = _parse_oxsty_gtfname(gt_fname)
        gt_fpath = join(gt_dpath, gt_fname)
        name_set.add(name)
        oxsty_annot_info_sublist = _read_oxsty_gtfile(
            gt_fpath, name, quality, img_dpath, ignore_list
        )
        if quality == 'query':
            for (gname, bbox) in oxsty_annot_info_sublist:
                query_annots.append((gname, bbox, name, num))
        else:
            for (gname, bbox) in oxsty_annot_info_sublist:
                gname2_annots_raw[gname].append((name, bbox, quality))
    print(' * num_query images = %d ' % len(query_annots))
    #
    # Remove duplicates img.jpg : (*1.txt, *2.txt, ...) -> (*.txt)
    gname2_annots = ut.ddict(list)
    multinamed_gname_list = []
    for gname, val in gname2_annots_raw.items():
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
    # make sure all queries have ground truth and there are no duplicate queries
    #
    assert len(query_gname_list) == len(
        query_gname_set.intersection(gname_with_groundtruth_list)
    )
    assert len(query_gname_list) == len(set(query_gname_list))
    # =======================================================
    # Build IBEIS database

    if not dryrun:
        ibs = wbia.opendb(dbdir, allow_newdir=True)
        print('adding to table: ')
        # Add images to wbia
        gpath_list = [join(img_dpath, gname).replace('\\', '/') for gname in gname_list]
        gid_list = ibs.add_images(gpath_list, auto_localize=False)

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
        ugid_list = [
            gid_list[gname_list.index(gname)] for gname in gname_without_groundtruth_list
        ]
        ubbox_list = [[0, 0, w, h] for (w, h) in ibs.get_image_sizes(ugid_list)]
        unote_list = ['distractor'] * len(ugid_list)

        # TODO Annotation consistency in terms of duplicate bounding boxes
        qaid_list = ibs.add_annots(
            qgid_list, bbox_list=qbbox_list, name_list=qname_list, notes_list=qnote_list
        )
        daid_list = ibs.add_annots(
            dgid_list, bbox_list=dbbox_list, name_list=dname_list, notes_list=dnote_list
        )
        uaid_list = ibs.add_annots(ugid_list, bbox_list=ubbox_list, notes_list=unote_list)
        print('Added %d query annototations' % len(qaid_list))
        print('Added %d database annototations' % len(daid_list))
        print('Added %d distractor annototations' % len(uaid_list))

    update = False
    if update:
        # TODO: integrate this into normal ingest pipeline
        'Oxford'
        ibs = wbia.opendb(dbdir)
        aid_list = ibs.get_valid_aids()
        notes_list = ibs.get_annot_notes(aid_list)
        _dict = {
            'ok': ibs.const.QUAL_OK,
            'good': ibs.const.QUAL_GOOD,
            'junk': ibs.const.QUAL_JUNK,
            # 'distractor': ibs.const.QUAL_JUNK
        }
        qual_text_list = [_dict.get(note, ibs.const.QUAL_UNKNOWN) for note in notes_list]
        ibs.set_annot_quality_texts(aid_list, qual_text_list)
        ibs._overwrite_all_annot_species_to('building')

        tags_list = [
            [note] if note in ['query', 'distractor'] else [] for note in notes_list
        ]
        from wbia import tag_funcs

        tag_funcs.append_annot_case_tags(ibs, aid_list, tags_list)
        # ibs._set
        # tags_ = ibs.get_annot_case_tags(aid_list)
        # pass
        """
        python -m wbia --tf filter_annots_general --db Oxford --has_any=[query]
        """
    if False:
        """
        dbname = 'Oxford'
        """
        import pandas as pd

        columns = ['gt_fname', 'name', 'num', 'quality']
        rows = [
            (gt_fname,) + _parse_oxsty_gtfname(gt_fname) for gt_fname in gt_fname_list
        ]
        df = pd.DataFrame(rows, columns=columns)
        query_df = df[df['quality'] == 'query']
        # query_df = query_df.assign(bbox=None)

        query_annot_rows = []
        for row in query_df.iterrows():
            gt_fname, name, num, quality = row[1]._values
            if gt_fname == 'corrupted_files.txt':
                continue
            # Get name, quality, and num from fname
            gt_fpath = join(gt_dpath, gt_fname)
            oxsty_annot_info_sublist = _read_oxsty_gtfile(
                gt_fpath, name, quality, img_dpath, ignore_list
            )
            for (gname, bbox) in oxsty_annot_info_sublist:
                query_annot_rows.append((gname, bbox, name, num))

        query_df2 = pd.DataFrame(
            query_annot_rows, columns=['gname', 'bbox', 'name', 'num']
        )

        # Fix query bounding boxes
        ibs = wbia.opendb(dbdir)
        qaids = ibs.filter_annots_general(has_any='query')
        qannots = ibs.annots(qaids)

        # Ensure query_df2 are aligned with database qannots
        import numpy as np

        qannot_basename = [basename(p) for p in ibs.get_image_uris_original(qannots.gids)]
        imgname_to_idx = ut.make_index_lookup(qannot_basename)
        sortx1 = ut.take(imgname_to_idx, query_df2['gname'].values)
        query_df3 = query_df2.take(np.argsort(sortx1))

        assert np.all(qannots.names == query_df3['name'].values)

        # Sort by name
        unique_names, groupxs = ut.group_indices(qannots.names)
        sortx2 = ut.flatten(groupxs)
        qannots4 = qannots.take(sortx2)
        query_df4 = query_df3.take(sortx2)

        old_bboxes = np.array(qannots4.bboxes, dtype=np.float)
        new_bboxes = np.array(query_df4['bbox'].values.tolist(), dtype=np.float)
        new_ar = new_bboxes.T[2] / new_bboxes.T[3]
        old_ar = old_bboxes.T[2] / old_bboxes.T[3]
        print(new_ar == old_ar)
        print(new_ar)
        print(old_ar)

        ibs.set_annot_bboxes(qannots4.aids, new_bboxes)

        ibs.images(qannots4.gids).append_to_imageset('Queries')


# def ingest_pascal_voc_style_db(dbdir, dryrun=False):
#     pass


def ingest_coco_style_db(dbdir, dryrun=False):
    """
    Ingest a PASCAL VOC formatted datbase

    Args:
        dbdir (str):

    CommandLine:
        python -m wbia.dbio.ingest_database --exec-ingest_coco_style_db --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> dbdir = '/Datasets/coco'
        >>> dryrun = True
        >>> ingest_coco_style_db(dbdir)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import simplejson as json
    import datetime

    print('Loading PASCAL VOC Style Images from: ' + dbdir)

    annot_path = join(dbdir, 'annotations')
    assert exists(annot_path)

    dataset_name_list = [
        'test2014',  # THIS SHOULD BE FIRST
        'test2015',
        'train2014',
        'val2014',
    ]

    test2014_seen_set = set([])
    cat_dict = {}
    lic_dict = {}
    image_dict = {}
    for dataset_name in dataset_name_list:
        print('Processing Dataset %r' % (dataset_name,))
        print('\tLoading Instances JSON...')
        inst_filepath = join(annot_path, 'instances_%s.json' % (dataset_name,))
        if exists(inst_filepath):
            with open(inst_filepath) as inst_file:
                inst_dict = json.load(inst_file)
        else:
            inst_dict = {}

        print('\tLoading Captions JSON...')
        capt_filepath = join(annot_path, 'captions_%s.json' % (dataset_name,))
        if exists(capt_filepath):
            with open(capt_filepath) as capt_file:
                capt_dict = json.load(capt_file)
        else:
            capt_dict = {}

        print('\tLoading Info JSON...')
        info_filepath = join(annot_path, 'image_info_%s.json' % (dataset_name,))
        if exists(info_filepath):
            with open(info_filepath) as info_file:
                info_dict = json.load(info_file)
        else:
            info_dict = {}

        if dataset_name == 'test2014':
            print('\tAssociate Test 2014')
            image_list = (
                inst_dict.get('images', [])
                + capt_dict.get('images', [])
                + info_dict.get('images', [])
            )
            id_dict = {}
            for image in image_list:
                image_id = image['id']
                test2014_seen_set.add(image_id)

            # Skip remaining processing to prevent duplicates
            continue

        print('\tLoading Licenses JSON...')
        lic_list = (
            inst_dict.get('licenses', [])
            + capt_dict.get('licenses', [])
            + info_dict.get('licenses', [])
        )
        for lic in lic_list:
            lic_name = lic['name']
            lic_url = lic['url']
            lic_id = lic['id']
            lic_dict_ = {
                'name': lic_name,
                'url': lic_url,
            }
            if lic_id in lic_dict:
                assert lic_dict[lic_id] == lic_dict_
            else:
                lic_dict[lic_id] = lic_dict_

        print('\tLoading Categories JSON...')
        cat_list = (
            inst_dict.get('categories', [])
            + capt_dict.get('categories', [])
            + info_dict.get('categories', [])
        )
        for cat in cat_list:
            cat_name = cat['name']
            cat_super = cat['supercategory']
            cat_id = cat['id']
            cat_dict_ = {
                'name': cat_name,
                'super': cat_super,
            }
            if cat_id in cat_dict:
                assert cat_dict[cat_id] == cat_dict_
            else:
                cat_dict[cat_id] = cat_dict_

        print('\tLoading Images...')
        image_path = join(dbdir, dataset_name)
        assert exists(image_path)

        image_filepath_list = ut.list_images(image_path, recursive=True, full=False)
        for image_filepath in image_filepath_list:
            if image_filepath not in image_dict:
                image_dict[image_filepath] = {}

        print('\tAssociate Images')
        image_list = (
            inst_dict.get('images', [])
            + capt_dict.get('images', [])
            + info_dict.get('images', [])
        )
        id_dict = {}
        for image in image_list:
            # Get file name
            image_filename = image['file_name']
            assert image_filename in image_dict
            # Get file ID
            image_id = image['id']
            if image_id in id_dict:
                assert id_dict[image_id] == image_filename
            else:
                id_dict[image_id] = image_filename
            # Add dataset name
            if 'datasets' not in image_dict[image_filename]:
                image_dict[image_filename]['datasets'] = set([])
            image_dict[image_filename]['datasets'].add(dataset_name)
            # Add test2014, if needed
            if dataset_name == 'test2015' and image_id in test2014_seen_set:
                image_dict[image_filename]['datasets'].add('test2014')
            # Get image's filepath
            image_filepath = abspath(join(image_path, image_filename))
            if 'filepath' in image_dict[image_filename]:
                assert image_dict[image_filename]['filepath'] == image_filepath
            else:
                image_dict[image_filename]['filepath'] = image_filepath
            # Get keys
            key_list = [
                'date_captured',
                'id',
                'coco_url',
                'license',
            ]
            for key in key_list:
                if key in image_dict[image_filename]:
                    assert image_dict[image_filename][key] == image[key]
                else:
                    image_dict[image_filename][key] = image[key]

        print('\tAssociate Captions')
        for caption in capt_dict.get('annotations', []):
            capt_id = caption['id']
            capt_str = caption['caption']
            image_id = caption['image_id']
            assert image_id in id_dict
            image_filename = id_dict[image_id]
            assert image_filename in image_dict
            if 'caption' not in image_dict[image_filename]:
                image_dict[image_filename]['captions'] = []
            image_dict[image_filename]['captions'].append(
                {'id': capt_id, 'str': capt_str}
            )

        print('\tAssociate Annotations')
        for annotation in inst_dict.get('annotations', []):
            annot_id = annotation['id']
            annot_bbox = annotation['bbox']
            annot_seg_list = annotation['segmentation']
            annot_verts_list = []
            for index, annot_seg in enumerate(annot_seg_list):
                annot_vert_list = []
                for index in range(len(annot_seg) // 2):
                    annot_vert_list.append(
                        (annot_seg[index * 2], annot_seg[index * 2 + 1],)
                    )

                annot_verts_list.append(annot_vert_list)
            annot_cat = annotation['category_id']
            image_id = annotation['image_id']
            annot_iscrowd = annotation['iscrowd'] == 1
            assert image_id in id_dict
            image_filename = id_dict[image_id]
            assert image_filename in image_dict
            if 'annotations' not in image_dict[image_filename]:
                image_dict[image_filename]['annotations'] = []
            image_dict[image_filename]['annotations'].append(
                {
                    'id': annot_id,
                    'bbox': annot_bbox,
                    'segmentations': annot_verts_list,
                    'category': annot_cat,
                    'crowd': annot_iscrowd,
                }
            )

    image_filename_list = sorted(image_dict.keys())
    image_filepath_list = []
    image_original_url_list = []
    image_unixtime_list = []
    image_metadata_list = []
    image_imagesets_list = []
    image_annot_bbox_list_list = []
    image_annot_species_list_list = []
    image_annot_metadata_list_list = []
    for image_filename in image_filename_list:
        print('Processing: %r' % (image_filename,))
        image = image_dict[image_filename]
        image_filepath_list.append(image['filepath'])
        image_original_url_list.append(image['coco_url'])
        image_datetime = datetime.datetime.strptime(
            image['date_captured'], '%Y-%m-%d %H:%M:%S'
        )
        image_unixtime = int(image_datetime.strftime('%s'))
        image_unixtime_list.append(image_unixtime)
        image_dataset_list = sorted(list(image['datasets']))
        image_metadata_dict = {
            'coco': {
                'dates': {'captured': image['date_captured']},
                'datasets': image_dataset_list,
                'license': lic_dict[image['license']],
                'captions': image.get('captions', None),
                'flickr': {'url': image.get('flickr_url', None)},
                'url': image['coco_url'],
                'id': image['id'],
            },
        }
        image_metadata_list.append(image_metadata_dict)
        # Get the Imagesets for the datasets
        image_imageset_list = [
            'DATASET: %s' % (image_dataset,) for image_dataset in image_dataset_list
        ]
        if len(image_imageset_list) == 0:
            image_imageset_list.append('DATASET: UNSPECIFIED')
        # Get the Imagesets for the annotations
        image_category_set = set([])
        image_super_category_set = set([])
        image_annot_bbox_list = []
        image_annot_species_list = []
        image_annot_metadata_list = []
        for annotation in image.get('annotations', []):
            cat = cat_dict[annotation['category']]
            image_annot_bbox_list.append(annotation['bbox'])
            image_annot_species_list.append(cat['name'])
            image_annot_metadata_dict = {
                'coco': {
                    'id': annotation['id'],
                    'segmentations': annotation['segmentations'],
                },
            }
            image_annot_metadata_list.append(image_annot_metadata_dict)
            # Get categories
            image_category_set.add(cat['name'])
            if annotation['crowd']:
                image_category_set.add('crowd')
            image_super_category_set.add(cat['super'])

        assert len(image_annot_bbox_list) == len(image_annot_species_list)
        assert len(image_annot_species_list) == len(image_annot_metadata_list)
        image_annot_bbox_list_list.append(image_annot_bbox_list)
        image_annot_species_list_list.append(image_annot_species_list)
        image_annot_metadata_list_list.append(image_annot_metadata_list)
        # Add categories
        image_imageset_list += [
            'CATEGORY: %s' % (image_category,)
            for image_category in sorted(list(image_category_set))
        ]
        image_imageset_list += [
            'SUPER CATEGORY: %s' % (image_super_category,)
            for image_super_category in sorted(list(image_super_category_set))
        ]
        image_imagesets_list.append(image_imageset_list)

    if not dryrun:
        ibs = wbia.opendb(dbdir, allow_newdir=True)

        # Add images to the database
        gid_list = ibs.add_images(image_filepath_list)

        seen_set = set([])
        flag_list = []
        for gid in gid_list:
            flag_list.append(gid not in seen_set)
            seen_set.add(gid)
        print('Duplicates: %d' % (flag_list.count(False),))
        gid_list = ut.filter_items(gid_list, flag_list)

        # Filter
        image_filepath_list = ut.filter_items(image_filepath_list, flag_list)
        image_original_url_list = ut.filter_items(image_original_url_list, flag_list)
        image_unixtime_list = ut.filter_items(image_unixtime_list, flag_list)
        image_metadata_list = ut.filter_items(image_metadata_list, flag_list)
        image_imagesets_list = ut.filter_items(image_imagesets_list, flag_list)
        image_annot_bbox_list_list = ut.filter_items(
            image_annot_bbox_list_list, flag_list
        )
        image_annot_species_list_list = ut.filter_items(
            image_annot_species_list_list, flag_list
        )
        image_annot_metadata_list_list = ut.filter_items(
            image_annot_metadata_list_list, flag_list
        )

        print('Adding Metadata')
        # Set image metadata
        ibs.set_image_uris_original(gid_list, image_original_url_list, overwrite=True)
        ibs.set_image_unixtime(gid_list, image_unixtime_list)
        ibs.set_image_metadata(gid_list, image_metadata_list)

        # Add images to imagesets
        print('Adding Imagesets')
        len_list = map(len, image_imagesets_list)
        gids_list = [[gid] * len_ for gid, len_ in zip(gid_list, len_list)]
        gid_list_ = ut.flatten(gids_list)
        image_imageset_list = ut.flatten(image_imagesets_list)
        ibs.set_image_imagesettext(gid_list_, image_imageset_list)

        print('Adding Annotations')
        len_list = map(len, image_annot_bbox_list_list)
        gids_list = [[gid] * len_ for gid, len_ in zip(gid_list, len_list)]
        gid_list_ = ut.flatten(gids_list)

        image_annot_bbox_list = ut.flatten(image_annot_bbox_list_list)
        image_annot_species_list = ut.flatten(image_annot_species_list_list)
        aid_list = ibs.add_annots(
            gid_list_, image_annot_bbox_list, species_list=image_annot_species_list
        )

        seen_set = set([])
        flag_list = []
        for aid in aid_list:
            flag_list.append(aid not in seen_set)
            seen_set.add(aid)
        print('Duplicates: %d' % (flag_list.count(False),))
        aid_list = ut.filter_items(aid_list, flag_list)

        print('Adding Metadata')
        image_annot_metadata_list = ut.flatten(image_annot_metadata_list_list)
        image_annot_metadata_list = ut.filter_items(image_annot_metadata_list, flag_list)
        ibs.set_annot_metadata(aid_list, image_annot_metadata_list)


def ingest_serengeti_mamal_cameratrap(species):
    """
    Downloads data from Serengeti dryad server

    References:
        http://datadryad.org/resource/doi:10.5061/dryad.5pt92
        Swanson AB, Kosmala M, Lintott CJ, Simpson RJ, Smith A, Packer C (2015)
        Snapshot Serengeti, high-frequency annotated camera trap images of 40
        mammalian species in an African savanna. Scientific Data 2: 150026.
        http://dx.doi.org/10.1038/sdata.2015.26
        Swanson AB, Kosmala M, Lintott CJ, Simpson RJ, Smith A, Packer C (2015)
        Data from: Snapshot Serengeti, high-frequency annotated camera trap
        images of 40 mammalian species in an African savanna. Dryad Digital
        Repository. http://dx.doi.org/10.5061/dryad.5pt92

    Args:
        species (?):

    CommandLine:
        python -m wbia.dbio.ingest_database --test-ingest_serengeti_mamal_cameratrap --species zebra_plains
        python -m wbia.dbio.ingest_database --test-ingest_serengeti_mamal_cameratrap --species cheetah

    Example:
        >>> # SCRIPT
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> import wbia
        >>> species = ut.get_argval('--species', type_=str, default=wbia.const.TEST_SPECIES.ZEB_PLAIN)
        >>> # species = ut.get_argval('--species', type_=str, default='cheetah')
        >>> result = ingest_serengeti_mamal_cameratrap(species)
        >>> print(result)
    """
    'https://snapshotserengeti.s3.msi.umn.edu/'
    import wbia

    if species is None:
        code = 'ALL'
    elif species == 'zebra_plains':
        code = 'PZ'
    elif species == 'cheetah':
        code = 'CHTH'
    else:
        raise NotImplementedError()

    if species == 'zebra_plains':
        serengeti_sepcies = 'zebra'
    else:
        serengeti_sepcies = species

    print('species = %r' % (species,))
    print('serengeti_sepcies = %r' % (serengeti_sepcies,))

    dbname = code + '_Serengeti'
    print('dbname = %r' % (dbname,))
    dbdir = ut.ensuredir(join(wbia.sysres.get_workdir(), dbname))
    print('dbdir = %r' % (dbdir,))
    image_dir = ut.ensuredir(join(dbdir, 'images'))

    base_url = 'http://datadryad.org/bitstream/handle/10255'
    all_images_url = base_url + '/dryad.86392/all_images.csv'
    consensus_metadata_url = base_url + '/dryad.86348/consensus_data.csv'
    search_effort_url = base_url + '/dryad.86347/search_effort.csv'
    gold_standard_url = base_url + '/dryad.76010/gold_standard_data.csv'

    all_images_fpath = ut.grab_file_url(all_images_url, download_dir=dbdir)
    consensus_metadata_fpath = ut.grab_file_url(
        consensus_metadata_url, download_dir=dbdir
    )
    search_effort_fpath = ut.grab_file_url(search_effort_url, download_dir=dbdir)
    gold_standard_fpath = ut.grab_file_url(gold_standard_url, download_dir=dbdir)

    print('all_images_fpath         = %r' % (all_images_fpath,))
    print('consensus_metadata_fpath = %r' % (consensus_metadata_fpath,))
    print('search_effort_fpath      = %r' % (search_effort_fpath,))
    print('gold_standard_fpath      = %r' % (gold_standard_fpath,))

    def read_csv(csv_fpath):
        import utool as ut

        csv_text = ut.read_from(csv_fpath)
        csv_lines = csv_text.split('\n')
        print(ut.repr2(csv_lines[0:2]))
        csv_data = [
            [field.strip('"').strip('\r') for field in line.split(',')]
            for line in csv_lines
            if len(line) > 0
        ]
        csv_header = csv_data[0]
        csv_data = csv_data[1:]
        return csv_data, csv_header

    def download_image_urls(image_url_info_list):
        # Find ones that we already have
        print('Requested %d downloaded images' % (len(image_url_info_list)))
        full_gpath_list = [
            join(image_dir, basename(gpath)) for gpath in image_url_info_list
        ]
        exists_list = [ut.checkpath(gpath) for gpath in full_gpath_list]
        image_url_info_list_ = ut.compress(image_url_info_list, ut.not_list(exists_list))
        print(
            'Already have %d/%d downloaded images'
            % (
                len(image_url_info_list) - len(image_url_info_list_),
                len(image_url_info_list),
            )
        )
        print('Need to download %d images' % (len(image_url_info_list_)))
        # import sys
        # sys.exit(0)
        # Download the rest
        imgurl_prefix = 'https://snapshotserengeti.s3.msi.umn.edu/'
        image_url_list = [imgurl_prefix + suffix for suffix in image_url_info_list_]
        for img_url in ut.ProgressIter(image_url_list, lbl='Downloading image'):
            ut.grab_file_url(img_url, download_dir=image_dir)
        return full_gpath_list

    # Data contains information about which events have which animals
    if False:
        species_class_csv_data, species_class_header = read_csv(gold_standard_fpath)
        species_class_eventid_list = ut.get_list_column(species_class_csv_data, 0)
        # gold_num_species_annots_list = ut.get_list_column(gold_standard_csv_data, 2)
        species_class_species_list = ut.get_list_column(species_class_csv_data, 2)
        # gold_count_list              = ut.get_list_column(gold_standard_csv_data, 3)
    else:
        species_class_csv_data, species_class_header = read_csv(consensus_metadata_fpath)
        species_class_eventid_list = ut.get_list_column(species_class_csv_data, 0)
        species_class_species_list = ut.get_list_column(species_class_csv_data, 7)

    # Find the zebra events
    serengeti_sepcies_set = sorted(list(set(species_class_species_list)))
    print(
        'serengeti_sepcies_hist = %s'
        % ut.repr2(ut.dict_hist(species_class_species_list), key_order_metric='val')
    )
    # print('serengeti_sepcies_set = %s' % (ut.repr2(serengeti_sepcies_set),))

    assert serengeti_sepcies in serengeti_sepcies_set, 'not a known  seregeti species'
    species_class_chosen_idx_list = ut.list_where(
        [serengeti_sepcies == species_ for species_ in species_class_species_list]
    )
    chosen_eventid_list = ut.take(
        species_class_eventid_list, species_class_chosen_idx_list
    )

    print('Number of chosen species:')
    print(
        ' * len(species_class_chosen_idx_list) = %r'
        % (len(species_class_chosen_idx_list),)
    )
    print(' * len(chosen_eventid_list) = %r' % (len(chosen_eventid_list),))

    # Read info about which events have which images
    images_csv_data, image_csv_header = read_csv(all_images_fpath)
    capture_event_id_list = ut.get_list_column(images_csv_data, 0)
    image_url_info_list = ut.get_list_column(images_csv_data, 1)
    # Group photos by eventid
    eventid_to_photos = ut.group_items(image_url_info_list, capture_event_id_list)

    # Filter to only chosens
    unflat_chosen_url_infos = ut.dict_take(eventid_to_photos, chosen_eventid_list)
    chosen_url_infos = ut.flatten(unflat_chosen_url_infos)
    image_url_info_list = chosen_url_infos
    chosen_path_list = download_image_urls(chosen_url_infos)

    ibs = wbia.opendb(dbdir=dbdir, allow_newdir=True)
    gid_list_ = ibs.add_images(chosen_path_list, auto_localize=False)  # NOQA

    # if False:
    #    # remove non-zebra photos
    #    from os.path import basename
    #    base_gname_list = list(map(basename, zebra_url_infos))
    #    all_gname_list = ut.list_images(image_dir)
    #    nonzebra_gname_list = ut.setdiff_ordered(all_gname_list, base_gname_list)
    #    nonzebra_gpath_list = ut.fnames_to_fpaths(nonzebra_gname_list, image_dir)
    #    ut.remove_fpaths(nonzebra_gpath_list)
    return ibs


def injest_main():
    r"""
    CommandLine:
        python -m wbia.dbio.ingest_database --test-injest_main
        python -m wbia.dbio.ingest_database --test-injest_main --db snow-leopards

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.dbio.ingest_database import *  # NOQA
        >>> injest_main()
    """
    print('__main__ = ingest_database.py')
    print(
        ut.unindent(
            """
        usage:
        python wbia/ingest/ingest_database.py --db [dbname]

        Valid dbnames:"""
        )
        + ut.indentjoin(STANDARD_INGEST_FUNCS.keys(), '\n  * ')
    )
    dbname = ut.get_argval('--db', str, None)
    force_delete = ut.get_argflag(('--force_delete', '--force-delete'))
    ibs = ingest_standard_database(dbname, force_delete)  # NOQA
    print('finished db injest')
    # img_dir = join(wbia.sysres.get_workdir(), 'polar_bears')
    # main_locals = wbia.main(dbdir=img_dir, gui=False)
    # ibs = main_locals['ibs']
    # ingest_rawdata(ibs, img_dir)


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m wbia.dbio.ingest_database
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
