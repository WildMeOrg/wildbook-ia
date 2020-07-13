# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from flask import current_app
from wbia.control import controller_inject
from wbia.web import appfuncs as appf
import utool as ut
from wbia.web import routes

(print, rrr, profile) = ut.inject2(__name__)

register_route = controller_inject.get_wbia_flask_route(__name__)


def get_associations_dict(ibs, desired_species=None, **kwargs):
    import itertools

    if ibs.dbname == 'ZEBRA_Kaia':
        valid_aid_set = set(ibs.get_valid_aids())
        imageset_list = ibs._princeton_kaia_imageset_filtering(**kwargs)
    else:
        valid_aid_set = set(ibs.get_valid_aids())
        imageset_list = ibs.get_valid_imgsetids(is_special=False)

    valid_nid_set = ibs.get_annot_nids(valid_aid_set)
    valid_nid_set = set([nid for nid in valid_nid_set if nid > 0])

    imageset_text_list = ibs.get_imageset_text(imageset_list)
    time_list = ibs.get_imageset_start_time_posix(imageset_list)
    nids_list = ibs.get_imageset_nids(imageset_list)

    ibs.delete_empty_nids()

    def _associate(dict_, name1, name2, label):
        if name2 < name1:
            name1, name2 = name2, name1
        if name1 not in dict_:
            dict_[name1] = {}
        if name2 not in dict_[name1]:
            dict_[name1][name2] = []
        dict_[name1][name2].append('%s' % (label,))

    assoc_dict = {}
    for imageset_text, time_, nid_list in zip(imageset_text_list, time_list, nids_list):
        if desired_species is not None:
            aids_list = ibs.get_name_aids(nid_list)

            flag_list = []
            for nid, aid_list in zip(nid_list, aids_list):
                aid_list = list(set(aid_list) & set(valid_aid_set))
                if len(aid_list) == 0:
                    flag = False
                else:
                    species_list = ibs.get_annot_species(aid_list)
                    species = max(set(species_list), key=species_list.count)
                    if ibs.dbname == 'ZEBRA_Kaia' and desired_species == 'zebra':
                        flag = species in ['zebra_plains', 'zebra_grevys']
                    else:
                        flag = species == desired_species
                flag_list.append(flag)
            nid_list = ut.compress(nid_list, flag_list)

        nid_list = list(set(nid_list) & valid_nid_set)
        name_list = ibs.get_name_texts(nid_list)
        # Add singles
        for name in name_list:
            _associate(assoc_dict, name, name, imageset_text)
            # _associate(assoc_dict, name, name, time_)
        # Add pairs
        comb_list = itertools.combinations(name_list, 2)
        for name1, name2 in sorted(list(comb_list)):
            _associate(assoc_dict, name1, name2, imageset_text)
            # _associate(assoc_dict, name, name, time_)

    return assoc_dict


@register_route('/csv/princeton/associations/list/', methods=['GET'])
def download_associations_list(**kwargs):
    ibs = current_app.ibs

    key_str_list = []
    for key in sorted(kwargs.keys()):
        key_str = '%s=%s' % (key, kwargs[key],)
        key_str_list.append(key_str)
    key_str = '.'.join(key_str_list)
    if len(key_str) > 0:
        key_str += '.'

    filename = 'associations.list.%scsv' % (key_str,)
    assoc_dict = get_associations_dict(ibs, **kwargs)

    combined_list = []
    max_length = 0
    for name1 in sorted(assoc_dict.keys()):
        for name2 in sorted(assoc_dict[name1].keys()):
            id_list = sorted(set(assoc_dict[name1][name2]))
            id_list = [id_.replace(',', ':COMMA:') for id_ in id_list]
            max_length = max(max_length, len(id_list))
            args = (
                name1,
                name2,
                len(id_list),
                ','.join(id_list),
            )
            combined_str = '%s,%s,%s,%s' % args
            combined_list.append(combined_str)

    if max_length == 1:
        # name_header_str = 'TIME'
        name_header_str = 'ENCOUTNER'
    else:
        # name_header_str = ','.join([ 'TIME%d' % (i + 1, ) for i in range(max_length) ])
        name_header_str = ','.join(['ENCOUNTER%d' % (i + 1,) for i in range(max_length)])
    combined_str = '\n'.join(combined_list)
    combined_str = 'NAME1,NAME2,ASSOCIATIONS,%s\n' % (name_header_str,) + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/princeton/associations/matrix/', methods=['GET'])
def download_associations_matrix(**kwargs):
    ibs = current_app.ibs

    key_str_list = []
    for key in sorted(kwargs.keys()):
        key_str = '%s=%s' % (key, kwargs[key],)
        key_str_list.append(key_str)
    key_str = '.'.join(key_str_list)
    if len(key_str) > 0:
        key_str += '.'

    filename = 'associations.matrix.%scsv' % (key_str,)

    assoc_dict = get_associations_dict(ibs, **kwargs)
    assoc_list = sorted(assoc_dict.keys())
    # max_length = len(assoc_list)

    combined_list = []
    for index1, name1 in enumerate(assoc_list):
        temp_list = [name1]
        for index2, name2 in enumerate(assoc_list):
            value = assoc_dict[name1].get(name2, [])
            value_len = len(value)
            value_str = '' if value_len == 0 else value_len
            temp_list.append('%s' % (value_str,))
        temp_str = ','.join(temp_list)
        combined_list.append(temp_str)

    # if max_length == 1:
    #     name_header_str = 'NAME'
    # else:
    #     name_header_str = ','.join([ 'NAME%d' % (i + 1, ) for i in range(max_length) ])
    name_header_str = ','.join(assoc_list)
    combined_str = '\n'.join(combined_list)
    combined_str = 'MATRIX,%s\n' % (name_header_str,) + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/princeton/sightings/', methods=['GET'])
def download_sightings(**kwargs):
    ibs = current_app.ibs
    filename = 'sightings.csv'
    kaia = ibs.dbname == 'ZEBRA_Kaia'
    sightings = routes.sightings(html_encode=False, kaia=kaia, **kwargs)
    return appf.send_csv_file(sightings, filename)


@register_route('/csv/princeton/images/', methods=['GET'])
def get_image_info(**kwargs):
    import datetime

    ibs = current_app.ibs
    filename = 'images.csv'

    valid_aid_set = set(ibs.get_valid_aids())

    gid_list = sorted(ibs.get_valid_gids())
    gname_list = ibs.get_image_gnames(gid_list)
    datetime_list = ibs.get_image_unixtime(gid_list)
    datetime_list_ = [
        datetime.datetime.fromtimestamp(datetime_).strftime('%Y-%m-%d %H:%M:%S')
        for datetime_ in datetime_list
    ]
    lat_list = ibs.get_image_lat(gid_list)
    lon_list = ibs.get_image_lon(gid_list)
    note_list = ibs.get_image_notes(gid_list)
    party_list = []
    contributor_list = []
    for note in note_list:
        try:
            note = note.split(',')
            party, contributor = note[:2]
            party_list.append(party)
            contributor_list.append(contributor)
        except Exception:
            party_list.append('UNKNOWN')
            contributor_list.append('UNKNOWN')

    zipped_list = zip(
        gid_list,
        gname_list,
        datetime_list_,
        lat_list,
        lon_list,
        party_list,
        contributor_list,
        note_list,
    )
    aids_list = ibs.get_image_aids(gid_list)
    aids_list = [
        [aid_ for aid_ in aid_list_ if aid_ in valid_aid_set] for aid_list_ in aids_list
    ]
    names_list = [ibs.get_annot_name_texts(aid_list) for aid_list in aids_list]
    combined_list = [
        ','.join(map(str, list(zipped) + name_list))
        for zipped, aid_list, name_list in zip(zipped_list, aids_list, names_list)
        if ibs.dbdir != 'ZEBRA_Kaia' or len(aid_list) > 0
    ]
    max_length = 0
    for name_list in names_list:
        max_length = max(max_length, len(name_list))
    if max_length == 1:
        name_header_str = 'NAME'
    else:
        name_header_str = ','.join(['NAME%d' % (i + 1,) for i in range(max_length)])
    combined_str = '\n'.join(combined_list)
    combined_str = (
        'GID,FILENAME,TIMESTAMP,GPSLAT,GPSLON,PARTY,CONTRIBUTOR,NOTES,%s\n'
        % (name_header_str,)
        + combined_str
    )
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/princeton/demographics/', methods=['GET'])
def get_demographic_info(**kwargs):
    ibs = current_app.ibs
    filename = 'demographics.csv'
    nid_list = sorted(ibs.get_valid_nids())
    name_list = ibs.get_name_texts(nid_list)
    sex_list = ibs.get_name_sex_text(nid_list)
    min_ages_list = ibs.get_name_age_months_est_min(nid_list)
    max_ages_list = ibs.get_name_age_months_est_max(nid_list)

    def _get_primary_species(aid_list):
        if len(aid_list) == 0:
            species = None
        else:
            species_list = ibs.get_annot_species_texts(aid_list)
            species = max(set(species_list), key=species_list.count)
        return species

    age_list = []
    for min_ages, max_ages in zip(min_ages_list, max_ages_list):
        if len(set(min_ages)) > 1 or len(set(max_ages)) > 1:
            age_list.append('AMBIGUOUS')
            continue
        min_age = None
        max_age = None
        if len(min_ages) > 0:
            min_age = min_ages[0]
        if len(max_ages) > 0:
            max_age = max_ages[0]
        # Histogram
        if (min_age is None and max_age is None) or (min_age is -1 and max_age is -1):
            age_list.append('UNREVIEWED')
            continue
        # Bins
        if min_age is None and max_age == 2:
            age = '0-3 Months'
        elif min_age == 3 and max_age == 5:
            age = '3-6 Months'
        elif min_age == 6 and max_age == 11:
            age = '6-12 Months'
        elif min_age == 12 and max_age == 23:
            age = 'Yearling'
        elif min_age == 24 and max_age == 35:
            age = '2-Year-Old'
        elif min_age == 36 and max_age is None:
            age = 'Adult'
        elif min_age is None and max_age is None:
            age = 'Unknown'
        else:
            age = 'Unknown'
        age_list.append(age)

    zipped_list = zip(nid_list, name_list, sex_list, age_list)
    combined_list = [','.join(map(str, list(zipped))) for zipped in zipped_list]
    combined_str = '\n'.join(combined_list)
    combined_str = 'NID,NAME,SEX,AGE\n' + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/princeton/special/kaia-dung-samples/', methods=['GET'])
def get_annotation_special_kaia_dung_samples(**kwargs):
    ibs = current_app.ibs

    filename = 'dungsamples.csv'
    aid_list = ibs.get_valid_aids()
    dungsample_list = ibs.get_annot_notes(aid_list)
    flag_list = [len(dungsample) > 0 for dungsample in dungsample_list]
    aid_list = ut.compress(aid_list, flag_list)
    dungsample_list = ut.compress(dungsample_list, flag_list)
    nid_list = ibs.get_annot_nids(aid_list)
    name_list = ibs.get_annot_name_texts(aid_list)

    assoc_dict = get_associations_dict(ibs, desired_species='zebra', tier=1)
    encounter_str_list = []
    max_length = 0
    for name in sorted(name_list):
        id_list = sorted(set(assoc_dict[name][name]))
        id_list = [id_.replace(',', ':COMMA:') for id_ in id_list]
        max_length = max(max_length, len(id_list))
        encounter_str = ','.join(id_list)
        encounter_str_list.append(encounter_str)

    sex_list = ibs.get_annot_sex(aid_list)
    age_min_list = ibs.get_annot_age_months_est_min(aid_list)
    age_max_list = ibs.get_annot_age_months_est_max(aid_list)
    condition_list = ibs.get_annot_qualities(aid_list)

    sex_list_ = []
    age_list_ = []
    condition_list_ = []
    zipped = zip(sex_list, age_min_list, age_max_list, condition_list)
    for sex_, age_min, age_max, condition_ in zipped:
        if sex_ == 1:
            sex_ = 'male'
        elif sex_ == 0:
            sex_ = 'female'
        else:
            sex_ = 'unknown'

        if age_min is None and age_max == 2:
            age_ = 'age1'
        elif age_min == 3 and age_max == 5:
            age_ = 'age2'
        elif age_min == 6 and age_max == 11:
            age_ = 'age3'
        elif age_min == 12 and age_max == 23:
            age_ = 'age4'
        elif age_min == 24 and age_max == 35:
            age_ = 'age5'
        elif age_min == 36 and age_max is None:
            age_ = 'age6'
        elif age_min is None and age_max is None:
            age_ = 'unknown'
        else:
            age_ = 'unknown'

        if condition_ is None:
            condition_ = 0

        assert age_ in ['age1', 'age2', 'age3', 'age4', 'age5', 'age6', 'unknown']
        assert sex_ in ['male', 'female', 'unknown']
        assert 0 <= condition_ and condition_ <= 5

        if sex_ == 'male':
            sex_ = 'Male'
        elif sex_ == 'female':
            sex_ = 'Female'
        elif sex_ == 'unknown':
            sex_ = 'Unknown'
        else:
            sex_ = 'Unknown'

        if age_ == 'age1':
            age_ = '0-3 Months'
        elif age_ == 'age2':
            age_ = '3-6 Months'
        elif age_ == 'age3':
            age_ = '6-12 Months'
        elif age_ == 'age4':
            age_ = 'Yearling'
        elif age_ == 'age5':
            age_ = '2-Year-Old'
        elif age_ == 'age6':
            age_ = 'Adult'
        elif age_ == 'unknown':
            age_ = 'Unknown'
        else:
            age_ = 'Unknown'

        sex_list_.append(sex_)
        age_list_.append(age_)
        condition_list_.append(condition_)

    if max_length == 1:
        name_header_str = 'ENCOUTNER'
    else:
        name_header_str = ','.join(['ENCOUNTER%d' % (i + 1,) for i in range(max_length)])

    zipped_list = sorted(
        zip(
            name_list,
            nid_list,
            aid_list,
            dungsample_list,
            age_list_,
            sex_list_,
            condition_list_,
            encounter_str_list,
        )
    )
    combined_list = [','.join(map(str, list(zipped_))) for zipped_ in zipped_list]
    combined_str = '\n'.join(combined_list)
    combined_str = (
        'NAME,NID,AID,DUNGSAMPLE,AGE,SEX,CONDITION,%s\n' % (name_header_str,)
        + combined_str
    )
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/princeton/special/monica-laurel-max/', methods=['GET'])
def get_annotation_special_monica_laurel_max(desired_species=None, **kwargs):
    ibs = current_app.ibs
    filename = 'special.monica-laurel-max.csv'

    def _process_annot_name_uuids_dict(ibs, filepath):
        import uuid

        auuid_list = []
        nuuid_list = []
        with open(filepath, 'r') as file_:
            for line in file_.readlines():
                line = line.strip().split(',')
                auuid = uuid.UUID(line[0])
                nuuid = None if line[1] == 'None' else uuid.UUID(line[1])
                auuid_list.append(auuid)
                nuuid_list.append(nuuid)

        annot_rowid_list = ibs.get_annot_aids_from_uuid(auuid_list)
        name_rowid_list = ibs.get_name_rowids_from_uuid(nuuid_list)

        zipped = zip(annot_rowid_list, name_rowid_list)
        mapping_dict = {aid: nid for aid, nid in zipped if aid is not None}
        return mapping_dict

    aid_list = sorted(ibs.get_valid_aids())
    annot_uuid_list = ibs.get_annot_uuids(aid_list)
    print('Found %d aids' % (len(aid_list),))
    nid_list = ibs.get_annot_nids(aid_list)
    name_uuid_list = ibs.get_name_uuids(nid_list)
    name_list = ibs.get_name_texts(nid_list)
    species_list = ibs.get_annot_species_texts(aid_list)
    sex_list = ibs.get_name_sex_text(nid_list)
    age_list = ibs.get_annot_age_months_est_texts(aid_list)
    gid_list = ibs.get_annot_gids(aid_list)
    contrib_list = ibs.get_image_contributor_tag(gid_list)
    gname_list = ibs.get_image_gnames(gid_list)
    imageset_rowids_list = ibs.get_image_imgsetids(gid_list)
    imageset_rowids_set = map(set, imageset_rowids_list)

    special_imageset_rowid_set = set(ibs.get_valid_imgsetids(is_special=True))
    imagesets_list = [
        list(imageset_rowid_set - special_imageset_rowid_set)
        for imageset_rowid_set in imageset_rowids_set
    ]

    imageset_list = [_[0] if len(_) > 0 else None for _ in imagesets_list]
    imageset_text_list = ibs.get_imageset_text(imageset_list)
    imageset_metadata_list = ibs.get_imageset_metadata(imageset_list)
    annot_metadata_list = ibs.get_annot_metadata(aid_list)

    assert len(imageset_metadata_list) == len(annot_metadata_list)

    imageset_metadata_list_ = ibs.get_imageset_metadata(ibs.get_valid_imgsetids())
    imageset_metadata_key_list = sorted(
        set(
            ut.flatten(
                [
                    imageset_metadata_dict_.keys()
                    for imageset_metadata_dict_ in imageset_metadata_list_
                ]
            )
        )
    )
    imageset_metadata_key_str = ','.join(imageset_metadata_key_list)

    annot_metadata_list_ = ibs.get_annot_metadata(ibs.get_valid_aids())
    annot_metadata_key_list = sorted(
        set(
            ut.flatten(
                [
                    annot_metadata_dict_.keys()
                    for annot_metadata_dict_ in annot_metadata_list_
                ]
            )
        )
    )
    annot_metadata_key_str = ','.join(annot_metadata_key_list)

    if 'Monica-Laurel' in ibs.dbdir:
        import wbia

        ibs1 = wbia.opendb('/home/zebra/Desktop/Monica/', web=False)
        ibs2 = wbia.opendb('/home/zebra/Desktop/Laurel/', web=False)
    if 'Monica-Max' in ibs.dbdir:
        import wbia

        ibs1 = wbia.opendb('/home/zebra/Desktop/Monica/', web=False)
        ibs2 = wbia.opendb('/home/zebra/Desktop/Max/', web=False)
    else:
        ibs1 = None
        ibs2 = None

    line_list = []
    zipped = zip(
        nid_list,
        aid_list,
        name_uuid_list,
        annot_uuid_list,
        name_list,
        species_list,
        sex_list,
        age_list,
        gname_list,
        contrib_list,
        imageset_list,
        imageset_text_list,
        imageset_metadata_list,
        annot_metadata_list,
    )
    zipped = sorted(list(zipped))
    for args in zipped:
        (
            nid,
            aid,
            name_uuid,
            annot_uuid,
            name,
            species,
            sex,
            age,
            gname,
            contrib,
            imageset_rowid,
            imageset_text,
            imageset_metadata_dict,
            annot_metadata_dict,
        ) = args

        contrib_str = '' if contrib is None else contrib.split(',')[0].upper()

        if desired_species is not None and species != desired_species:
            continue

        if nid <= 0:
            continue

        nid_old = ''
        name_old = ''
        name_changed = False
        cross_database_match = False

        try:
            if ibs1 is not None and ibs2 is not None:
                aid1 = ibs1.get_annot_aids_from_uuid(annot_uuid)
                aid2 = ibs2.get_annot_aids_from_uuid(annot_uuid)

                if aid1 is not None:
                    assert aid2 is None
                    name_uuid_old = ibs1.get_annot_name_uuids(aid1)
                elif aid2 is not None:
                    assert aid1 is None
                    name_uuid_old = ibs2.get_annot_name_uuids(aid2)
                else:
                    raise AssertionError('Should be in one of these original databases')

                if name_uuid_old != name_uuid:
                    name_changed = True
                    if name_uuid_old is None:
                        nid_old = 'UNKNOWN NID'
                        name_old = 'UNKNOWN NAME'
                    else:
                        nid_old = ibs.get_name_rowids_from_uuid(name_uuid_old)
                        name_old = ibs.get_name_texts(nid_old)

                    cross_database_match = not name.startswith(contrib_str)
        except Exception:
            print('ERROR WITH ABOVE')
            ut.embed()

        line_list_ = (
            [
                contrib_str,
                annot_uuid,
                aid,
                annot_uuid,
                nid,
                name,
                'Yes' if name_changed else '',
                'Yes' if cross_database_match else '',
                nid_old,
                name_old,
                species,
                sex,
                age,
                gname,
                imageset_rowid,
                imageset_text,
                '|',
            ]
            + [
                imageset_metadata_dict.get(imageset_metadata_key, '')
                for imageset_metadata_key in imageset_metadata_key_list
            ]
            + ['|']
            + [
                annot_metadata_dict.get(annot_metadata_key, '')
                for annot_metadata_key in annot_metadata_key_list
            ]
        )
        line_list_ = ['' if item is None else item for item in line_list_]
        line = ','.join(map(str, line_list_))
        line_list.append(line)

    combined_str = '\n'.join(line_list)
    combined_str = (
        'DB,Annotation UUID,AID,NID,Name,Name Changed,Cross-Database Match,Old NID,Old Name,Species,Sex,Age,Image Name,Encounter ID,Encounter Name,| SEPERATOR |,%s,| SEPERATOR |,%s\n'
        % (imageset_metadata_key_str, annot_metadata_key_str,)
        + combined_str
    )
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/princeton/special/megan/', methods=['GET'])
def get_annotation_special_megan(**kwargs):
    ibs = current_app.ibs
    filename = 'special.megan.csv'

    gid_list = ibs.get_valid_gids()
    uri_list = ibs.get_image_uris(gid_list)
    path_list = ibs.get_image_uris_original(gid_list)

    tag_list = []
    date1_list = []
    date2_list = []
    note_list = []
    for path, gid in zip(path_list, gid_list):
        path = path.replace('/home/zebra/Desktop/MEGAN/', '')
        path = path.split('/')
        base = path[0]
        if base == 'SIDE1_3-22-2016-5-11-2016':  # HACK
            base = 'SIDE1_3-22-2016_5-11-2016'
        path = base.split('_')
        try:
            assert len(path) >= 3
            if len(path) > 3:
                note = path[3:]
                note = '_'.join(note)
                path = path[:3]
            else:
                note = ''
            assert len(path) == 3
            tag, date1, date2 = path
            date1 = date1.replace('-', '/')
            date2 = date2.replace('-', '/')
            tag_list.append(tag)
            date1_list.append(date1)
            date2_list.append(date2)
            note_list.append(note)
        except AssertionError:
            print('ERROR PROCESSING: %r' % (base,))

    zipped = zip(gid_list, uri_list, tag_list, date1_list, date2_list, note_list)
    combined_list = [','.join(map(str, value_list)) for value_list in zipped]
    combined_str = '\n'.join(combined_list)
    combined_str = 'GID,FILENAME,LOCATION,DATE1,DATE2,NOTE\n' + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/nids_with_gids/', methods=['GET'])
def get_nid_with_gids_csv(**kwargs):
    ibs = current_app.ibs
    filename = 'nids_with_gids.csv'
    combined_dict = ibs.get_name_nids_with_gids()
    combined_list = [
        ','.join(map(str, [nid] + [name] + gid_list))
        for name, (nid, gid_list) in sorted(list(combined_dict.iteritems()))
    ]
    combined_str = '\n'.join(combined_list)
    max_length = 0
    for aid_list in combined_dict.values(**kwargs):
        max_length = max(max_length, len(aid_list[1]))
    if max_length == 1:
        gid_header_str = 'GID'
    else:
        gid_header_str = ','.join(['GID%d' % (i + 1,) for i in range(max_length)])
    combined_str = 'NID,NAME,%s\n' % (gid_header_str,) + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/gids_with_aids/', methods=['GET'])
def get_gid_with_aids_csv(**kwargs):
    ibs = current_app.ibs
    combined_dict = ibs.get_image_gids_with_aids()
    filename = 'gids_with_aids.csv'
    combined_list = [
        ','.join(map(str, [gid] + aid_list))
        for gid, aid_list in sorted(list(combined_dict.iteritems()))
    ]
    combined_str = '\n'.join(combined_list)
    max_length = 0
    for aid_list in combined_dict.values(**kwargs):
        max_length = max(max_length, len(aid_list))
    if max_length == 1:
        aid_header_str = 'AID'
    else:
        aid_header_str = ','.join(['AID%d' % (i + 1,) for i in range(max_length)])
    combined_str = 'GID,%s\n' % (aid_header_str,) + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/image/', methods=['GET'])
def get_gid_list_csv(**kwargs):
    filename = 'gids.csv'
    ibs = current_app.ibs
    gid_list = ibs.get_valid_gids()
    return_str = '\n'.join(map(str, gid_list))
    return_str = 'GID\n' + return_str
    return appf.send_csv_file(return_str, filename)


@register_route('/csv/annot/', methods=['GET'])
def get_aid_list_csv(**kwargs):
    filename = 'aids.csv'
    ibs = current_app.ibs
    aid_list = ibs.get_valid_aids()
    return_str = '\n'.join(map(str, aid_list))
    return_str = 'AID\n' + return_str
    return appf.send_csv_file(return_str, filename)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.web.app
        python -m wbia.web.app --allexamples
        python -m wbia.web.app --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
