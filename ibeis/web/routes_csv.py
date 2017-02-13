# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from flask import current_app
from ibeis.control import controller_inject
from ibeis.web import appfuncs as appf
import utool as ut
from ibeis.web import routes


register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_route('/csv/sightings/', methods=['GET'])
def download_sightings():
    filename = 'sightings.csv'
    sightings = routes.sightings(html_encode=False)
    return appf.send_csv_file(sightings, filename)


@register_route('/csv/nids_with_gids/', methods=['GET'])
def get_nid_with_gids_csv():
    ibs = current_app.ibs
    filename = 'nids_with_gids.csv'
    combined_dict = ibs.get_name_nids_with_gids()
    combined_list = [
        ','.join( map(str, [nid] + [name] + gid_list) )
        for name, (nid, gid_list) in sorted(list(combined_dict.iteritems()))
    ]
    combined_str = '\n'.join(combined_list)
    max_length = 0
    for aid_list in combined_dict.values():
        max_length = max(max_length, len(aid_list[1]))
    if max_length == 1:
        gid_header_str = 'GID'
    else:
        gid_header_str = ','.join([ 'GID%d' % (i + 1, ) for i in range(max_length) ])
    combined_str = 'NID,NAME,%s\n' % (gid_header_str, ) + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/image_info/', methods=['GET'])
def get_image_info():
    import datetime
    ibs = current_app.ibs
    filename = 'image_info.csv'
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
        except:
            party_list.append('UNKNOWN')
            contributor_list.append('UNKNOWN')

    zipped_list = zip(gid_list, gname_list, datetime_list_, lat_list, lon_list,
                      party_list, contributor_list, note_list)
    aids_list = ibs.get_image_aids(gid_list)
    names_list = [ ibs.get_annot_name_texts(aid_list) for aid_list in aids_list ]
    combined_list = [
        ','.join( map(str, list(zipped) + name_list) )
        for zipped, name_list in zip(zipped_list, names_list)
    ]
    max_length = 0
    for name_list in names_list:
        max_length = max(max_length, len(name_list))
    if max_length == 1:
        name_header_str = 'NAME'
    else:
        name_header_str = ','.join([ 'NAME%d' % (i + 1, ) for i in range(max_length) ])
    combined_str = '\n'.join(combined_list)
    combined_str = 'GID,FILENAME,TIMESTAMP,GPSLAT,GPSLON,PARTY,CONTRIBUTOR,NOTES,%s\n' % (name_header_str, ) + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/demographics/', methods=['GET'])
def get_demographic_info():
    ibs = current_app.ibs
    filename = 'demographics.csv'
    nid_list = sorted(ibs.get_valid_nids())
    name_list = ibs.get_name_texts(nid_list)
    sex_list = ibs.get_name_sex_text(nid_list)
    min_ages_list = ibs.get_name_age_months_est_min(nid_list)
    max_ages_list = ibs.get_name_age_months_est_max(nid_list)

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
        if (min_age is None or min_age < 12) and max_age < 12:
            age_list.append('FOAL')
        elif 12 <= min_age and min_age < 24 and 12 <= max_age and max_age < 24:
            age_list.append('YEARLING')
        elif 24 <= min_age and min_age < 36 and 24 <= max_age and max_age < 36:
            age_list.append('2 YEARS')
        elif 36 <= min_age and (36 <= max_age or max_age is None):
            age_list.append('3+ YEARS')
        else:
            age_list.append('UNKNOWN')

    zipped_list = zip(nid_list, name_list, sex_list, age_list)
    combined_list = [
        ','.join( map(str, list(zipped)) )
        for zipped in zipped_list
    ]
    combined_str = '\n'.join(combined_list)
    combined_str = 'NID,NAME,SEX,AGE\n' + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/gids_with_aids/', methods=['GET'])
def get_gid_with_aids_csv():
    ibs = current_app.ibs
    combined_dict = ibs.get_image_gids_with_aids()
    filename = 'gids_with_aids.csv'
    combined_list = [
        ','.join( map(str, [gid] + aid_list) )
        for gid, aid_list in sorted(list(combined_dict.iteritems()))
    ]
    combined_str = '\n'.join(combined_list)
    max_length = 0
    for aid_list in combined_dict.values():
        max_length = max(max_length, len(aid_list))
    if max_length == 1:
        aid_header_str = 'AID'
    else:
        aid_header_str = ','.join([ 'AID%d' % (i + 1, ) for i in range(max_length) ])
    combined_str = 'GID,%s\n' % (aid_header_str, ) + combined_str
    return appf.send_csv_file(combined_str, filename)


@register_route('/csv/image/', methods=['GET'])
def get_gid_list_csv():
    filename = 'gids.csv'
    ibs = current_app.ibs
    gid_list = ibs.get_valid_gids()
    return_str = '\n'.join( map(str, gid_list) )
    return_str = 'GID\n' + return_str
    return appf.send_csv_file(return_str, filename)


@register_route('/csv/annot/', methods=['GET'])
def get_aid_list_csv():
    filename = 'aids.csv'
    ibs = current_app.ibs
    aid_list = ibs.get_valid_aids()
    return_str = '\n'.join( map(str, aid_list) )
    return_str = 'AID\n' + return_str
    return appf.send_csv_file(return_str, filename)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
