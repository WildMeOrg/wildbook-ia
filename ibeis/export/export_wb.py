#!/usr/bin/env python
"""
Converts an IBEIS database to a wildbook db
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#import ibeis
import utool
import json
import requests
from itertools import izip
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[export_wb]')


def export_ibeis_to_wildbook(ibs, eid_list):
    target_url_encounter = ibs.wbaddr + '/rest/org.ecocean.Encounter'
    target_url_image = ibs.wbaddr + '/EncounterAddImage'
    # list of tuples, each has the names associated with the encounter
    names_list = ibs.get_encounter_nids(eid_list)
    # list of lists, each has the rois associated with an encounter
    rids_list = ibs.get_encounter_rids(eid_list)
    images_list = ibs.get_encounter_gids(eid_list)

    assert len(names_list) == len(rids_list) == len(images_list) == len(eid_list)
    for eid, nids, rois, images in izip(eid_list, names_list, rids_list, images_list):
        # the actual names corresponding to the name ids
        names_text = ibs.get_names(nids)
        # list of list of images that correspond to each name id
        images_lists = ibs.get_name_gids(nids)

        assert len(nids) == len(names_text) == len(images_lists)
        # creates wildbook encounters for each name in an ibeis encounter
        for name_id, name_text, images in izip(nids, names_text, images_lists):
            # the actual text encounter name corresponding to the encounter id
            enctext = ibs.get_encounter_enctext(eid)
            wbenc_name = enctext + '_' + str(name_id)
            payload_encounter = {'dwcImageURL': ibs.wbaddr + '/encounters/encounter.jsp?number=' + wbenc_name,
                                 'state': 'unapproved',
                                 'dateInMilliseconds': 100,
                                 'approved': True,
                                 'catalogNumber': wbenc_name,
                                 'recordedBy': 'hotspotter',
                                 'individualID': 'Unassigned',
                                 'class': 'org.ecocean.Encounter',
                                 'guid': 'MY_CATALOG:MY_SPECIES:' + wbenc_name,
                                 'decimalLatitude': '-1.0',
                                 'decimalLongitude': '-1.0'}

            response = requests.post(target_url_encounter, data=json.dumps(payload_encounter))
            print ('POSTed %s to %s with status %d' % (wbenc_name, target_url_encounter, response.status_code))
            # get the paths to all the images in which this name id appears
            paths = ibs.get_image_paths(images)
            payload_image = {'number': wbenc_name}
            # upload the images one by one
            for path in paths:
                try:
                    with open(path, 'rb') as my_file:
                        response = requests.post(target_url_image, data=payload_image, files={'file': my_file})
                        print ('POSTed %s to %s with status %d' % (path, target_url_image, response.status_code))
                except IOError:
                    print('Could not open file at %s' % (path))
                    continue
            # update the wildbook encounter thumbnail image
            response = requests.post(ibs.wbaddr + '/resetThumbnail.jsp?number=' + wbenc_name + '&imageNum=1')


def export_ibeis_to_wildbook2(ibs, eid_list):
    """ LECACY CODE. DEPRICATE? """
    wbaddr = ibs.wbaddr
    wb_encounters = {}

    for encounter_id in eid_list:
        encounter = ibs.get_encounter_enctext(encounter_id)
        # make occurrence id from enctext
        name_roi_mapping = {}
        image_roi_mapping = {}
        names = ibs.get_encounter_nids(encounter_id)
        rois = ibs.get_encounter_rids(encounter_id)
        images = ibs.get_encounter_gids(encounter_id)

        for name_id in names:
            wbenc_name = encounter + str(name_id)
            wb_encounters[name_id] = wbenc_name
            name_roi_mapping[name_id] = []
            image_roi_mapping[name_id] = []

        for roi_id in rois:
            assoc_name_id = ibs.get_roi_nids([roi_id])[0]
            name_roi_mapping[assoc_name_id].append(roi_id)

        print (ibs.get_image_nids(images))
        #for img_id in images:
        #    print (ibs.get_image_nids([img_id]))
        #    assoc_name_id = ibs.get_image_nids([img_id])[0]
        #    image_roi_mapping[assoc_name_id].append(img_id)

        #print (names)
        for name_id in names:
            wbenc_name = wb_encounters[name_id]
            roi_id_list = name_roi_mapping[name_id]
            image_id_list = image_roi_mapping[name_id]
            print (image_id_list)
            wbenc_time_avg = int(sum(ibs.get_image_unixtime(ibs.get_roi_gids(roi_id_list))) / len(roi_id_list))

            lat, lng = map(lambda x: sum(x) / len(x), izip(*ibs.get_image_gps(ibs.get_roi_gids(roi_id_list))))
            print (lat)
            print (lng)
            print (wbenc_time_avg)
            payload_encounter = {'dwcImageURL': wbaddr + '/encounters/encounter.jsp?number=' + wbenc_name,  # NOQA
                                 'state': 'unapproved',
                                 'dateInMilliseconds': wbenc_time_avg,
                                 'approved': True,
                                 'catalogNumber': wbenc_name,
                                 'recordedBy': 'hotspotter',
                                 'individualID': 'Unassigned',
                                 'class': 'org.ecocean.Encounter',
                                 'guid': 'MY_CATALOG:MY_SPECIES:' + wbenc_name,
                                 'decimalLatitude': str(lat),
                                 'decimalLongitude': str(lng)}

            # make occurrence ID part of the request

            #response = requests.post(wbaddr + '/rest/org.ecocean.Encounter', data=json.dumps(payload_encounter))
            #print (response.status_code)
