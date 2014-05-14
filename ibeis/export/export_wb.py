#!/usr/bin/env python
"""
Converts an IBEIS database to a wildbook db
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from os.path import join, relpath
#import ibeis
import utool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[export_wb]')


def export_ibeis_to_wildbook(ibs, eid_list):
    import json
    import requests
    wbaddr = ibs.wbaddr
    wb_encounters = {}

    for encounter_id in eid_list:
        encounter = ibs.get_encounter_enctext(encounter_id)
        # make occurrence id from enctext
        name_roi_mapping = {}
        names = ibs.get_encounter_nids(encounter_id)
        rois = ibs.get_encounter_rids(encounter_id)
        for name_id in names:
            wbenc_name = encounter + name_id
            wb_encounters[name_id] = wbenc_name
            name_roi_mapping[name_id] = []

        for roi_id in rois:
            assoc_name_id = ibs.get_roi_name(roi_id)
            name_roi_mapping[assoc_name_id].append(roi_id)
        print (names)
        for name_id in names:
            wbenc_name = wb_encounters[name_id]
            roi_id_list = name_roi_mapping[name_id]
            wbenc_time_avg = int(sum(ibs.get_image_unixtime(ibs.get_roi_gids(roi_id_list)))/len(roi_id_list)) 
            
            lat, lng = map(lambda x: sum(x)/len(x),zip(*ibs.get_image_gps(ibs.get_roi_gids(roi_id_list))))

            payload_encounter = {"dwcImageURL":wbaddr + "/encounters/encounter.jsp?number=" + wbenc_name,
                                 "state":"unapproved",
                                 "dateInMilliseconds":wbenc_time_avg,
                                 "approved":True,
                                 "catalogNumber":wbenc_name,
                                 "recordedBy":"hotspotter",
                                 "individualID":"Unassigned",
                                 "class":"org.ecocean.Encounter",
                                 "guid":"MY_CATALOG:MY_SPECIES:"+ wbenc_name,
                                 "decimalLatitude":str(lat),
                                 "decimalLongitude":str(lng)}

            # make occurrence ID part of the request

            response = requests.post(wbaddr + "/rest/org.ecocean.Encounter", data=json.dumps(payload_encounter))
            print (response.text)
            