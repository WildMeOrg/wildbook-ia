## Sanity check
#num_results = len(result_list)
#if num_results != 0 and num_results != num_params:
#    raise lite.Error('num_params=%r != num_results=%r'
#                     % (num_params, num_results))
# Transactions halve query time
# list comprehension cuts time by 10x
#result_list = [_unpacker(results_) for results_ in result_list]


            #if verbose:
                #caller_name = utool.util_dbg.get_caller_name()
                #print('[sql.commit] caller_name=%r' % caller_name)


#printDBG('<ACTUAL COMMIT>')
#printDBG('</ACTUAL COMMIT>')


#if verbose:
#    caller_name = utool.util_dbg.get_caller_name()
#    print('[sql.result] caller_name=%r' % caller_name)



    @adder
    def add_images(ibs, gpath_list):
        """
        Adds a list of image paths to the database.  Returns gids

        Initially we set the image_uri to exactely the given gpath.
        Later we change the uri, but keeping it the same here lets
        us process images asychronously.

        TEST CODE:
            from ibeis.dev.all_imports import *
            gpath_list = grabdata.get_test_gpaths(ndata=7) + ['doesnotexist.jpg']
        """
        print('[ibs] add_images')
        print('[ibs] len(gpath_list) = %d' % len(gpath_list))
        # Processing an image might fail, yeilding a None instead of a tup
        gpath_list = ibsfuncs.assert_and_fix_gpath_slashes(gpath_list)
        # Create param_iter and filter out nones before passing to SQL
        # Eager Evaluation for working with uuids
        params_list  = list(preproc_image.add_images_params_gen(gpath_list))
        isvalid_list = [params is not None for params in params_list]
        # Error reporting
        print('\n'.join(
            [' ! Failed reading gpath=%r' % (gpath,) for (gpath, isvalid)
             in izip(gpath_list, isvalid_list) if not isvalid]))
        # Extract uuids from the params list (requires eager eval)
        uuid_list = [params[0] if isvalid else None for (params, isvalid)
                       in izip(params_list, isvalid_list)]
        # Get the gids to catch any duplicate images
        gid_list_ = ibs.get_image_gids_from_uuid(uuid_list)
        # Get the params for the images that need to be added (valid and dirty)
        isdirty_list = [gid is None and isvalid for (gid, isvalid)
                        in izip(gid_list_, isvalid_list)]
        dirty_params = utool.filter_items(params_list, isdirty_list)
        # Add any unadded images
        print('[ibs] adding %r/%r new images' % len(dirty_params), len(gpath_list))
        if len(dirty_params) > 0:
            tblname = 'images'
            colname_list = ('image_uuid', 'image_uri', 'image_original_name',
                            'image_ext', 'image_width', 'image_height',
                            'image_exif_time_posix', 'image_exif_gps_lat',
                            'image_exif_gps_lon', 'image_notes',)
            params_iter = dirty_params
            # Execute SQL Add
            ibs.db.add(tblname, colname_list, params_iter)
            gid_list = ibs.get_image_gids_from_uuid(uuid_list)
        else:
            gid_list = gid_list_
        return gid_list




        # Insert the new ROIs into the SQL database
        rid_list = ibs.db.executemany(
            operation='''
            INSERT OR REPLACE INTO rois
            (
                roi_rowid,
                roi_uuid,
                image_rowid,
                name_rowid,
                roi_xtl,
                roi_ytl,
                roi_width,
                roi_height,
                roi_theta,
                roi_viewpoint,
                roi_notes
            )
            VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            params_iter=params_iter)



        # Ensure input list is unique
        # name_list = tuple(set(name_list_))
        # HACKY, the adder decorator should specify this
        #nid_list = ibs.get_name_nids(name_list, ensure=False)
        #dirty_names = utool.get_dirty_items(name_list, nid_list)
        #if len(dirty_names) > 0:
        #    print('[ibs] adding %d names' % len(dirty_names))
        #    ibsfuncs.assert_valid_names(name_list)
        #    notes_list = ['' for _ in xrange(len(dirty_names))]
        #    # All names are individuals and so may safely receive the INDIVIDUAL_KEY label
        #    key_rowid_list = [ibs.INDIVIDUAL_KEY for name in name_list]
        #    new_nid_list = ibs.add_labels(key_rowid_list, dirty_names, notes_list)
        #    #print('new_nid_list = %r' % (new_nid_list,))
        #    #get_rowid_from_uuid = partial(ibs.get_name_nids, ensure=False)
        #    #new_nid_list = ibs.db.add_cleanly(LABEL_TABLE, colnames, params_iter, get_rowid_from_uuid)
        #    new_nid_list  # this line silences warnings

        #    # All the names should have been ensured
        #    # this nid list should correspond to the input
        #    nid_list = ibs.get_name_nids(name_list, ensure=False)
        #    #print('nid_list = %r' % (new_nid_list,))
        # # Return nids in input order
        # namenid_dict = {name: nid for name, nid in izip(name_list, nid_list)}
