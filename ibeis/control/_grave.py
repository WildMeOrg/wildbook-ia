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
                roi_uid,
                roi_uuid,
                image_uid,
                name_uid,
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
