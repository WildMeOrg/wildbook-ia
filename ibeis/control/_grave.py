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



        #fid_list = ibs.db.get_executeone_where(FEATURE_TABLE, ('feature_rowid',), 'config_rowid=?', (feat_config_rowid,))
        #fid_list = sorted(fid_list)
        #params = (chip_config_rowid,)
        #cid_list = ibs.db.get_executeone_where(CHIP_TABLE, ('chip_rowid',), 'config_rowid=?', params)
        #cid_list = sorted(cid_list)
        #return sorted(cid_list)

        #all_nids = ibs.db.get_executeone_where(LABEL_TABLE, ('label_rowid',), where_clause, params)
        #all_nids = sorted(all_nids)



        #all_gids = sorted(ibs.db.get_executeone(IMAGE_TABLE, ('image_rowid',)))
        #all_rids = sorted(ibs.db.get_executeone(ANNOTATION_TABLE, ('annot_rowid',)))
        #all_eids = sorted(ibs.db.get_executeone(ENCOUNTER_TABLE,
        #                                        ('encounter_rowid',)))
        #all_cids = sorted(ibs.db.get_executeone(CHIP_TABLE, ('chip_rowid',)))
        #all_fids = sorted(ibs.db.get_executeone(FEATURE_TABLE, ('feature_rowid',)))





    #@default_decorator
    #def get_executeone(db, tblname, colnames, **kwargs):
    #    """ DEPRICATE """
    #    if isinstance(colnames, (str, unicode)):
    #        colnames = (colnames,)
    #    fmtdict = {
    #        'tblname'         : tblname,
    #        'colnames_str'    : ', '.join(colnames),
    #    }
    #    operation_fmt = '''
    #        SELECT {colnames_str}
    #        FROM {tblname}
    #        '''
    #    val_list = db._executeone_operation_fmt(operation_fmt, fmtdict, **kwargs)
    #    return val_list

    #@default_decorator
    #def get_executeone_where(db, tblname, colnames, where_clause, params, **kwargs):
    #    """ DEPRICATE """
    #    if isinstance(colnames, (str, unicode)):
    #        colnames = (colnames,)
    #    fmtdict = {
    #        'tblname'         : tblname,
    #        'colnames_str'    : ', '.join(colnames),
    #        'where_clause'    : where_clause
    #    }
    #    operation_fmt = '''
    #        SELECT {colnames_str}
    #        FROM {tblname}
    #        WHERE {where_clause}
    #        '''
    #    val_list = db._executeone_operation_fmt(operation_fmt, fmtdict, params=params, **kwargs)
    #    return val_list



        #OFF printDBG('------------------------')
        #OFF printDBG('set_(table=%r, prop_key=%r)' % (table, prop_key))
        #OFF printDBG('set_(rowid_list=%r, val_list=%r)' % (rowid_list, val_list))
        #from operator import xor
        #assert not xor(utool.isiterable(rowid_list),
        #               utool.isiterable(val_list)), 'invalid mixing of iterable and scalar inputs'

        #if not utool.isiterable(rowid_list) and not utool.isiterable(val_list):
        #    rowid_list = (rowid_list,)
        #    val_list = (val_list,)



    @default_decorator
    def commit(db, qstat_flag_list=[],  verbose=VERBOSE, errmsg=None):
        """ Commits staged changes to the database and saves the binary
            representation of the database to disk.  All staged changes can be
            commited one at a time or after a batch - which allows for batch
            error handling without comprimising the integrity of the database.
        """
        try:
            if not all(qstat_flag_list):  # DEPRICATE
                raise lite.DatabaseError(errmsg)  # DEPRICATE
            else:
                db.connection.commit()
                if AUTODUMP:
                    db.dump(auto_commit=False)
        except lite.Error as ex2:
            print('\n<!!! ERROR>')  # DEPRICATE
            utool.printex(ex2, '[!sql] Caught ex2=')
            caller_name = utool.util_dbg.get_caller_name()  # DEPRICATE
            print('[!sql] caller_name=%r' % caller_name)  # DEPRICATE
            print('</!!! ERROR>\n')  # DEPRICATE
            raise lite.DatabaseError('%s --- %s' % (errmsg, ex2))


        #configid_list = ibs.get_config_rowid_from_suffix(cfgsuffix_list, ensure=False)
        ##print('configid_list %r' % (configid_list,))
        ##print('cfgsuffix_list %r' % (cfgsuffix_list,))
        #try:
        #    # [Jon]FIXME: This check is really weird? Why is it here?
        #    isdirty_list = [
        #        rowid is None or (isinstance(rowid, list) and len(rowid) == 0)
        #        for rowid in configid_list]
        #    if any(isdirty_list):
        #        params_iter = ((suffix,) for suffix in cfgsuffix_list)
        #        colnames = ('config_suffix',)
        #        get_rowid_from_uuid = partial(ibs.get_config_rowid_from_suffix, ensure=False)
        #        configid_list = ibs.db.add_cleanly(CONFIG_TABLE, colnames, params_iter, get_rowid_from_uuid)
        #except Exception as ex:
        #    utool.printex(ex)
        #    print('FATAL ERROR')
        #    utool.sys.exit(1)



        #label_uuid_list = [uuid.uuid4() for _ in xrange(len(value_list))]
        # FIXME: This should actually be a random uuid, but (key, vals) should be
        # enforced as unique as well
        # NOTE:
        # A Case for Name UUIDs to not be deterministic
        # Premise: names should just be uuids
        # 0) Name text is constrained to be unique
        # 1) UUIDs are not needed for JOINS.
        #    The next name are generated each time you merge a name
        # A Case against deterministic UUIDS:
        # 0) Changing the nickname would mean you have to change the UUID
        #label_uuid_list = [utool.deterministic_uuid(repr((key, value))) for key, value in
        #                   izip(key_list, value_list)]
