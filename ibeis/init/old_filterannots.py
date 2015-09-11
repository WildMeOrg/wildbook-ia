@profile
def expand_to_default_aids(ibs, aidcfg, prefix='', verbose=VERB_TESTDATA):
    default_aids = aidcfg['default_aids']

    if verbose:
        print(' * [INCLUDE %sAIDS]' % (prefix.upper()))
        #print(' * PARSING %saidcfg = %s' % (prefix, ut.dict_str(aidcfg, align=True),))
        print(' * default_%saids = %s' % (prefix, ut.obj_str(default_aids,
                                                             truncate=True,
                                                             nl=False)))

    if isinstance(default_aids, six.string_types):
        #if verbose:
        #    print(' * interpreting default %saids.' % (prefix,))
        # Abstract default aids
        if default_aids in ['all']:
            default_aids = ibs.get_valid_aids()
        elif default_aids in ['allgt', 'gt']:
            default_aids = ibs.get_valid_aids(hasgt=True)
        elif default_aids in ['largetime24']:
            # HACK for large timedelta base sample pool
            default_aids = ibs.get_valid_aids(
                is_known=True,
                has_timestamp=True,
                min_timedelta=24 * 60 * 60,
            )
        elif default_aids in ['largetime12']:
            # HACK for large timedelta base sample pool
            default_aids = ibs.get_valid_aids(
                is_known=True,
                has_timestamp=True,
                min_timedelta=12 * 60 * 60,
            )
        elif default_aids in ['other']:
            # Hack, should actually become the standard.
            # Use this function to build the default aids
            default_aids = ibs.get_valid_aids(
                is_known=aidcfg['is_known'],
                min_timedelta=aidcfg['min_timedelta'],
                has_timestamp=aidcfg['require_timestamp']
            )
        #elif default_aids in ['reference_gt']:
        #    pass
        else:
            raise NotImplementedError('Unknown default string = %r' % (default_aids,))
    else:
        if verbose:
            print(' ... default %saids specified.' % (prefix,))

    #if aidcfg['include_aids'] is not None:
    #    raise NotImplementedError('Implement include_aids')

    available_aids = default_aids

    if len(available_aids) == 0:
        print(' WARNING no %s annotations available' % (prefix,))

    #if aidcfg['exclude_aids'] is not None:
    #    if verbose:
    #        print(' * Excluding %d custom aids' % (len(aidcfg['exclude_aids'])))
    #    available_aids = ut.setdiff_ordered(available_aids, aidcfg['exclude_aids'])
    available_aids = sorted(available_aids)

    if verbose:
        print(' * HAHID: ' + ibs.get_annot_hashid_semantic_uuid(
            available_aids, prefix=prefix.upper()))
        print(' * DEFAULT: len(available_%saids)=%r\n' % (prefix, len(available_aids)))
    return available_aids


