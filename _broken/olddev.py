        mod_fpath = ut.truepath('~/code/ibeis/ibeis/expt/results_all.py')
        module = ut.import_module_from_fpath(mod_fpath)
        user_profile = ut.ensure_user_profile()
        doctestables = list(ut.iter_module_doctestable(module, include_builtin=False))
        grepkw = {}
        grepkw['exclude_dirs'] = user_profile.project_exclude_dirs
        grepkw['dpath_list'] = user_profile.project_dpaths
        grepkw['verbose'] = True

        usage_map = {}
        for funcname, func in doctestables:
            print('Searching for funcname = %r' % (funcname,))
            found_fpath_list, found_lines_list, found_lxs_list = ut.grep([funcname], **grepkw)
            used_in = (found_fpath_list, found_lines_list, found_lxs_list)
            usage_map[funcname] = used_in

        external_usage_map = {}
        for funcname, used_in in usage_map.items():
            (found_fpath_list, found_lines_list, found_lxs_list) = used_in
            isexternal_flag = [normpath(fpath) != normpath(mod_fpath) for fpath in found_fpath_list]
            ext_used_in = (ut.compress(found_fpath_list, isexternal_flag),
                           ut.compress(found_lines_list, isexternal_flag),
                           ut.compress(found_lxs_list, isexternal_flag))
            external_usage_map[funcname] = ext_used_in

        for funcname, used_in in external_usage_map.items():
            (found_fpath_list, found_lines_list, found_lxs_list) = used_in

        print('Calling modules: \n' +
              ut.repr2(ut.unique_keep_order(ut.flatten([used_in[0] for used_in in  external_usage_map.values()])), nl=True))

