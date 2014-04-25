"""
UTool
   Your friendly neighborhood utility tools

TODO: INSERT APACHE LICENCE
"""
# disable syntax checking
# flake8: noqa
# We hope to support python3
from __future__ import absolute_import, division, print_function
import sys
import textwrap
#import time
#starttime = time.time()
__self_module__ = sys.modules[__name__]

__version__ = '(.878 + .478i)'

__DYNAMIC__ = not '--nodyn' in sys.argv

if __DYNAMIC__:
    __DEVELOPING__ = True
    __PRINT_IMPORTS__ = ('--dump-%s-init' % __name__) in sys.argv

    # Dynamically import listed util libraries and their members.
    # Create reload_subs function.

    # Using __imoprt__ like this is typically not considered good style However,
    # it is better than import * and this will generate the good file text that
    # can be used when the module is "frozen"

    UTOOLS_LIST = [
        ('util_alg',       ['cartesian', 'almost_eq',]),
        ('util_arg',       ['get_arg', 'get_flag', 'argv_flag_dec', 'QUIET',
                            'VERBOSE']),
        ('util_cache',     ['global_cache_read', 'global_cache_write']),
        ('util_cplat',     ['cmd', 'view_directory',]),
        ('util_csv',       None),
        ('util_dbg',       ['execstr_dict', 'save_testdata', 'load_testdata',
                            'get_caller_name', 'import_testdata', 'embed',
                            'quitflag', 'inIPython', 'printvar2', 'all_rrr']),
        ('util_dev',       ['printable_mystats', 'mystats', 'myprint',
                            'get_object_size']),
        ('util_decor',     ['ignores_exc_tb', 'indent_decor', 'indent_func',
                            'lru_cache', 'accepts_numpy',
                            'accepts_scalar_input_vector_output',
                            'accepts_scalar_input']),
        ('util_distances', ['nearest_point',]),
        ('util_dict',      None),
        ('util_hash',      ['hashstr_arr', 'hashstr',]),
        ('util_inject',    ['inject', 'inject_all', 'inject_print_functions']),
        ('util_iter',      ['iflatten', 'ichunks', 'interleave',]),
        ('util_list',      ['alloc_lists', 'list_index', 'npfind', 'index_of',
                            'flatten']),
        ('util_num',       None),
        ('util_path',      ['checkpath', 'ensuredir', 'assertpath', 'truepath',
                            'list_images']),
        ('util_print',     ['horiz_print', 'printshape', 'Indenter']),
        ('util_progress',  ['progress_func']),
        ('util_str',       ['byte_str2', 'horiz_string', 'theta_str']),
        ('util_sysreq',    None),
        ('util_regex',     ['regex_search']),
        ('util_time',      ['tic', 'toc', 'Timer']),
        ('util_type',      None),
        ('DynStruct',      ['DynStruct']),
        ('Preferences',    ['Pref']),
        ]

    IMPORTS      = [name for name, fromlist in UTOOLS_LIST]
    FROM_IMPORTS = [(name, fromlist) for name, fromlist in UTOOLS_LIST
                    if fromlist is not None and len(fromlist) > 0]

    # Module Imports
    for name in IMPORTS:
        tmp = __import__(name, globals(), locals(), fromlist=[], level=-1)


    # Injection and Reload String Defs
    module_rrr_strings = [name + '.rrr()' for name in IMPORTS]
    reload_subs_head = textwrap.dedent('''
    def reload_subs():
        """Reloads %s and submodules """
        rrr()
    ''') % __name__
    reload_subs_body = '\n    ' + ('\n    '.join(module_rrr_strings))
    reload_subs_footer = textwrap.dedent('''
    rrrr = reload_subs
    ''')
    reload_subs_func_str = reload_subs_head  + reload_subs_body + reload_subs_footer

    utool_inject_str = 'print, print_, printDBG, rrr, profile = util_inject.inject(__name__, \'[%s]\')' % __name__

    exec(utool_inject_str)
    exec(reload_subs_func_str)

    # From imports
    if not __DEVELOPING__:
        for name, fromlist in FROM_IMPORTS:
            tmp = __import__(name, globals(), locals(), fromlist=fromlist, level=-1)
            for member in fromlist:
                setattr(__self_module__, member, getattr(tmp, member))
    # Effectively import * statements
    else:
        FROM_IMPORTS = []
        for name, fromlist in UTOOLS_LIST:
            tmp = sys.modules[__name__ + '.' + name]
            varset = set(vars())
            fromset = set(fromlist) if fromlist is not None else set()
            def valid_member(member):
                """ Returns if a member of a module is able to be added as a
                from import """
                is_private = member.startswith('__')
                is_conflit = member in varset
                is_module  = member in sys.modules  # Isn't fool proof (next step is)
                is_forced  = member in fromset
                return  (is_forced or not (is_private or is_conflit or is_module))
            fromlist_ = [member for member in dir(tmp) if valid_member(member)]
            valid_fromlist_ = []
            for member in fromlist_:
                member_val = getattr(tmp, member)
                try:
                    # Disallow importing modules
                    if getattr(member_val, '__name__') in sys.modules:
                        print(str(member_val))
                        continue
                except AttributeError:
                    pass
                valid_fromlist_.append(member)
                setattr(__self_module__, member, member_val)
            FROM_IMPORTS.append((name, valid_fromlist_))

    # Print what this module should look like
    if __PRINT_IMPORTS__:
        print('')
        pack_into = util_str.pack_into
        import_str = '\n'.join(['from . import %s' % (name,) for name in IMPORTS])
        def _fromimport_str(name, fromlist):
            from_module_str = 'from .%s import (' % name
            newline_prefix = (' ' * len(from_module_str))
            rawstr = from_module_str + ', '.join(fromlist) + ',)'
            packstr = pack_into(rawstr, textwidth=80, newline_prefix=newline_prefix)
            return packstr
        from_str   = '\n'.join([_fromimport_str(name, fromlist) for (name, fromlist) in FROM_IMPORTS])
        print(util_str.indent(import_str))
        print(util_str.indent(from_str))
        print(util_str.indent(utool_inject_str))
        print(util_str.indent(reload_subs_func_str))
        print('')
else:
    from . import util_alg
    from . import util_arg
    from . import util_cache
    from . import util_cplat
    from . import util_csv
    from . import util_dbg
    from . import util_dev
    from . import util_decor
    from . import util_distances
    from . import util_dict
    from . import util_hash
    from . import util_inject
    from . import util_iter
    from . import util_list
    from . import util_num
    from . import util_path
    from . import util_print
    from . import util_progress
    from . import util_str
    from . import util_sysreq
    from . import util_regex
    from . import util_time
    from . import util_type
    from . import DynStruct
    from . import Preferences
    from .util_alg import (almost_eq, cartesian, choose, find_std_inliers,
                           norm_zero_one, normalize, xywh_to_tlbr,)
    from .util_arg import (ArgumentParser2, Indenter, QUIET, VERBOSE, argv_flag,
                           argv_flag_dec, argv_flag_dec_true, get_arg, get_flag,
                           inject, make_argparse2, switch_sanataize, try_cast,)
    from .util_cache import (close_global_shelf, delete_global_cache,
                             get_global_cache_dir, get_global_shelf,
                             get_global_shelf_fpath, global_cache_dump,
                             global_cache_read, global_cache_write, join, normpath,
                             read_from, write_to,)
    from .util_cplat import (COMPUTER_NAME, DARWIN, LINUX, WIN32, cmd, exists,
                             expanduser, get_computer_name, get_flops,
                             get_resource_dir, getroot, startfile, view_directory,)
    from .util_csv import (is_float, is_int, is_list, is_str, make_csv_table,
                           numpy_to_csv,)
    from .util_dbg import (IPYTHON_EMBED_STR, all_rrr, debug_exception,
                           debug_hstack, debug_list, debug_npstack, debug_vstack,
                           dict_dbgstr, embed, execstr_attr_list, execstr_dict,
                           execstr_embed, execstr_func, execstr_parent_locals,
                           execstr_src, explore_module, explore_stack, formatex,
                           get_caller_locals, get_caller_name, get_caller_prefix,
                           get_func_name, get_parent_frame, get_parent_globals,
                           get_parent_locals, get_reprs, get_stack_frame, get_type,
                           haveIPython, horiz_string, import_testdata, inIPython,
                           indent, ipython_execstr, ipython_execstr2, is_listlike,
                           keys_dbgstr, len_dbgstr, list_dbgstr, list_eq,
                           load_testdata, module_functions, pack_into, print_frame,
                           print_varlen, printex, printvar, printvar2,
                           public_attributes, qflag, quit, quitflag, save_testdata,
                           search_stack_for_localvar, search_stack_for_var,
                           truncate_str,)
    from .util_dev import (DEPRICATED, OrderedDict, byte_str2,
                           disable_garbage_collection, enable_garbage_collection,
                           garbage_collect, get_object_base, get_object_size, info,
                           listinfo, memory_profile, myprint, mystats, npinfo,
                           numpy_list_num_bits, print_object_size, printableVal,
                           printable_mystats, runprofile, stats_str,)
    from .util_decor import (DISABLE_WRAPPERS, IGNORE_EXC_TB, UNIQUE_NUMPY,
                             accepts_numpy, accepts_scalar_input,
                             accepts_scalar_input_vector_output, common_wrapper,
                             composed, ignores_exc_tb, imap, indent_decor,
                             indent_func, isiterable, islice, lru_cache,
                             printVERBOSE, wraps,)
    from .util_distances import (L1, L2, L2_sqrd, compute_distances, emd,
                                 hist_isect, izip, nearest_point,)
    from .util_dict import (all_dict_combinations, build_conflict_dict, defaultdict,
                            dict_union, dict_union2, iprod, items_sorted_by_value,
                            keys_sorted_by_value,)
    from .util_hash import (ALPHABET, BIGBASE, HASH_LEN, augment_uuid,
                            convert_hexstr_to_base57, get_zero_uuid, hash_to_uuid,
                            hashstr, hashstr_arr, hashstr_md5, hashstr_sha1,
                            image_uuid,)
    from .util_inject import (ARGV_DEBUG_FLAGS, _add_injected_module, _get_module,
                              _inject_funcs, argv, get_injected_modules, inject,
                              inject_all, inject_colored_exceptions,
                              inject_print_functions, inject_profile_function,
                              inject_reload_function,)
    from .util_iter import (chain, cycle, ensure_iterable, ichunks, iflatten,
                            iflatten_scalars, interleave,)
    from .util_list import (alloc_lists, alloc_nones, assert_all_not_None,
                            deterministic_shuffle, ensure_list_size, filter_Nones,
                            filter_items, flatten, flatten_items, get_dirty_items,
                            inbounds, index_of, intersect2d, intersect2d_numpy,
                            intersect_ordered, list_index, list_replace, listfind,
                            npfind, random_indexes, safe_listget, safe_slice,
                            scalarflatten, spaced_indexes, spaced_items,
                            tiled_range, tuplize, unique_keep_order,
                            unique_keep_order2,)
    from .util_num import (commas, fewest_digits_float_str, float_to_decimal,
                           format_, int_comma_str, num2_sigfig, num_fmt,
                           order_of_magnitude_ceil, sigfig_str,)
    from .util_path import (BadZipfile, IMG_EXTENSIONS, assertpath, checkpath,
                            copy_all, copy_list, copy_task, delete, dirname,
                            dirsplit, download_url, ensuredir, ensurepath, ext,
                            file_bytes, file_megabytes, fnames_to_fpaths,
                            fpaths_to_fnames, get_module_dir, glob, isdir, isfile,
                            islink, ismount, list_images, longest_existing_path,
                            matches_image, move_list, num_images_in_dir,
                            path_ndir_split, progress_func, realpath, relpath,
                            remove_dirs, remove_file, remove_files_in_dir, split,
                            symlink, truepath, unixpath, unzip_file, win_shortcut,)
    from .util_print import (Indenter, NO_INDENT, NpPrintOpts, filesize_str,
                             horiz_print, printNOTQUIET, print_filesize,
                             printshape,)
    from .util_progress import (VALID_PROGRESS_TYPES, prog_func, progress_func,
                                progress_str, simple_progres_func,)
    from .util_str import (GLOBAL_TYPE_ALIASES, byte_str, byte_str2,
                           dict_aliased_repr, dict_itemstr_list, dict_str,
                           extend_global_aliases, file_megabytes_str,
                           full_numpy_repr, func_str, get_unix_timedelta,
                           get_unix_timedelta_str, horiz_string, indent_list,
                           indentjoin, joins, list_aliased_repr, listinfo_str,
                           newlined_list, remove_chars, str2, str_between,
                           theta_str, unindent, var_aliased_repr,)
    from .util_sysreq import (DEBUG, ensure_in_pythonpath, locate_path,)
    from .util_regex import (RE_FLAGS, RE_KWARGS, get_match_text, named_field,
                             named_field_regex, regex_parse, regex_search,
                             regex_split,)
    from .util_time import (Timer, exiftime_to_unixtime, get_timestamp, tic, toc,)
    from .util_type import (VALID_FLOAT_TYPES, VALID_INT_TYPES, assert_int, is_bool,
                            is_dict, is_type,)
    from .DynStruct import (AbstractPrintable, DynStruct,)
    from .Preferences import (EditPrefWidget, Pref, PrefChoice, PrefInternal,
                              PrefNode, PrefTree, QAbstractItemModel, QModelIndex,
                              QObject, QPreferenceModel, QString, QVariant, QWidget,
                              Qt, Ui_editPrefSkel, _encoding, _fromUtf8, _translate,
                              pyqtSlot, report_thread_error,)
    print, print_, printDBG, rrr, profile = util_inject.inject(__name__, '[utool]')

    def reload_subs():
        """Reloads utool and submodules """
        rrr()

        util_alg.rrr()
        util_arg.rrr()
        util_cache.rrr()
        util_cplat.rrr()
        util_csv.rrr()
        util_dbg.rrr()
        util_dev.rrr()
        util_decor.rrr()
        util_distances.rrr()
        util_dict.rrr()
        util_hash.rrr()
        util_inject.rrr()
        util_iter.rrr()
        util_list.rrr()
        util_num.rrr()
        util_path.rrr()
        util_print.rrr()
        util_progress.rrr()
        util_str.rrr()
        util_sysreq.rrr()
        util_regex.rrr()
        util_time.rrr()
        util_type.rrr()
        DynStruct.rrr()
        Preferences.rrr()
    rrrr = reload_subs

# Aliases
getflag = util_arg.get_flag
#print('imported utool in %r seconds' % (time.time() - starttime,))
