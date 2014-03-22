from __future__ import print_function, division

import util_alg  # NOQA
import util_arg  # NOQA
import util_cache  # NOQA
import util_cplat  # NOQA
import util_dbg  # NOQA
import util_dev  # NOQA
import util_dict  # NOQA
import util_inject  # NOQA
import util_iter  # NOQA
import util_list  # NOQA
import util_num  # NOQA
import util_path  # NOQA
import util_print  # NOQA
import util_progress  # NOQA
import util_str  # NOQA
import util_time  # NOQA
import util_type  # NOQA

from .util_time import (tic, toc, get_timestamp, Timer)  # NOQA

from .util_cplat import (get_computer_name, _cmd, startfile, view_directory)  # NOQA

from .util_progress import (simple_progres_func, prog_func, progress_func,  # NOQA
                            progress_str)

from .util_alg import (normalize, norm_zero_one, find_std_inliers, choose,  # NOQA
                       cartesian, almost_eq)

from .util_hash import (hashstr_arr, hashstr, hex2_base57, hashstr_md5)  # NOQA

from .util_cache import (lru_cache)  # NOQA

from .util_dev import (DEPRICATED, stats_str, printable_mystats, mystats,  # NOQA
                       myprint, info, npinfo, listinfo, numpy_list_num_bits,
                       runprofile, memory_profile, disable_garbage_collection,
                       enable_garbage_collection, garbage_collect,
                       get_object_size, print_object_size)

from .util_dict import (all_dict_combinations, dict_union2, dict_union)  # NOQA

from .util_arg import (get_arg, get_flag, argv_flag)  # NOQA

from .util_inject import (inject_print_functions,  # NOQA
                          inject_reload_function, inject_profile_function,
                          inject)

from .util_print import (horiz_print, horiz_string, str2, rectify_wrapped_func,  # NOQA
                         indent_decor, printshape, Indenter, NpPrintOpts)

from .util_str import (remove_chars, indent, truncate_str, pack_into, joins,  # NOQA
                       indent_list)

from .util_type import (try_cast, assert_int, is_type, is_int, is_float, is_str,  # NOQA
                        is_bool, is_dict, is_list)

from .util_iter import (ensure_iterable, iflatten, ichunks, interleave,  # NOQA
                        class_iter_input)

from .util_num import (order_of_magnitude_ceil, format, float_to_decimal,  # NOQA
                       sigfig_str, num2_sigfig, num_fmt, int_comma_str,
                       fewest_digits_float_str, commas)

from .util_path import (path_ndir_split, remove_file, remove_dirs,  # NOQA
                        remove_files_in_dir, delete, longest_existing_path,
                        checkpath, ensurepath, ensuredir, assertpath, copy_task,
                        copy, copy_all, copy_list, move_list, win_shortcut,
                        symlink, file_bytes, byte_str2, byte_str,
                        file_megabytes, file_megabytes_str, glob)

from .util_dbg import (ipython_execstr, execstr_parent_locals,  # NOQA
                       execstr_attr_list, execstr_dict, execstr_func,
                       execstr_src, save_testdata, load_testdata,
                       import_testdata, embed, quitflag, qflag, quit, inIPython,
                       haveIPython, print_frame, search_stack_for_localvar,
                       get_stack_frame, get_parent_locals, get_parent_globals,
                       get_caller_locals, module_functions, public_attributes,
                       explore_stack, explore_module, debug_npstack, debug_list,
                       debug_hstack, debug_vstack, get_caller_name,
                       debug_exception, printvar2, printvar)

from .util_list import (alloc_lists, ensure_list_size, tiled_range,  # NOQA
                        random_indexes, safe_listget, list_index, listfind,
                        npfind, index_of, list_replace, flatten, is_listlike,
                        list_eq, inbounds, intersect_ordered, intersect2d_numpy,
                        intersect2d, unique_keep_order)

from .DynStruct import DynStruct   # NOQA


print, print_, printDBG, rrr, profile = inject(__name__, '[util]')
