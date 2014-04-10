"""
UTool
   Your friendly neighborhood utility tools

TODO: INSERT APACHE LICENCE
"""
# disable syntax checking
# flake8: noqa

# We hope to support python3
from __future__ import print_function, division


# Import each utility module
import util_alg
import util_arg
import util_cache
import util_cplat
import util_dbg
import util_dev
import util_decor
import util_distances
import util_dict
import util_inject
import util_iter
import util_list
import util_num
import util_path
import util_print
import util_progress
import util_str
import util_sysreq
import util_time
import util_type

# Import common functions from each utilty module
from .util_alg import (cartesian, almost_eq,)

from .util_arg import (get_arg, get_flag, argv_flag_dec, QUIET, VERBOSE)

from .util_cache import (global_cache_read, global_cache_write)

from .util_cplat import (cmd, view_directory,)

from .util_dbg import (execstr_dict, save_testdata, load_testdata,
                       get_caller_name, import_testdata, embed, quitflag,
                       inIPython, printvar2, all_rrr)

from .util_decor import (ignores_exc_tb, indent_decor, indent_func, lru_cache,
                         accepts_numpy, accepts_scalar_input_vector_output,
                         accepts_scalar_input)

from .util_dev import (printable_mystats, mystats, myprint, get_object_size, )

from .util_distances import (nearest_point,)

from .util_hash import (hashstr_arr, hashstr,)

from .util_inject import (inject, inject_all, inject_print_functions)

from .util_iter import (iflatten, ichunks, interleave,)

from .util_list import (alloc_lists, list_index, npfind, index_of, flatten,
                        list_eq, listfind)

from .util_path import (checkpath, ensuredir, assertpath, truepath, list_images)

from .util_print import (horiz_print, printshape, Indenter,
                         NpPrintOpts, printVERBOSE, printNOTQUIET)

from .util_progress import (progress_func)

from .util_str  import (byte_str2, horiz_string, theta_str)

from .util_time import (tic, toc, Timer)

from .DynStruct import DynStruct

from .Preferences import Pref

from . import Preferences

# "Good" functions that are commonly used should be explicitly listed because
# we are trying to develop quickly, we are being a little hacky
#if not get_flag('--strict', help='raises all exceptions'):
from util_path import *
from util_alg import *
from util_arg import *
from util_cache import *
from util_cplat import *
from util_dbg import *
from util_dev import *
from util_decor import *
from util_dict import *
from util_inject import *
from util_iter import *
from util_list import *
from util_num import *
from util_print import *
from util_progress import *
from util_str import *
from util_time import *
from util_type import *
from util_sysreq import *
from util_distances import *

print, print_, printDBG, rrr, profile = inject(__name__, '[util]')

# Aliases
getflag = util_arg.get_flag
