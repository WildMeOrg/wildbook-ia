from __future__ import print_function, division

import util_alg  # NOQA
import util_arg  # NOQA
import util_cache  # NOQA
import util_cplat  # NOQA
import util_dbg  # NOQA
import util_dev  # NOQA
import util_distances  # NOQA
import util_dict  # NOQA
import util_inject  # NOQA
import util_iter  # NOQA
import util_list  # NOQA
import util_num  # NOQA
import util_path  # NOQA
import util_print  # NOQA
import util_progress  # NOQA
import util_str  # NOQA
import util_sysreq  # NOQA
import util_time  # NOQA
import util_type  # NOQA

from .util_arg import (get_arg, get_flag, argv_flag_dec)  # NOQA

from .util_time import (tic, toc, Timer)  # NOQA

from .util_cplat import (cmd, view_directory,)  # NOQA

from .util_progress import (progress_func)  # NOQA

from .util_alg import (cartesian, almost_eq,)  # NOQA

from .util_hash import (hashstr_arr, hashstr,)  # NOQA

from .util_cache import (lru_cache)  # NOQA

from .util_dbg import (execstr_dict, save_testdata, load_testdata,   # NOQA
                       get_caller_name, import_testdata, embed, quitflag,
                       inIPython, printvar2, all_rrr)

from .util_dev import (printable_mystats, mystats, myprint, get_object_size, )  # NOQA

from .util_distances import (nearest_point,)  # NOQA

from .util_inject import (inject, inject_all, inject_print_functions)  # NOQA

from .util_print import (horiz_print, horiz_string, indent_decor, indent_func, printshape, Indenter, NpPrintOpts)  # NOQA

from .util_iter import (iflatten, ichunks, interleave,)  # NOQA

from .util_path import (checkpath, ensuredir, assertpath, truepath, list_images)  # NOQA

from .util_list import (alloc_lists, list_index, npfind, index_of, flatten,  # NOQA
                        list_eq, listfind)


from .util_str  import (byte_str2, theta_str)  # NOQA

from .DynStruct import DynStruct   # NOQA

#if not get_flag('--strict', help='raises all exceptions'):
from util_path import *  # NOQA
from util_alg import *  # NOQA
from util_arg import *  # NOQA
from util_cache import *  # NOQA
from util_cplat import *  # NOQA
from util_dbg import *  # NOQA
from util_dev import *  # NOQA
from util_dict import *  # NOQA
from util_inject import *  # NOQA
from util_iter import *  # NOQA
from util_list import *  # NOQA
from util_num import *  # NOQA
from util_print import *  # NOQA
from util_progress import *  # NOQA
from util_str import *  # NOQA
from util_time import *  # NOQA
from util_type import *  # NOQA
from util_sysreq import *  # NOQA
from util_distances import *  # NOQA

print, print_, printDBG, rrr, profile = inject(__name__, '[util]')

# Aliases
getflag = util_arg.get_flag
