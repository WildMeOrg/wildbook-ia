# flake8: noqa
from __future__ import absolute_import, division, print_function
__version__ = '1.0.0.dev1'


# TODO utoolify this
IMPORT_TUPLES = [
    ('image', None),
    ('histogram', None),
    ('image', None),
    ('exif', None),
    ('keypoint', None),
    ('patch', None),
    ('chip', None),
    ('spatial_verification', None),
    ('trig', None),
    ('math', None),
    ('geometry', None),
    ('clustering2', None),
]

from . import histogram
from . import linalg
from . import image
from . import exif
from . import keypoint
from . import ellipse
from . import patch
from . import chip
from . import spatial_verification
from . import trig
from . import math
from . import geometry
from . import clustering
from . import nearest_neighbors
from . import clustering2

from . import histogram as htool
from . import linalg as ltool
from . import image as gtool
from . import exif as exiftool
from . import keypoint as ktool
from . import ellipse as etool
from . import patch as ptool
from . import chip as ctool
from . import spatial_verification as svtool
from . import clustering2 as clustertool
from . import trig
from . import math as mtool

#try:
#    from . import _linalg_cyth
#    #print('[vtool] cython is on')
#except ImportError as ex:
#    #import utool
#    #utool.printex(ex, iswarning=True)
#    #print('[vtool] cython is off')
#    raise


r"""

ls vtool

rm -rf build ; rm vtool/*.pyd ; rm vtool/*.c

python setup.py build_ext --inplace

cyth.py vtool/_linalg_cyth.pyx

C:\Python27\Scripts\cython.exe C:\Users\joncrall\code\vtool\vtool\_linalg_cyth.pyx

#C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\_linalg_cyth.c -o vtool\_linalg_cyth.o
#C:\MinGW\bin\gcc.exe -shared -s build\temp.win32-2.7\Release\vtool\_linalg_cyth.o build\temp.win32-2.7\Release\vtool\_linalg_cyth.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o  build\lib.win32-2.7\_linalg_cyth.pyd

C:\MinGW\bin\gcc.exe -shared -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -static-libgcc -static-libstdc++ -c _linalg_cyth.c -o _linalg_cyth.pyd
python -c "import vtool"


C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\_linalg_cyth.c -o _linalg_cyth.o
C:\MinGW\bin\gcc.exe -shared -s _linalg_cyth.o _linalg_cyth.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o _linalg_cyth.pyd


C:\MinGW\bin\gcc.exe -w -Wall -m32 -lpython27 -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -IC:\Python27\Lib\site-packages\numpy\core\include -LC:\Python27\libs -o _linalg_cyth.pyd -c _linalg_cyth.c

python -c "import vtool"
"""

import sys
__DYNAMIC__ = not '--nodyn' in sys.argv

#__DYNAMIC__ = '--dyn' in sys.argv
"""
python -c "import vtool" --dump-vtool-init
python -c "import vtool" --update-vtool-init
"""

DOELSE = False
if __DYNAMIC__:
    # TODO: import all utool external prereqs. Then the imports will not import
    # anything that has already in a toplevel namespace
    # COMMENTED OUT FOR FROZEN __INIT__
    # Dynamically import listed util libraries and their members.
    from utool._internal import util_importer
    # FIXME: this might actually work with rrrr, but things arent being
    # reimported because they are already in the modules list
    import_execstr = util_importer.dynamic_import(__name__, IMPORT_TUPLES)
    exec(import_execstr)
    DOELSE = False
else:
    # Do the nonexec import (can force it to happen no matter what if alwyas set
    # to True)
    DOELSE = True

if DOELSE:
    pass
    # <AUTOGEN_INIT>
    # </AUTOGEN_INIT>
