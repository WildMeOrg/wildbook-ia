# flake8: noqa
from __future__ import absolute_import, division, print_function
__version__ = '1.0.0.dev1'

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

from . import histogram as htool
from . import linalg as ltool
from . import image as gtool
from . import exif as exiftool
from . import keypoint as ktool
from . import ellipse as etool
from . import patch as ptool
from . import chip as ctool
from . import spatial_verification as svtool
from . import trig
from . import math as mtool

#try:
#    from . import linalg_cython
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

cyth.py vtool/linalg_cython.pyx

C:\Python27\Scripts\cython.exe C:\Users\joncrall\code\vtool\vtool\linalg_cython.pyx

#C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\linalg_cython.c -o vtool\linalg_cython.o
#C:\MinGW\bin\gcc.exe -shared -s build\temp.win32-2.7\Release\vtool\linalg_cython.o build\temp.win32-2.7\Release\vtool\linalg_cython.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o  build\lib.win32-2.7\linalg_cython.pyd

C:\MinGW\bin\gcc.exe -shared -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -static-libgcc -static-libstdc++ -o vtool\linalg_cython.pyd
python -c "import vtool"


C:\MinGW\bin\gcc.exe -mdll -O -Wall -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -c vtool\linalg_cython.c -o vtool\linalg_cython.o
C:\MinGW\bin\gcc.exe -shared -s vtool\linalg_cython.o vtool\linalg_cython.def -LC:\Python27\libs -LC:\Python27\PCbuild -lpython27 -lmsvcr90 -o vtool\linalg_cython.pyd


C:\MinGW\bin\gcc.exe -w -Wall -m32 -lpython27 -IC:\Python27\Lib\site-packages\numpy\core\include -IC:\Python27\include -IC:\Python27\PC -IC:\Python27\Lib\site-packages\numpy\core\include -LC:\Python27\libs -o vtool\linalg_cython.pyd -c vtool\linalg_cython.c

python -c "import vtool"
"""
