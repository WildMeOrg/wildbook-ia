#!/usr/bin/env python
"""
This is a hacky script meant to be run interactively
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
from dbgimport2 import *  # NOQA
import utool
from drawtool import draw_func2 as df2
from tests.__testing__ import printTEST  # NOQA
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dev]', DEBUG=True)


if __name__ == '__main__':
    print('\n [DEV] __DEV__\n')
    main_locals = main_api.main()
    ibs = main_locals['ibs']
    back = main_locals['back']
    print('\n [DEV] ENTER EXEC \n')
    exec(df2.present())
