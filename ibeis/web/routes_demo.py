# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from flask import request, redirect, url_for, current_app
from ibeis.control import controller_inject
from ibeis.web import appfuncs as appf
from ibeis import constants as const
import utool as ut
import numpy as np
import uuid
import six


register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_route('/demo/', methods=['GET'], __route_authenticate__=False)
def demo(*args, **kwargs):
    # Return HTML

    embedded = dict(globals(), **locals())
    return appf.template(None, 'demo', **embedded)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.web.app
        python -m ibeis.web.app --allexamples
        python -m ibeis.web.app --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
