# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
import logging

import utool as ut

from wbia.control import controller_inject
from wbia.web import appfuncs as appf

(print, rrr, profile) = ut.inject2(__name__)
logger = logging.getLogger('wbia')

register_route = controller_inject.get_wbia_flask_route(__name__)


@register_route('/demo/', methods=['GET'], __route_authenticate__=False)
def demo(*args, **kwargs):
    # Return HTML

    embedded = dict(globals(), **locals())
    return appf.template(None, 'demo', **embedded)
