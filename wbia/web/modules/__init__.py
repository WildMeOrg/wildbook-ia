# -*- coding: utf-8 -*-
"""
Modules
=======

Modules enable logical resource separation.

You may control enabled modules by modifying ``ENABLED_MODULES`` config
variable.
"""
import logging


def init_app(app, **kwargs):
    from importlib import import_module

    module_names = [
        'auth',
        'users',
        'detect',
        'swagger_ui',
    ]
    for module_name in module_names:
        logging.debug('Init module {!r}'.format(module_name))
        import_module('.%s' % module_name, package=__name__).init_app(app, **kwargs)
