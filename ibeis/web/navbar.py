from flask import request


class NavbarClass(object):
    def __init__(nav):
        nav.item_list = [
            ('',     'Home'),
            ('view', 'View'),
            ('turk', 'Turk'),
            ('api',  'API'),
        ]

    def __iter__(nav):
        _link = request.path.strip('/').split('/')
        for link, nice in nav.item_list:
            yield link == _link[0], link, nice
