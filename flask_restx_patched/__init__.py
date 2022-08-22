# -*- coding: utf-8 -*-
from flask_restx import *  # NOQA

from .api import Api  # NOQA
from .model import DefaultHTTPErrorSchema, Schema  # NOQA

try:
    from .model import ModelSchema  # NOQA
except ImportError:  # pragma: no cover
    pass
from .namespace import Namespace  # NOQA
from .parameters import (  # NOQA
    Parameters,
    PatchJSONParameters,
    PatchJSONParametersWithPassword,
    PostFormParameters,
)
from .resource import Resource  # NOQA
from .swagger import Swagger  # NOQA
