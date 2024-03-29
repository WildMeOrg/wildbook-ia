# -*- coding: utf-8 -*-
import logging

import requests
from flask import jsonify
from flask import make_response as original_flask_make_response
from flask_restx import Api as OriginalApi
from flask_restx._http import HTTPStatus
from werkzeug import cached_property

from .namespace import Namespace
from .swagger import Swagger

log = logging.getLogger(__name__)


class Api(OriginalApi):
    @cached_property
    def __schema__(self):
        """
        The Swagger specifications/schema for this API

        :returns dict: the schema as a serializable dict
        """
        if not self._schema:
            try:
                self._schema = Swagger(self).as_dict()
            except Exception:
                # Log the source exception for debugging purpose
                # and return an error message
                msg = 'Unable to render schema'
                log.exception(msg)  # This will provide a full traceback
                return {'error': msg}
        return self._schema

    def init_app(self, app, **kwargs):
        # This solves the issue of late resources registration:
        # https://github.com/frol/flask-restplus-server-example/issues/110
        # https://github.com/noirbizarre/flask-restplus/pull/483
        self.app = app

        super(Api, self).init_app(app, **kwargs)
        app.errorhandler(HTTPStatus.UNPROCESSABLE_ENTITY.value)(handle_validation_error)

    def namespace(self, *args, **kwargs):
        # The only purpose of this method is to pass a custom Namespace class
        _namespace = Namespace(*args, **kwargs)
        self.add_namespace(_namespace)
        return _namespace

    def make_response(self, data, *args, **kwargs):
        if isinstance(data, requests.models.Response):
            headers = list(data.headers.items())
            response = original_flask_make_response(
                data.content, data.status_code, headers
            )
            for key in response.headers.keys():
                if key not in data.headers:
                    response.headers.pop(key)
            assert response.status_code == data.status_code
        else:
            response = super(Api, self).make_response(data, *args, **kwargs)
        return response


# Return validation errors as JSON
def handle_validation_error(err):
    exc = err.data['exc']
    return (
        jsonify(
            {'status': HTTPStatus.UNPROCESSABLE_ENTITY.value, 'message': exc.messages}
        ),
        HTTPStatus.UNPROCESSABLE_ENTITY.value,
    )
