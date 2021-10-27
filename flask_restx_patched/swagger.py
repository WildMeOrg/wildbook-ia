# -*- coding: utf-8 -*-
from apispec.ext.marshmallow.swagger import schema2parameters
from flask_restx.swagger import Swagger as OriginalSwagger


class Swagger(OriginalSwagger):
    def parameters_for(self, doc):
        schema = doc['params']

        if not schema:
            return []
        if isinstance(schema, list):
            return schema
        if isinstance(schema, dict) and all(
            isinstance(field, dict) for field in schema.values()
        ):
            return list(schema.values())

        if 'in' in schema.context and 'json' in schema.context['in']:
            default_location = 'body'
        else:
            default_location = 'query'
        return schema2parameters(schema, default_in=default_location, required=True)

    def expected_params(self, doc):
        if 'params' in doc.keys():
            params = doc.get('params', None)
            assert params is not None

            for name, param in params.items():
                get_func = getattr(param, 'get', None)
                if get_func is None:

                    def _patched_get(value, default):
                        return param.context.get(value, default)

                    param.get = _patched_get

            doc['params'] = params
        return super(Swagger, self).expected_params(doc)
