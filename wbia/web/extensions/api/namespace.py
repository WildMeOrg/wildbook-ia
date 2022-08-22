# -*- coding: utf-8 -*-
"""
Extended Api Namespace implementation with an application-specific helpers
--------------------------------------------------------------------------
"""
import logging
from contextlib import contextmanager
from functools import wraps

import sqlalchemy
from marshmallow import ValidationError

from flask_restx_patched._http import HTTPStatus
from flask_restx_patched.namespace import Namespace as BaseNamespace

from . import http_exceptions
from .webargs_parser import CustomWebargsParser

log = logging.getLogger(__name__)


class Namespace(BaseNamespace):
    """
    Having app-specific handlers here.
    """

    WEBARGS_PARSER = CustomWebargsParser()

    def paginate(self, parameters=None, locations=None):
        """
        Endpoint parameters registration decorator special for pagination.
        If ``parameters`` is not provided default PaginationParameters will be
        used.

        Also, any custom Parameters can be used, but it needs to have ``limit`` and ``offset``
        fields.
        """
        if not parameters:
            # Use default parameters if None specified
            from bia.web.extensions.api.parameters import PaginationParameters

            parameters = PaginationParameters()

        if not all(
            mandatory in parameters.declared_fields for mandatory in ('limit', 'offset')
        ):
            raise AttributeError(
                '`limit` and `offset` fields must be in Parameter passed to `paginate()`'
            )

        def decorator(func):
            @wraps(func)
            def wrapper(self_, parameters_args, *args, **kwargs):
                queryset = func(self_, parameters_args, *args, **kwargs)
                total_count = queryset.count()
                return (
                    queryset.offset(parameters_args['offset']).limit(
                        parameters_args['limit']
                    ),
                    HTTPStatus.OK,
                    {'X-Total-Count': total_count},
                )

            return self.parameters(parameters, locations)(wrapper)

        return decorator

    @contextmanager
    def commit_or_abort(
        self, session, default_error_message='The operation failed to complete'
    ):
        """
        Context manager to simplify a workflow in resources

        Args:
            session: db.session instance
            default_error_message: Custom error message

        Exampple:
        >>> with api.commit_or_abort(db.session):
        ...     family = Family(**args)
        ...     db.session.add(family)
        ...     return family
        """
        try:
            with session.begin():
                yield
        except (ValueError, ValidationError) as exception:
            log.info('Database transaction was rolled back due to: %r', exception)
            http_exceptions.abort(code=HTTPStatus.CONFLICT, message=str(exception))
        except sqlalchemy.exc.IntegrityError as exception:
            log.info('Database transaction was rolled back due to: %r', exception)
            http_exceptions.abort(code=HTTPStatus.CONFLICT, message=default_error_message)
