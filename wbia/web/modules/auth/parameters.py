# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-order
"""
Input arguments (Parameters) for Auth resources
-----------------------------------------------
"""
from flask_login import current_user
from flask_marshmallow import base_fields
from marshmallow import ValidationError, validates

from flask_restx_patched import Parameters
from wbia.web.extensions import api
from wbia.web.extensions.api.parameters import PaginationParameters


class ListOAuth2ClientsParameters(PaginationParameters):
    user_guid = base_fields.UUID(required=True)

    @validates('user_guid')
    def validate_user_guid(self, data):
        if current_user.guid != data:
            raise ValidationError('It is only allowed to query your own OAuth2 clients.')


class CreateOAuth2SessionParameters(Parameters):
    email = base_fields.Email(description='Example: root@gmail.com', required=True)
    password = base_fields.String(description='No rules yet', required=True)


class CreateOAuth2ClientParameters(Parameters):
    redirect_uris = base_fields.List(base_fields.String, required=False)
    default_scopes = base_fields.List(base_fields.String, required=True)

    @validates('default_scopes')
    def validate_default_scopes(self, data):
        unknown_scopes = set(data) - set(
            api.api_v2.authorizations['oauth2_password']['scopes']
        )
        if unknown_scopes:
            raise ValidationError(
                "'%s' scope(s) are not supported." % (', '.join(unknown_scopes))
            )
