# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""
Auth schemas
------------
"""

from flask_marshmallow import base_fields

from flask_restx_patched import ModelSchema, Schema

from .models import OAuth2Client


class BaseOAuth2ClientSchema(ModelSchema):
    """
    Base OAuth2 client schema exposes only the most general fields.
    """

    default_scopes = base_fields.List(base_fields.String, required=True)
    redirect_uris = base_fields.List(base_fields.String, required=True)

    class Meta:
        # pylint: disable=missing-docstring
        model = OAuth2Client
        fields = (
            OAuth2Client.guid.key,
            OAuth2Client.user_guid.key,
            OAuth2Client.level.key,
            OAuth2Client.default_scopes.key,
            OAuth2Client.redirect_uris.key,
        )
        dump_only = (
            OAuth2Client.guid.key,
            OAuth2Client.user_guid.key,
        )


class DetailedOAuth2ClientSchema(BaseOAuth2ClientSchema):
    """
    Detailed OAuth2 client schema exposes all useful fields.
    """

    class Meta(BaseOAuth2ClientSchema.Meta):
        fields = BaseOAuth2ClientSchema.Meta.fields + (OAuth2Client.secret.key,)


class ReCaptchaPublicServerKeySchema(Schema):
    recaptcha_public_key = base_fields.String(required=True)
