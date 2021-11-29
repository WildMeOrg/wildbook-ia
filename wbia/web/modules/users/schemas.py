# -*- coding: utf-8 -*-
# pylint: disable=too-few-public-methods
"""
User schemas
------------
"""

from flask_restx_patched import ModelSchema

from .models import User


class BaseUserSchema(ModelSchema):
    """
    Base user schema exposes only the most general fields.
    """

    class Meta:
        # pylint: disable=missing-docstring
        model = User
        fields = (
            User.guid.key,
            User.email.key,
        )
        dump_only = (User.guid.key,)


class DetailedUserSchema(BaseUserSchema):
    """Detailed user schema exposes all fields used to render a normal user profile."""

    class Meta(BaseUserSchema.Meta):
        fields = BaseUserSchema.Meta.fields + (
            User.created.key,
            User.updated.key,
            User.viewed.key,
        )
