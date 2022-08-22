# -*- coding: utf-8 -*-
"""
User database models
--------------------
"""
import logging
import uuid

from flask_login import current_user  # NOQA
from sqlalchemy_utils import types as column_types

from wbia.web.extensions import TimestampViewed, db

log = logging.getLogger(__name__)


class User(db.Model, TimestampViewed):
    """
    User database model.
    """

    def __init__(self, *args, **kwargs):
        if 'password' not in kwargs:
            raise ValueError('User must have a password')
        super().__init__(*args, **kwargs)

    guid = db.Column(
        db.GUID, default=uuid.uuid4, primary_key=True
    )  # pylint: disable=invalid-name
    email = db.Column(db.String(length=120), index=True, unique=True, nullable=False)

    password = db.Column(
        column_types.PasswordType(max_length=128, schemes=('bcrypt',)), nullable=False
    )  # can me migrated from EDM field "password"

    def __repr__(self):
        return (
            '<{class_name}('
            'guid={self.guid}, '
            'email="{self.email}", '
            ')>'.format(class_name=self.__class__.__name__, self=self)
        )

    @property
    def is_active(self):
        return True

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return self.guid
