# -*- coding: utf-8 -*-
"""
OAuth2 provider models.

It is based on the code from the example:
https://github.com/lepture/example-oauth2-server

More details are available here:
* http://flask-oauthlib.readthedocs.org/en/latest/oauth2.html
* http://lepture.com/en/2013/create-oauth-server
"""
import datetime
import enum
import logging
import uuid

import pytz
from sqlalchemy_utils.types import ScalarListType

from wbia.web.extensions import db
from wbia.web.extensions.auth import security
from wbia.web.modules.users.models import User

log = logging.getLogger(__name__)


class OAuth2Client(db.Model):
    """
    Model that binds OAuth2 Client ID and Secret to a specific User.
    """

    __tablename__ = 'oauth2_client'

    guid = db.Column(db.GUID, default=uuid.uuid4, primary_key=True)
    secret = db.Column(
        db.String(length=64), default=security.generate_random_64, nullable=False
    )

    user_guid = db.Column(
        db.ForeignKey('user.guid', ondelete='CASCADE'), index=True, nullable=False
    )
    user = db.relationship(User)

    class ClientLevels(str, enum.Enum):
        public = 'public'
        session = 'session'
        confidential = 'confidential'

    level = db.Column(db.Enum(ClientLevels), default=ClientLevels.public, nullable=False)
    redirect_uris = db.Column(ScalarListType(separator=' '), default=[], nullable=False)
    default_scopes = db.Column(ScalarListType(separator=' '), nullable=False)

    @property
    def default_redirect_uri(self):
        redirect_uris = self.redirect_uris
        if redirect_uris:
            return redirect_uris[0]
        return None

    @property
    def client_id(self):
        return self.guid

    @property
    def client_secret(self):
        return self.secret

    @classmethod
    def find(cls, guid):
        if not guid:
            return None
        return cls.query.get(guid)

    def validate_scopes(self, scopes):
        # The only reason for this override is that Swagger UI has a bug which leads to that
        # `scope` parameter contains extra spaces between scopes:
        # https://github.com/frol/flask-restplus-server-example/issues/131
        return set(self.default_scopes).issuperset(set(scopes) - {''})

    def delete(self):
        with db.session.begin():
            db.session.delete(self)


class OAuth2Grant(db.Model):
    """
    Intermediate temporary helper for OAuth2 Grants.
    """

    __tablename__ = 'oauth2_grant'

    guid = db.Column(
        db.GUID, default=uuid.uuid4, primary_key=True
    )  # pylint: disable=invalid-name

    user_guid = db.Column(
        db.ForeignKey('user.guid', ondelete='CASCADE'), index=True, nullable=False
    )
    user = db.relationship('User')

    client_guid = db.Column(
        db.GUID,
        db.ForeignKey('oauth2_client.guid'),
        index=True,
        nullable=False,
    )
    client = db.relationship('OAuth2Client')

    code = db.Column(db.String(length=255), index=True, nullable=False)

    redirect_uri = db.Column(db.String(length=255), nullable=False)
    expires = db.Column(db.DateTime, nullable=False)

    scopes = db.Column(ScalarListType(separator=' '), nullable=False)

    def delete(self):
        with db.session.begin():
            db.session.delete(self)

    @property
    def client_id(self):
        return self.client_guid

    @classmethod
    def find(cls, client_guid, code):
        return cls.query.filter_by(client_guid=client_guid, code=code).first()

    @property
    def is_expired(self):
        now_utc = datetime.datetime.now(tz=pytz.utc)
        expired = now_utc > self.expires.replace(tzinfo=pytz.utc)
        return expired


class OAuth2Token(db.Model):
    """
    OAuth2 Access Tokens storage model.
    """

    __tablename__ = 'oauth2_token'

    guid = db.Column(
        db.GUID, default=uuid.uuid4, primary_key=True
    )  # pylint: disable=invalid-name

    client_guid = db.Column(
        db.GUID,
        db.ForeignKey('oauth2_client.guid'),
        index=True,
        nullable=False,
    )
    client = db.relationship('OAuth2Client')

    user_guid = db.Column(
        db.ForeignKey('user.guid', ondelete='CASCADE'), index=True, nullable=False
    )
    user = db.relationship('User')

    class TokenTypes(str, enum.Enum):
        # currently only bearer is supported
        Bearer = 'Bearer'

    token_type = db.Column(db.Enum(TokenTypes), nullable=False)

    access_token = db.Column(
        db.String(length=128),
        default=security.generate_random_128,
        unique=True,
        nullable=False,
    )
    refresh_token = db.Column(
        db.String(length=128),
        default=security.generate_random_128,
        unique=True,
        nullable=True,
    )
    expires = db.Column(db.DateTime, nullable=False)
    scopes = db.Column(ScalarListType(separator=' '), nullable=False)

    @property
    def client_id(self):
        return self.client_guid

    @classmethod
    def find(cls, access_token=None, refresh_token=None):
        response = None

        if access_token:
            response = cls.query.filter_by(access_token=access_token).first()

        if refresh_token:
            response = cls.query.filter_by(refresh_token=refresh_token).first()

        return response

    def delete(self):
        with db.session.begin():
            db.session.delete(self)

    @property
    def is_expired(self):
        now_utc = datetime.datetime.now(tz=pytz.utc)
        expired = now_utc > self.expires.replace(tzinfo=pytz.utc)
        return expired
