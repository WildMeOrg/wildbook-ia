# -*- coding: utf-8 -*-
from unittest import mock

from oauthlib.oauth2 import TokenExpiredError

from wbia.utils import call_houston


def disabled_test_call_houston(request):
    client_patch = mock.patch('wbia.utils.BackendApplicationClient')
    BackendApplicationClient = client_patch.start()
    request.addfinalizer(client_patch.stop)

    session = mock.Mock()
    session_patch = mock.patch('wbia.utils.OAuth2Session', return_value=session)
    OAuth2Session = session_patch.start()
    request.addfinalizer(session_patch.stop)

    getenv_patch = mock.patch(
        'wbia.utils.os.getenv',
        side_effect={
            'HOUSTON_CLIENT_ID': 'houston-client-id',
            'HOUSTON_CLIENT_SECRET': 'houston-client-secret',
        }.get,
    )
    getenv_patch.start()
    request.addfinalizer(getenv_patch.stop)

    response = mock.Mock()
    session.request.return_value = response

    # Case 1: Call houston for the first time
    result = call_houston(
        'houston+http://houston:5000/api/v1/users/me',
        misc=10,
    )
    assert BackendApplicationClient.call_count == 1
    assert OAuth2Session.call_count == 1
    assert session.fetch_token.call_count == 1
    assert session.fetch_token.call_args == mock.call(
        token_url='http://houston:5000/api/v1/auth/tokens',
        client_id='houston-client-id',
        client_secret='houston-client-secret',
    )
    assert session.request.call_count == 1
    assert session.request.call_args == mock.call(
        'GET',
        'http://houston:5000/api/v1/users/me',
        misc=10,
    )
    assert result == response
    BackendApplicationClient.reset_mock()
    OAuth2Session.reset_mock()
    session.request.reset_mock()
    session.fetch_token.reset_mock()

    # Case 2: Call houston again
    result = call_houston('houston+http://houston:5000/favicon.ico')
    assert not BackendApplicationClient.called
    assert not OAuth2Session.called
    assert not session.fetch_token.called
    assert session.request.call_count == 1
    assert session.request.call_args == mock.call(
        'GET',
        'http://houston:5000/favicon.ico',
    )
    assert result == response
    session.request.reset_mock()

    # Case 3: Token expired
    def session_request(*args, **kwargs):
        if session.request.call_count == 1:
            raise TokenExpiredError
        return response

    session.request.side_effect = session_request
    result = call_houston('houston+https://houston:5000/')
    assert not BackendApplicationClient.called
    assert not OAuth2Session.called
    assert session.fetch_token.call_count == 1
    assert session.fetch_token.call_args == mock.call(
        token_url='https://houston:5000/api/v1/auth/tokens',
        client_id='houston-client-id',
        client_secret='houston-client-secret',
    )
    assert session.request.call_count == 2
    assert session.request.call_args_list == [
        mock.call('GET', 'https://houston:5000/'),
        mock.call('GET', 'https://houston:5000/'),
    ]
    session.reset_mock()

    # Case 4: Token not expired but 401 returned
    def session_request(*args, **kwargs):
        if session.request.call_count == 1:
            response.status_code = 401
        return response

    session.request.side_effect = session_request
    result = call_houston('houston+https://houston:5000/')
    assert not BackendApplicationClient.called
    assert not OAuth2Session.called
    assert session.fetch_token.call_count == 1
    assert session.fetch_token.call_args == mock.call(
        token_url='https://houston:5000/api/v1/auth/tokens',
        client_id='houston-client-id',
        client_secret='houston-client-secret',
    )
    assert session.request.call_count == 2
    assert session.request.call_args_list == [
        mock.call('GET', 'https://houston:5000/'),
        mock.call('GET', 'https://houston:5000/'),
    ]
