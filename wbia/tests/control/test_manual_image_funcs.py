# -*- coding: utf-8 -*-
import pathlib
import tempfile
import uuid
from unittest import mock
from urllib.parse import quote

IMAGE_UUIDS = [
    uuid.UUID('784817e0-ca90-425d-8a81-edccfe3e538b'),
    uuid.UUID('8f0dba1e-534f-43a1-8bf2-78f80b20427d'),
]
IMAGE_URIS = [
    f'http://houston/api/v1/assets/src/{image_uuid}' for image_uuid in IMAGE_UUIDS
]
IMAGE_URIS_UTF8 = [
    f'http://housトン:5000/api/ヴィ1/assets/src/{image_uuid}' for image_uuid in IMAGE_UUIDS
]
QUOTED_URI_PATHS = [
    f'http://housトン:5000/api/{quote("ヴィ")}1/assets/src/{image_uuid}'
    for image_uuid in IMAGE_UUIDS
]
QUOTED_AGAIN_URIS = [
    f'http://hous{quote("トン")}:5000/api/{quote("ヴィ")}1/assets/src/{image_uuid}'
    for image_uuid in IMAGE_UUIDS
]


def test_add_images_uris():
    from wbia.control.manual_image_funcs import _compute_image_uuids, add_images

    ibs = mock.Mock()
    ibs.cfg.other_cfg.auto_localize = False
    ibs.get_image_paths.return_value = IMAGE_URIS_UTF8
    ibs._compute_image_uuids.side_effect = lambda *args, **kwargs: _compute_image_uuids(
        ibs, *args, **kwargs
    )
    ibs.db.add_cleanly.return_value = [1, 2]
    ibs.check_image_loadable.return_value = [], []

    response = mock.Mock(status_code=200)
    with open('wbia/web/static/images/logo-wildme.png', 'rb') as f:
        response.iter_content.return_value = [f.read()]

    def _get(*args, **kwargs):
        if get.call_count == 1:
            return mock.Mock(status_code=500)
        return response

    with mock.patch('requests.get', side_effect=_get) as get:
        result_ids = add_images(ibs, IMAGE_URIS_UTF8)

    assert result_ids == [1, 2]
    assert get.call_args_list == [
        mock.call(QUOTED_URI_PATHS[0], allow_redirects=True, stream=True),
        mock.call(QUOTED_AGAIN_URIS[0], allow_redirects=True, stream=True),
        mock.call(QUOTED_URI_PATHS[1], allow_redirects=True, stream=True),
    ]
    assert not ibs.localize_images.called


def test_add_images_auto_localized():
    from wbia.control.manual_image_funcs import _compute_image_uuids, add_images

    ibs = mock.Mock()
    ibs.cfg.other_cfg.auto_localize = False
    ibs.get_image_paths.return_value = [f'{image_uri}.png' for image_uri in IMAGE_URIS]
    ibs._compute_image_uuids.side_effect = lambda *args, **kwargs: _compute_image_uuids(
        ibs, *args, **kwargs
    )
    ibs.db.add_cleanly.return_value = [1, 2]
    ibs.check_image_loadable.return_value = [], []

    response = mock.Mock(status_code=200)
    with open('wbia/web/static/images/logo-wildme.png', 'rb') as f:
        response.iter_content.return_value = [f.read()]

    def _get(*args, **kwargs):
        if get.call_count == 1:
            return mock.Mock(status_code=500)
        return response

    with mock.patch('requests.get', side_effect=_get) as get:
        result_ids = add_images(ibs, IMAGE_URIS, auto_localize=True)

    assert result_ids == [1, 2]
    assert get.call_args_list == [
        mock.call(IMAGE_URIS[0], allow_redirects=True, stream=True),
        mock.call(IMAGE_URIS[0], allow_redirects=True, stream=True),
        mock.call(IMAGE_URIS[1], allow_redirects=True, stream=True),
    ]
    assert ibs.localize_images.call_args_list[0][0] == ([1, 2],)


def test_localize_images(request):
    from wbia.control.manual_image_funcs import localize_images

    td = tempfile.TemporaryDirectory()
    imgdir = pathlib.Path(td.name)
    request.addfinalizer(td.cleanup)
    ibs = mock.Mock(imgdir=str(imgdir))
    ibs.get_image_uris.return_value = IMAGE_URIS_UTF8
    ibs.get_image_uuids.return_value = IMAGE_UUIDS
    ibs.get_image_exts.return_value = ['.png', '.png']

    response = mock.Mock(status_code=200)
    with open('wbia/web/static/images/logo-wildme.png', 'rb') as f:
        response.iter_content.return_value = [f.read()]

    def _get(*args, **kwargs):
        if get.call_count == 1:
            return mock.Mock(status_code=500)
        return response

    with mock.patch(
        'wbia.control.manual_image_funcs.ut.grab_s3_contents'
    ) as grab_s3_contents:
        with mock.patch('requests.get', side_effect=_get) as get:
            localize_images(ibs, [1, 2])

    assert get.call_args_list == [
        mock.call(QUOTED_URI_PATHS[0], stream=True, allow_redirects=True),
        mock.call(QUOTED_AGAIN_URIS[0], stream=True, allow_redirects=True),
        mock.call(QUOTED_URI_PATHS[1], stream=True, allow_redirects=True),
    ]
    assert not grab_s3_contents.called
    assert sorted(imgdir.glob('*')) == [
        imgdir / f'{image_uuid}.png' for image_uuid in IMAGE_UUIDS
    ]
    for image_uuid in IMAGE_UUIDS:
        with open('wbia/web/static/images/logo-wildme.png', 'rb') as f:
            with (imgdir / f'{image_uuid}.png').open('rb') as g:
                assert f.read() == g.read()
    assert ibs.set_image_uris.call_args_list == [
        mock.call([1, 2], [f'{image_uuid}.png' for image_uuid in IMAGE_UUIDS]),
    ]
