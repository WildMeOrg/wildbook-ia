from __future__ import absolute_import, division, print_function
from vtool.tests import __sysreq__  # NOQA
import utool
from os.path import exists, dirname, join, realpath

RF_MODEL_DIR = realpath(join(dirname(__file__), 'rf'))

MODEL_DIRS = {
    'rf': RF_MODEL_DIR,
}

URLS = {
    'rf': 'https://dl.dropboxusercontent.com/s/9814r3d2rkiq5t3/rf.zip'    
}

def assert_models():
    for MODEL_DIR in MODEL_DIRS.values():
        assert exists(MODEL_DIR), ('MODEL_DIR=%r does not exist' % MODEL_DIR)


def ensure_models():
    for TYPE, MODEL_DIR in MODEL_DIRS.items():
        if not exists(MODEL_DIR):
            download_model(TYPE, MODEL_DIR)
    assert_models()


def get_model_dir(TYPE, ensure=True):
    if ensure:
        ensure_models()
    else:
        assert_models()
    return MODEL_DIRS[TYPE]


def download_model(TYPE, MODEL_DIR):
    zip_fpath = realpath(join(MODEL_DIR, '..', TYPE+'.zip'))
   
    # Download and unzip model
    print('[grabmodels] Downloading MODEL_DIR=%s' % MODEL_DIR)

    dropbox_link = URLS[TYPE]
    utool.download_url(dropbox_link, zip_fpath)
    utool.unzip_file(zip_fpath)
    # Cleanup
    utool.delete(zip_fpath)

if __name__ == '__main__':
    ensure_models()
