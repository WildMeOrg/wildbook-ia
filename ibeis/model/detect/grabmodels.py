from __future__ import absolute_import, division, print_function
import utool
from os.path import exists, join, realpath
(print, print_, printDBG, rrr, profile) = utool.inject(
    __name__, '[grabmodels]', DEBUG=False)


#DETECTMODELS_DIR = realpath(join(dirname(__file__), 'rf'))
DETECTMODELS_DIR = utool.get_app_resource_dir('ibeis', 'detectmodels')

MODEL_DIRS = {
    'rf': join(DETECTMODELS_DIR, 'rf'),
}

MODEL_URLS = {
    'rf': 'https://dl.dropboxusercontent.com/s/9814r3d2rkiq5t3/rf.zip'
}


def assert_models():
    for model_dir in MODEL_DIRS.values():
        assert exists(model_dir), ('model_dir=%r does not exist' % model_dir)


def ensure_models():
    utool.ensuredir(DETECTMODELS_DIR)
    for algo, model_dir in MODEL_DIRS.items():
        if not exists(model_dir):
            download_model(algo, model_dir)
    assert_models()


def get_model_dir(algo, ensure=True):
    if ensure:
        ensure_models()
    else:
        assert_models()
    return MODEL_DIRS[algo]


def download_model(algo, model_dir):
    zip_fpath = realpath(join(DETECTMODELS_DIR, algo + '.zip'))
    # Download and unzip model
    print('[grabmodels] Downloading model_dir=%s' % zip_fpath)
    dropbox_link = MODEL_URLS[algo]
    utool.download_url(dropbox_link, zip_fpath)
    utool.unzip_file(zip_fpath)
    # Cleanup
    utool.delete(zip_fpath)


def get_species_trees_paths(species):
    rf_model_dir   = MODEL_DIRS['rf']
    trees_path     = join(rf_model_dir, species)
    return trees_path


if __name__ == '__main__':
    ensure_models()
