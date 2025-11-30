# -*- coding: utf-8 -*-
import logging
from os.path import exists, join, realpath
import os
import urllib.request
import urllib.parse
import time

import utool as ut

(print, rrr, profile) = ut.inject2(__name__, '[grabmodels]')
logger = logging.getLogger('wbia')


# DETECTMODELS_DIR = realpath(join(dirname(__file__), 'rf'))
# DEFAULT_DETECTMODELS_DIR = ut.get_app_resource_dir('wbia', 'detectmodels')

def _choose_default_detectmodels_dir():
    """
    Resolve the base detectmodels directory with this priority:
      1. MODEL_DIR env var (docker volume). Can be:
         - the detectmodels directory itself,
         - a directory containing a 'detectmodels' subdir,
         - a base dir where we'll create/use a 'detectmodels' subdir.
      2. WBIA_MODELS_DIR env var (legacy compat; same structural rules)
      3. /models (same structural rules as above – typical mount point)
      4. Original per-user/app cache: ~/.cache/wbia/detectmodels (ut.get_app_resource_dir)

    """
    def _resolve_detectmodels_from_base(cand_dir):
        """Given a base directory, resolve a detectmodels dir under it.
        If cand_dir is already a detectmodels dir or contains one, return that.
        Otherwise, create/use cand_dir/detectmodels.
        """
        if not cand_dir or not os.path.isdir(cand_dir):
            return None
        # If it contains a detectmodels subdir, prefer that
        if os.path.isdir(join(cand_dir, 'detectmodels')):
            return realpath(join(cand_dir, 'detectmodels'))
        # If it already looks like the detectmodels dir (name or has rf)
        if os.path.basename(cand_dir).lower() == 'detectmodels' or os.path.isdir(
            join(cand_dir, 'rf')
        ):
            return realpath(cand_dir)
        # Otherwise, create/use a detectmodels subdir
        detectmodels_dir = realpath(join(cand_dir, 'detectmodels'))
        ut.ensuredir(detectmodels_dir)
        return detectmodels_dir

    # Priority list of candidate bases
    candidates = []
    model_dir = os.getenv('MODEL_DIR')  # preferred new env var
    if model_dir:
        candidates.append(model_dir)
    legacy_env_dir = os.getenv('WBIA_MODELS_DIR')
    if legacy_env_dir:
        candidates.append(legacy_env_dir)
    candidates.append('/models')

    for cand in candidates:
        resolved = _resolve_detectmodels_from_base(cand)
        if resolved is not None and os.path.isdir(resolved):
            logger.info('[grabmodels] Using mounted detectmodels dir: %s', resolved)
            return resolved

    # Fallback to app cache
    fallback = ut.get_app_resource_dir('wbia', 'detectmodels')
    ut.ensuredir(fallback)
    logger.info('[grabmodels] Using cached detectmodels dir: %s', fallback)
    return fallback


# Original default (now override-aware)
DEFAULT_DETECTMODELS_DIR = _choose_default_detectmodels_dir()

DETECTOR_KEY_RF = 'rf'

MODEL_ALGO_SUBDIRS = {
    DETECTOR_KEY_RF: 'rf',
}

MODEL_URLS = {
    DETECTOR_KEY_RF: 'https://wildbookiarepository.azureedge.net/models/rf.v3.zip',
}

# Optional: MD5 checksums for integrity verification (set MODEL_VERIFY_CHECKSUM=1 to enable)
MODEL_CHECKSUMS = {
    DETECTOR_KEY_RF: None,  # Add MD5 hash here if available
}


def _expand_modeldir(modeldir='default'):
    """returns default unless another path is specified"""
    if modeldir == 'default':
        modeldir = DEFAULT_DETECTMODELS_DIR
    return modeldir


def get_species_trees_paths(species, modeldir='default'):
    r"""
    Args:
        species (?):
        modeldir (str):

    Returns:
        ?: trees_path

    CommandLine:
        python -m wbia.algo.detect.grabmodels --test-get_species_trees_paths

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.detect.grabmodels import *  # NOQA
        >>> import wbia
        >>> from wbia import constants as const
        >>> # build test data
        >>> species = const.TEST_SPECIES.ZEB_PLAIN
        >>> modeldir = 'default'
        >>> # execute function
        >>> trees_path = get_species_trees_paths(species, modeldir)
        >>> # verify results
        >>> result = str(trees_path)
        >>> print(result)
    """
    modeldir = _expand_modeldir(modeldir)
    algosubdir = MODEL_ALGO_SUBDIRS[DETECTOR_KEY_RF]
    rf_model_dir = join(modeldir, algosubdir)
    trees_path = join(rf_model_dir, species)
    return trees_path


def iter_algo_modeldirs(modeldir='default', ensurebase=False):
    modeldir = _expand_modeldir(modeldir)
    if ensurebase:
        ut.ensuredir(modeldir)
    for algo, algosubdir in MODEL_ALGO_SUBDIRS.items():
        yield algo, join(modeldir, algosubdir)


def assert_models(modeldir='default', verbose=True):
    for algo, algo_modeldir in iter_algo_modeldirs(modeldir):
        ut.assertpath(algo_modeldir, verbose=verbose)
        # assert ut.checkpath(algo_modeldir, verbose=True), ('algo_modeldir=%r does not exist' % algo_modeldir)


def ensure_models(modeldir='default', verbose=True):
    r"""
    Args:
        modeldir (str):

    CommandLine:
        python -m wbia.algo.detect.grabmodels --test-ensure_models

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.detect.grabmodels import *  # NOQA
        >>> modeldir = 'default'
        >>> result = ensure_models(modeldir)
        >>> print(result)
    """
    modeldir = _expand_modeldir(modeldir)
    for algo, algo_modeldir in iter_algo_modeldirs(modeldir, ensurebase=True):
        if not exists(algo_modeldir):
            _download_model(algo, algo_modeldir)
    assert_models(modeldir, verbose=verbose)


def redownload_models(modeldir='default', verbose=True):
    r"""
    Args:
        modeldir (str): (default = 'default')
        verbose (bool):  verbosity flag(default = True)

    CommandLine:
        python -m wbia.algo.detect.grabmodels --test-redownload_models

    Example:
        >>> # SCRIPT
        >>> from wbia.algo.detect.grabmodels import *  # NOQA
        >>> result = redownload_models()
    """
    logger.info('[grabmodels] redownload_detection_models')
    modeldir = _expand_modeldir(modeldir)
    ut.delete(modeldir)
    ensure_models(modeldir=modeldir, verbose=verbose)
    if verbose:
        logger.info('[grabmodels] finished redownload_detection_models')


def _download_model(algo, algo_modeldir):
    """
    Download and overwrites models
    """
    # Build the source URL with optional SAS token from environment variables
    base_url = MODEL_URLS[algo]
    url_with_sas, masked_url = _build_sas_url(base_url)

    # Determine destination filename from URL (without query)
    parsed = urllib.parse.urlparse(url_with_sas)
    dest_name = os.path.basename(parsed.path) or (algo + '.bin')
    dest_fpath = realpath(join(algo_modeldir, dest_name))

    # Download without leaking SAS token in logs
    logger.info('[grabmodels] Downloading model for %s from %s -> %s', algo, masked_url, dest_fpath)
    _stream_download(url_with_sas, dest_fpath)

    # Optional integrity check
    if os.getenv('MODEL_VERIFY_CHECKSUM') == '1' and MODEL_CHECKSUMS.get(algo):
        _verify_checksum(dest_fpath, MODEL_CHECKSUMS[algo])

    # If it's a zip, unzip and remove the archive
    if dest_name.lower().endswith('.zip'):
        try:
            ut.unzip_file(dest_fpath)
        finally:
            ut.delete(dest_fpath)


def _stream_download(url, dest_fpath, chunk_size=1 << 20, retries=3, backoff=2.0, timeout=60):
    """Download a URL to a file path with streaming, retries, and timeout."""
    ut.ensuredir(os.path.dirname(dest_fpath))
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'wbia-grabmodels/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_fpath, 'wb') as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
            return
        except Exception as ex:
            last_err = ex
            if attempt < retries:
                sleep_for = backoff ** (attempt - 1)
                logger.warning('[grabmodels] download failed (attempt %d/%d): %s; retrying in %.1fs', attempt, retries, type(ex).__name__, sleep_for)
                time.sleep(sleep_for)
            else:
                break
    # If all retries failed, raise the last error
    raise last_err


def _build_sas_url(base_url):
    """
    Compose a URL with an optional SAS token from env in a safe way.
    Environment variables checked (in order):
      - MODEL_SAS_QUERY (recommended; either starts with '?' or not)
      - MODEL_SAS_TOKEN (alias)
      - WBIA_MODELS_SAS (legacy)

    Returns (full_url, masked_url_for_logs)
    """
    sas = os.getenv('MODEL_SAS_QUERY') or os.getenv('MODEL_SAS_TOKEN') or os.getenv('WBIA_MODELS_SAS')
    if not sas:
        # Nothing to append
        return base_url, _mask_sas(base_url)
    sas = sas.strip()
    # Normalize: remove leading '?' so we can decide ? or &
    if sas.startswith('?'):
        sas = sas[1:]
    parsed = urllib.parse.urlparse(base_url)
    if parsed.query:
        query = parsed.query + '&' + sas
    else:
        query = sas
    full = parsed._replace(query=query).geturl()
    return full, _mask_sas(full)


def _mask_sas(url):
    """Mask any query string to avoid leaking SAS tokens in logs."""
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.query:
            return url
        # If query contains sig=, redact only its value; otherwise redact entire query
        qparams = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        redacted = []
        for k, v in qparams:
            if k.lower() in {'sig', 'signature', 'se'}:
                redacted.append((k, '***'))
            else:
                redacted.append((k, '***'))
        masked_query = urllib.parse.urlencode(redacted)
        return parsed._replace(query=masked_query).geturl()
    except Exception:
        # On any failure, drop the query entirely
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed._replace(query='').geturl()
        except Exception:
            return 'URL(redacted)'


def _verify_checksum(filepath, expected_md5):
    """Verify MD5 checksum of downloaded file."""
    import hashlib
    logger.info('[grabmodels] Verifying checksum for %s', filepath)
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            md5.update(chunk)
    actual = md5.hexdigest()
    if actual != expected_md5:
        raise ValueError(
            f'Checksum mismatch for {filepath}: '
            f'expected {expected_md5}, got {actual}'
        )
    logger.info('[grabmodels] Checksum verified: %s', expected_md5)
