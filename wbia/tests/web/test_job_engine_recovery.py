# -*- coding: utf-8 -*-
"""Startup recovery must survive a truncated / unreadable job record .pkl.

Regression test for a production startup failure:

    queue_interrupted_jobs -> initialize_process_record -> ut.load_cPkl
    _pickle.UnpicklingError: pickle data was truncated

A job record that was only partially written (process killed mid-write)
must be quarantined into the ``_ARCHIVE`` directory and reported as
archived+corrupted so that the rest of startup can proceed.
"""
import glob
import pickle
from os.path import basename, exists, join

import pytest
import utool as ut

from wbia.web import job_engine

GOOD_RECORD = {
    'request': {'action': 'noop', 'args': (), 'kwargs': {}},
    'attempts': 0,
    'completed': False,
}


def _make_dirs(tmp_path):
    shelve_path = str(tmp_path / 'engine_shelves')
    archive_path = '{}_ARCHIVE'.format(shelve_path)
    ut.ensuredir(shelve_path)
    ut.ensuredir(archive_path)
    return shelve_path, archive_path


def _unpack(values):
    (
        jobcounter,
        jobid,
        engine_request,
        archived,
        completed,
        suppressed,
        corrupted,
    ) = values
    return {
        'jobcounter': jobcounter,
        'jobid': jobid,
        'engine_request': engine_request,
        'archived': archived,
        'completed': completed,
        'suppressed': suppressed,
        'corrupted': corrupted,
    }


def test_truncated_record_is_archived_not_fatal(tmp_path):
    shelve_path, archive_path = _make_dirs(tmp_path)
    jobid = 'jobid-truncated'
    record_filepath = join(shelve_path, '{}.pkl'.format(jobid))

    # Write a valid record, then chop it in half to simulate a crash mid-write
    ut.save_cPkl(record_filepath, GOOD_RECORD, verbose=False)
    with open(record_filepath, 'rb') as file_:
        data = file_.read()
    with open(record_filepath, 'wb') as file_:
        file_.write(data[: len(data) // 2])

    values = job_engine.initialize_process_record(
        record_filepath, shelve_path, archive_path, 0, job_store=None
    )
    result = _unpack(values)

    assert result['jobid'] == jobid
    assert result['engine_request'] is None
    assert result['archived'] is True
    assert result['corrupted'] is True
    # Quarantined: gone from the live shelves dir, present in the archive
    assert not exists(record_filepath)
    assert exists(join(archive_path, '{}.pkl'.format(jobid)))


def test_empty_record_is_archived_not_fatal(tmp_path):
    shelve_path, archive_path = _make_dirs(tmp_path)
    jobid = 'jobid-empty'
    record_filepath = join(shelve_path, '{}.pkl'.format(jobid))
    open(record_filepath, 'wb').close()

    values = job_engine.initialize_process_record(
        record_filepath, shelve_path, archive_path, 0, job_store=None
    )
    result = _unpack(values)

    assert result['archived'] is True
    assert result['corrupted'] is True
    assert not exists(record_filepath)
    assert exists(join(archive_path, '{}.pkl'.format(jobid)))


def test_non_dict_record_is_archived_not_fatal(tmp_path):
    shelve_path, archive_path = _make_dirs(tmp_path)
    jobid = 'jobid-notadict'
    record_filepath = join(shelve_path, '{}.pkl'.format(jobid))
    ut.save_cPkl(record_filepath, ['not', 'a', 'record'], verbose=False)

    values = job_engine.initialize_process_record(
        record_filepath, shelve_path, archive_path, 0, job_store=None
    )
    result = _unpack(values)

    assert result['archived'] is True
    assert result['corrupted'] is True
    assert not exists(record_filepath)
    assert exists(join(archive_path, '{}.pkl'.format(jobid)))


def test_truncated_record_drops_sqlite_row_and_sweeps_leftover_tmp(tmp_path):
    from wbia.web.job_store import JobStore

    shelve_path, archive_path = _make_dirs(tmp_path)
    jobid = 'jobid-with-row'
    record_filepath = join(shelve_path, '{}.pkl'.format(jobid))
    open(record_filepath, 'wb').close()
    # A temp file abandoned by a crash mid-write must travel with the record
    leftover_tmp = '{}.deadbeef.tmp'.format(record_filepath)
    open(leftover_tmp, 'wb').close()

    job_store = JobStore(join(shelve_path, 'jobs.db'))
    try:
        job_store.register_job(jobid, 'working', 7)
        assert job_store.job_exists(jobid)

        values = job_engine.initialize_process_record(
            record_filepath, shelve_path, archive_path, 0, job_store=job_store
        )
        result = _unpack(values)

        assert result['archived'] is True
        assert not job_store.job_exists(jobid)
    finally:
        job_store.close()

    assert not exists(record_filepath)
    assert not exists(leftover_tmp)
    assert exists(join(archive_path, '{}.pkl'.format(jobid)))
    assert exists(join(archive_path, basename(leftover_tmp)))


def test_quarantine_failure_skips_record_for_this_boot(tmp_path, monkeypatch):
    shelve_path, archive_path = _make_dirs(tmp_path)
    jobid = 'jobid-unquarantinable'
    record_filepath = join(shelve_path, '{}.pkl'.format(jobid))
    open(record_filepath, 'wb').close()

    def _boom(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(job_engine, '_archive_job_files', _boom)

    # Must not raise: startup skips the record instead of dying
    values = job_engine.initialize_process_record(
        record_filepath, shelve_path, archive_path, 0, job_store=None
    )
    result = _unpack(values)

    assert result['archived'] is True
    assert result['engine_request'] is None
    # Original left in place for the next boot to retry
    assert exists(record_filepath)


def test_late_quarantine_failure_still_skips_record(tmp_path):
    shelve_path, archive_path = _make_dirs(tmp_path)
    jobid = 'jobid-late-failure'
    record_filepath = join(shelve_path, '{}.pkl'.format(jobid))
    open(record_filepath, 'wb').close()

    class _StoreThatCannotDelete(object):
        def get_metadata(self, jobid):
            return None

        def delete_job(self, jobid):
            raise RuntimeError('database is locked')

    # SQLite step fails AFTER the file has been moved: must still not raise
    values = job_engine.initialize_process_record(
        record_filepath,
        shelve_path,
        archive_path,
        0,
        job_store=_StoreThatCannotDelete(),
    )
    result = _unpack(values)

    assert result['archived'] is True
    assert result['engine_request'] is None
    # Partial quarantine: file work completed before the failure
    assert not exists(record_filepath)
    assert exists(join(archive_path, '{}.pkl'.format(jobid)))


def test_save_engine_record_round_trips_and_leaves_no_tmp(tmp_path):
    shelve_path, _ = _make_dirs(tmp_path)
    record_filepath = join(shelve_path, 'jobid-atomic.pkl')

    job_engine._save_engine_record(record_filepath, GOOD_RECORD)

    assert ut.load_cPkl(record_filepath, verbose=False) == GOOD_RECORD
    leftover = sorted(basename(p) for p in glob.glob(join(shelve_path, '*')))
    assert leftover == ['jobid-atomic.pkl']


def test_save_engine_record_keeps_previous_record_when_write_fails(tmp_path, monkeypatch):
    shelve_path, _ = _make_dirs(tmp_path)
    record_filepath = join(shelve_path, 'jobid-atomic.pkl')
    ut.save_cPkl(record_filepath, GOOD_RECORD, verbose=False)
    new_record = dict(GOOD_RECORD, completed=True)

    def _partial_write_then_die(fpath, data, verbose=None, n=None):
        with open(fpath, 'wb') as file_:
            file_.write(pickle.dumps(data, protocol=2)[:5])
        raise IOError('simulated crash mid-write')

    monkeypatch.setattr(job_engine.ut, 'save_cPkl', _partial_write_then_die)

    with pytest.raises(IOError):
        job_engine._save_engine_record(record_filepath, new_record)

    # Destination untouched, partial temp file cleaned up
    assert ut.load_cPkl(record_filepath, verbose=False) == GOOD_RECORD
    leftover = sorted(basename(p) for p in glob.glob(join(shelve_path, '*')))
    assert leftover == ['jobid-atomic.pkl']


def test_save_engine_record_uses_distinct_temp_files(tmp_path, monkeypatch):
    shelve_path, _ = _make_dirs(tmp_path)
    record_filepath = join(shelve_path, 'jobid-atomic.pkl')
    real_save = job_engine.ut.save_cPkl
    seen = []

    def _spy(fpath, data, verbose=None, n=None):
        seen.append(fpath)
        return real_save(fpath, data, verbose=verbose, n=n)

    monkeypatch.setattr(job_engine.ut, 'save_cPkl', _spy)

    job_engine._save_engine_record(record_filepath, GOOD_RECORD)
    job_engine._save_engine_record(record_filepath, GOOD_RECORD)

    assert len(seen) == 2
    # Two writers can never share a temp file
    assert seen[0] != seen[1]
    for tmp in seen:
        assert tmp.startswith(record_filepath + '.')
        assert tmp.endswith('.tmp')
        assert not tmp.endswith('.pkl')


def test_leftover_tmp_is_ignored_by_startup_glob(tmp_path):
    shelve_path, _ = _make_dirs(tmp_path)
    good = join(shelve_path, 'jobid-b.pkl')
    ut.save_cPkl(good, GOOD_RECORD, verbose=False)
    open(join(shelve_path, 'jobid-a.pkl.deadbeef.tmp'), 'wb').close()

    class _StubIbs(object):
        def get_shelves_path(self):
            return shelve_path

    assert job_engine._get_engine_job_paths(_StubIbs()) == [good]
