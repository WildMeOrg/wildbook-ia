#!/usr/bin/env python
"""
CommandLine:
    ib
    python _scripts/rsync_ibeisdb.py
    python _scripts/rsync_ibeisdb.py --dryrun
"""
import utool as ut


DRY_RUN = ut.get_argflag('--dryrun')


def rsync(src_uri, dst_uri, exclude_dirs=[], dryrun=DRY_RUN):
    """
    rsync [OPTION]... SRC [SRC]... DEST

    General function to push or pull a directory from a remote server to a local path

    References:
        http://www.tecmint.com/rsync-local-remote-file-synchronization-commands/
        http://serverfault.com/questions/219013/showing-total-progress-in-rsync-is-it-possible
    """
    rsync_exe = 'rsync'
    #-v : verbose
    #-r : copies data recursively (but dont preserve timestamps and permission while transferring data
    #-a : archive mode, allows recursive copying and preserves symlinks, permissions, user and group ownerships, and timestamps
    #-z : compress file data
    #-h : human-readable, output numbers in a human-readable format
    #-P                          same as --partial --progress
    rsync_options = '-vazhP'
    if len(exclude_dirs) > 0:
        exclude_tup = ['--exclude ' + dir_ for dir_ in exclude_dirs]
        exclude_opts = ' '.join(exclude_tup)
        rsync_options += ' ' + exclude_opts

    cmdtuple = (rsync_exe, rsync_options, src_uri, dst_uri)
    cmdstr = ' '.join(cmdtuple)
    print('src_uri = %r ' % (src_uri,))
    print('dst_uri = %r ' % (dst_uri,))
    print('cmdstr = %r' % cmdstr)

    if not dryrun:
        ut.cmd(cmdstr)


def sync_ibeisdb(remote_uri, dbname, mode='pull'):
    """
    syncs an ibeisdb without syncing the cache or the chip directory
    (or the top level image directory because it shouldnt exist unless it is an
    old hots database)
    """
    print('[sync_ibeisdb] Syncing dbname=%r with remote_uri=%r mode=%r' % (dbname, remote_uri, mode))
    import ibeis
    # localworkdir
    exclude_dirs = [
        ut.ensure_unixslash(ibeis.const.REL_PATHS.chips),
        ut.ensure_unixslash(ibeis.const.REL_PATHS.cache),
        ut.ensure_unixslash(ibeis.const.REL_PATHS.backups),
        ut.ensure_unixslash(ibeis.const.REL_PATHS.figures),
        #'_ibsdb/_ibeis_cache',
        #'_ibsdb/chips',
        './images',  # the hotspotter images dir
    ]
    local_uri = ut.ensure_unixslash(ibeis.sysres.get_workdir())
    if ut.WIN32:
        # fix for mingw rsync
        local_uri = ut.ensure_mingw_drive(local_uri)
    if mode == 'pull':
        # pull remote to local
        rsync(ut.unixjoin(remote_uri, dbname), local_uri, exclude_dirs)
    elif mode == 'push':
        # push local to remote
        rsync(ut.unixjoin(local_uri, dbname), remote_uri, exclude_dirs)
    else:
        raise AssertionError('unknown mode=%r' % (mode,))


#def sync_pz_mugu_19(mode='pull'):
#    remote_uri = 'joncrall@hyrule.cs.rpi.edu:/raid/work'
#    dbname = 'PZ_MUGU_19'
#    sync_ibeisdb(remote_uri, dbname, mode)


if __name__ == '__main__':
    """
    CommandLine:
        ib
        python _scripts/rsync_ibeisdb.py push
        python _scripts/rsync_ibeisdb.py pull --db MUGU_Master
        python _scripts/rsync_ibeisdb.py pull --db GIRM_MUGU_20
        python _scripts/rsync_ibeisdb.py pull --db PZ_MUGU_ALL
        python _scripts/rsync_ibeisdb.py push --db MUGU_Master  --user joncrall --dryrun
    """
    import sys
    default_user = ut.get_user_name()
    default_db = 'MUGU_Master'
    if len(sys.argv) < 2:
        print('Usage: rsync_ibeisdb.py [push, pull] --db <db=%s> --user <user=%s>' % (default_db, default_user,))
        sys.exit(1)
    user = ut.get_argval('--user', type_=str, default=default_user)
    dbname = ut.get_argval(('--db', '--dbname'), type_=str, default=default_db)
    mode = sys.argv[1]

    assert mode in ['push', 'pull'], 'mode=%r must be push or pull' % (mode,)
    remote_uri = user + '@hyrule.cs.rpi.edu:/raid/work'
    sync_ibeisdb(remote_uri, dbname, mode)
