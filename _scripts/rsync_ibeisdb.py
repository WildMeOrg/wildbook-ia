#/usr/env/bin python
"""
CommandLine:
    ib
    python _scripts/rsync_ibeisdb.py
    python _scripts/rsync_ibeisdb.py --dryrun
"""
import utool as ut
import os.path


DRY_RUN = ut.get_argflag('--dryrun')


def rsync_pull(host, remote_dpath, local_dpath, dname, exclude_dirs=[], dryrun=DRY_RUN):
    """
    General function to pull a directory from a remote server to a local path

    References:
        http://www.tecmint.com/rsync-local-remote-file-synchronization-commands/
        http://serverfault.com/questions/219013/showing-total-progress-in-rsync-is-it-possible
    """
    remote_path = os.path.join(remote_dpath, dname).replace('\\', '/')
    local_uri = os.path.join(local_dpath).replace('\\', '/')
    remote_uri = host + ':' + remote_path

    if ut.WIN32:
        # fix for mingw rsync
        win32_drive, _path = os.path.splitdrive(local_uri)
        mingw_drive = '/' + win32_drive[:-1].lower()
        local_uri = mingw_drive + _path

    rsync_exe = 'rsync'
    #-v : verbose
    #-r : copies data recursively (but dont preserve timestamps and permission while transferring data
    #-a : archive mode, archive mode allows copying files recursively and it also preserves symlinks, permissions, user and group ownerships, and timestamps
    #-z : compress file data
    #-h : human-readable, output numbers in a human-readable format
    #-P                          same as --partial --progress
    rsync_options = '-vrazhP'
    if len(exclude_dirs) > 0:
        exclude_tup = ['--exclude ' + dir_ for dir_ in exclude_dirs]
        exclude_opts = ' '.join(exclude_tup)
        rsync_options += ' ' + exclude_opts

    cmdtuple = (rsync_exe, rsync_options, remote_uri, local_uri)
    cmdstr = ' '.join(cmdtuple)
    print('remote_uri = %r ' % (remote_uri,))
    print('local_uri  = %r ' % (local_uri,))
    print('cmdstr = %r' % cmdstr)

    if not dryrun:
        ut.cmd(cmdstr)


def rsync_pull_ibeisdb(host, dbname):
    """
    syncs an ibeisdb without syncing the cache or the chip directory
    (or the top level image directory because it shouldnt exist unless it is an
    old hots database)
    """
    import ibeis
    # remote workdir
    remote_dpath = '/raid/work'
    # localworkdir
    local_dpath = ibeis.sysres.get_workdir().replace('\\', '/')
    exclude_dirs = [
        '_ibsdb/_ibeis_cache',
        '_ibsdb/chips',
        './images',
    ]
    dname = dbname
    rsync_pull(host, remote_dpath, local_dpath, dname, exclude_dirs)
    #--exclude 'dir1'


def pull_pz_mugu_19():
    host = 'joncrall@hyrule.cs.rpi.edu'
    dbname = 'PZ_MUGU_19'
    rsync_pull_ibeisdb(host, dbname)


if __name__ == '__main__':
    pull_pz_mugu_19()
