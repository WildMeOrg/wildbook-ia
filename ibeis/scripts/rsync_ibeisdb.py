#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CommandLine:
    python -m ibeis.scripts.rsync_ibeisdb
    python -m ibeis.scripts.rsync_ibeisdb --dryrun
"""
from __future__ import absolute_import, division, print_function
import utool as ut


def rsync(src_uri, dst_uri, exclude_dirs=[], port=22, dryrun=False):
    """
    General function to push or pull a directory from a remote server to a local path

    References:
        http://www.tecmint.com/rsync-local-remote-file-synchronization-commands/
        http://serverfault.com/questions/219013/showing-total-progress-in-rsync-is-it-possible

    Notes (rsync commandline options):
        rsync [OPTION]... SRC [SRC]... DEST
        -v : verbose
        -r : copies data recursively (but dont preserve timestamps and
                permission while transferring data
        -a : archive mode, allows recursive copying and preserves symlinks,
                permissions, user and group ownerships, and timestamps
        -z : compress file data
        -i, --itemize-changes       output a change-summary for all updates
        -s, --protect-args :        no space-splitting; only wildcard special-chars
        -h : human-readable, output numbers in a human-readable format
        -P                          same as --partial --progress
    """
    rsync_exe = 'rsync'
    rsync_options = '-avhzP'
    #rsync_options += ' --port=%d' % (port,)
    rsync_options += ' -e "ssh -p %d"' % (port,)
    if len(exclude_dirs) > 0:
        exclude_tup = ['--exclude ' + dir_ for dir_ in exclude_dirs]
        exclude_opts = ' '.join(exclude_tup)
        rsync_options += ' ' + exclude_opts

    cmdtuple = (rsync_exe, rsync_options, src_uri, dst_uri)
    cmdstr = ' '.join(cmdtuple)
    print('[rsync] src_uri = %r ' % (src_uri,))
    print('[rsync] dst_uri = %r ' % (dst_uri,))
    print('[rsync] cmdstr = %r' % cmdstr)
    print(cmdstr)

    #if not dryrun:
    ut.cmd(cmdstr, dryrun=dryrun)


def sync_ibeisdb(remote_uri, dbname, mode='pull', workdir=None, port=22, dryrun=False):
    """
    syncs an ibeisdb without syncing the cache or the chip directory
    (or the top level image directory because it shouldnt exist unless it is an
    old hots database)
    """
    print('[sync_ibeisdb] Syncing')
    print('  * dbname=%r ' % (dbname,))
    print('  * remote_uri=%r' % (remote_uri,))
    print('  * mode=%r' % (mode))
    import ibeis
    # Excluded temporary and cached data
    exclude_dirs = list(map(ut.ensure_unixslash, ibeis.const.EXCLUDE_COPY_REL_DIRS))
    # Specify local workdir
    if workdir is None:
        workdir = ibeis.sysres.get_workdir()
    local_uri = ut.ensure_unixslash(workdir)
    if ut.WIN32:
        # fix for mingw rsync
        local_uri = ut.ensure_mingw_drive(local_uri)
    if mode == 'pull':
        # pull remote to local
        remote_src = ut.unixjoin(remote_uri, dbname)
        ut.assert_exists(local_uri)
        rsync(remote_src, local_uri, exclude_dirs, port, dryrun=dryrun)
    elif mode == 'push':
        # push local to remote
        local_src = ut.unixjoin(local_uri, dbname)
        if not dryrun:
            ut.assert_exists(local_src)
        rsync(local_src, remote_uri, exclude_dirs, port, dryrun=dryrun)
        if dryrun:
            ut.assert_exists(local_src)
    else:
        raise AssertionError('unknown mode=%r' % (mode,))


def rsync_ibsdb_main():
    import sys
    default_user = ut.get_user_name()
    default_db = 'MUGU_Master'
    if len(sys.argv) < 2:
        print('Usage: '
              'python -m ibeis.scripts.rsync_ibeisdb'
              '[push, pull] --db <db=%s> --user <user=%s>' %
              (default_db, default_user,))
        sys.exit(1)
    user = ut.get_argval('--user', type_=str, default=default_user)
    port = ut.get_argval('--port', type_=int, default=22)
    dbname = ut.get_argval(('--db', '--dbname'), type_=str, default=default_db)
    workdir = ut.get_argval(('--workdir', '--dbname'), type_=str, default=None,
                            help_='local work dir override')
    dry_run = ut.get_argflag(('--dryrun', '--dry-run', '--dry'))
    mode = sys.argv[1]

    assert mode in ['push', 'pull'], 'mode=%r must be push or pull' % (mode,)
    remote_key = ut.get_argval('--remote', type_=str, default='hyrule')
    remote_map = {
        'hyrule': 'hyrule.cs.rpi.edu',
        'pachy': 'pachy.cs.uic.edu',
        'lewa': '41.203.223.178',
        'lev': 'lev.cs.rpi.edu',
    }
    remote_workdir_map = {
        'hyrule': '/raid/work',
        'pachy': '/home/shared_ibeis/data/work',
        'lewa': '/data/ibeis',
        'lev': '/media/hdd/work',
    }
    if ':' in remote_key:
        remote_key_, remote_workdir = remote_key.split(':')
    else:
        remote_key_ = remote_key
        remote_workdir = remote_workdir_map.get(remote_key, '')

    remote = remote_map.get(remote_key_, remote_key_)
    remote_uri = user + '@' + remote + ':' + remote_workdir

    ut.change_term_title('RSYNC IBEISDB %r' % (dbname,))
    sync_ibeisdb(remote_uri, dbname, mode, workdir, port, dry_run)


if __name__ == '__main__':
    """
    CommandLine:
        ib
        python -m ibeis.scripts.rsync_ibeisdb push
        python -m ibeis.scripts.rsync_ibeisdb pull --db MUGU_Master
        python -m ibeis.scripts.rsync_ibeisdb pull --db GIRM_MUGU_20
        python -m ibeis.scripts.rsync_ibeisdb pull --db PZ_MUGU_ALL
        python -m ibeis.scripts.rsync_ibeisdb push --db MUGU_Master  --user joncrall --dryrun

        mv "NNP_Master3_nids=arr((3)wjybfvpk)_1" NNP_Master3_nids=arr__3_wjybfvpk__1

        python -m ibeis.scripts.rsync_ibeisdb pull --db NNP_Master3_nids=arr__3_wjybfvpk__1 --user jonc  --remote pachy --dryrun
        python -m ibeis.scripts.rsync_ibeisdb pull --db NNP_Master3_nids=arr__3_wjybfvpk__1 --user jonc  --remote pachy
        python -m ibeis.scripts.rsync_ibeisdb pull --db NNP_Master3 --user jonc --remote pachy
        python -m ibeis.scripts.rsync_ibeisdb pull --db testdb3 --user joncrall --remote hyrule
        python -m ibeis.scripts.rsync_ibeisdb pull --db NNP_MasterGIRM_core --user jonc --remote pachy

        #python -m ibeis.scripts.rsync_ibeisdb push --db lewa_grevys --user joncrall --remote hyrule --port 1022 --workdir=/data/ibeis --dryrun
        python -m ibeis.scripts.rsync_ibeisdb pull --db lewa_grevys --user jonathan --remote lewa --port 1022 --dryrun

        python -m ibeis.scripts.rsync_ibeisdb push --db ELEPH_Master --user jonc --remote pachy --workdir=/raid/work2/Turk --dryrun
        python -m ibeis.scripts.rsync_ibeisdb push --db ELPH_Master --user jonc --remote pachy --workdir=/raid/work2/Turk

        python -m ibeis.scripts.rsync_ibeisdb pull --db PZ_ViewPoints --user joncrall --remote hyrule --dryrun

        python -m ibeis.scripts.rsync_ibeisdb push --db PZ_Master1 --user joncrall --remote lev --dryrun


        stty -echo; ssh jonc@pachy.cs.uic.edu sudo -v; stty echo
        rsync -avhzP -e "ssh -p 22" --rsync-path="sudo rsync" jonc@pachy.cs.uic.edu:/home/ibeis-repos/snow-leopards /raid/raw_rsync
        rsync -avhzP -e "ssh -p 22" jonc@pachy.cs.uic.edu:snow-leopards /raid/raw_rsync
        rsync -avhzP -e "ssh -p 22" jonc@pachy.cs.uic.edu:iberian-lynx /raid/raw_rsync

        # make sure group read bits are set
        ssh -t jonc@pachy.cs.uic.edu "sudo chmod -R g+r /home/ibeis-repos"
        rsync -avhzP -e "ssh -p 22" jonc@pachy.cs.uic.edu:/home/ibeis-repos/african-dogs /raid/raw_rsync
        rsync -avhzP -e "ssh -p 22" joncrall@hyrule.cs.rpi.edu/raid/raw_rsync/iberian-lynx .
        rsync -avhzP joncrall@hyrule.cs.rpi.edu:/raid/raw_rsync/iberian-lynx .

        python -m ibeis.scripts.rsync_ibeisdb pull --db humpbacks --user joncrall --remote lev:/home/zach/data/IBEIS/ --dryrun
        python -m ibeis.scripts.rsync_ibeisdb pull --db humpbacks --user joncrall --remote lev:/home/zach/data/IBEIS/
    """
    rsync_ibsdb_main()
