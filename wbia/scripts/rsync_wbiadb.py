#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CommandLine:
    python -m wbia.scripts.rsync_wbiadb
    python -m wbia.scripts.rsync_wbiadb --dryrun
"""
from __future__ import absolute_import, division, print_function
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def sync_wbiadb(remote_uri, dbname, mode='pull', workdir=None, port=22, dryrun=False):
    """
    syncs an wbiadb without syncing the cache or the chip directory
    (or the top level image directory because it shouldnt exist unlese
    it is an old hots database)
    """
    print('[sync_wbiadb] Syncing')
    print('  * dbname=%r ' % (dbname,))
    print('  * remote_uri=%r' % (remote_uri,))
    print('  * mode=%r' % (mode))
    import wbia

    assert dbname is not None, 'must specify a database name'
    # Excluded temporary and cached data
    exclude_dirs = list(map(ut.ensure_unixslash, wbia.const.EXCLUDE_COPY_REL_DIRS))
    # Specify local workdir
    if workdir is None:
        workdir = wbia.sysres.get_workdir()
    local_uri = ut.ensure_unixslash(workdir)
    if ut.WIN32:
        # fix for mingw rsync
        local_uri = ut.ensure_mingw_drive(local_uri)
    if mode == 'pull':
        # pull remote to local
        remote_src = ut.unixjoin(remote_uri, dbname)
        ut.assert_exists(local_uri)
        ut.rsync(remote_src, local_uri, exclude_dirs, port, dryrun=dryrun)
    elif mode == 'push':
        # push local to remote
        local_src = ut.unixjoin(local_uri, dbname)
        if not dryrun:
            ut.assert_exists(local_src)
        ut.rsync(local_src, remote_uri, exclude_dirs, port, dryrun=dryrun)
        if dryrun:
            ut.assert_exists(local_src)
    else:
        raise AssertionError('unknown mode=%r' % (mode,))


def rsync_ibsdb_main():
    import sys

    default_user = ut.get_user_name()
    # default_db = 'MUGU_Master'
    default_db = None

    # Get positional commandline arguments
    cmdline_varags = ut.get_cmdline_varargs()
    if len(cmdline_varags) > 0 and cmdline_varags[0] == 'rsync':
        # ignore rsync as first command (b/c we are calling from
        # wbia.__main__)
        cmdline_varags = cmdline_varags[1:]
    valid_modes = ['push', 'pull', 'list']

    if len(cmdline_varags) < 1:
        print(
            'Usage: '
            # 'python -m wbia.scripts.rsync_wbiadb'
            'python -m wbia rsync'
            '%s --db <db=%s> --user <user=%s>' % (valid_modes, default_db, default_user,)
        )
        sys.exit(1)

    varargs_dict = dict(enumerate(cmdline_varags))

    mode = varargs_dict.get(0, None)
    default_db = varargs_dict.get(1, None)

    user = ut.get_argval('--user', type_=str, default=default_user)
    port = ut.get_argval('--port', type_=int, default=22)
    dbnames = ut.get_argval(('--db', '--dbs', '--dbname'), type_=str, default=default_db)
    dbnames = ut.smart_cast(dbnames, list)
    workdir = ut.get_argval(
        ('--workdir'), type_=str, default=None, help_='local work dir override'
    )
    dry_run = ut.get_argflag(('--dryrun', '--dry-run', '--dry'))

    assert mode in valid_modes, 'mode=%r must be in %r' % (mode, valid_modes)
    remote_key = ut.get_argval('--remote', type_=str, default='hyrule')
    remote_map = {
        'hyrule': 'hyrule.cs.rpi.edu',
        'pachy': 'pachy.cs.uic.edu',
        'lewa': '41.203.223.178',
    }
    remote_workdir_map = {
        'hyrule': '/raid/work',
        'pachy': '/home/shared_wbia/data/work',
        'lewa': '/data/wbia',
    }
    if ':' in remote_key:
        remote_key_, remote_workdir = remote_key.split(':')
    else:
        remote_key_ = remote_key
        if remote_key not in remote_workdir_map:
            import warnings

            warnings.warn('Workdir not specified for remote')
        remote_workdir = remote_workdir_map.get(remote_key, '')

    remote = remote_map.get(remote_key_, remote_key_)
    remote_uri = user + '@' + remote + ':' + remote_workdir

    if mode == 'list':
        print('remote = %r' % (remote,))
        print('need to list')
        remote_paths = ut.list_remote(remote_uri)
        print('REMOTE LS -- TODO need to get only wbia dirs')
        print('\n'.join(remote_paths))
    elif mode in ['push', 'pull']:
        print('dbnames = {!r}'.format(dbnames))
        for dbname in ut.ProgIter(dbnames, label='sync db'):
            ut.change_term_title('RSYNC IBEISDB %r' % (dbname,))
            sync_wbiadb(remote_uri, dbname, mode, workdir, port, dry_run)


if __name__ == '__main__':
    """
    CommandLine:
        ib
        wbia rsync push
        wbia rsync pull --db MUGU_Master
        wbia rsync pull --db GIRM_MUGU_20
        wbia rsync pull --db PZ_MUGU_ALL
        wbia rsync push --db MUGU_Master  --user joncrall --dryrun

        mv "NNP_Master3_nids=arr((3)wjybfvpk)_1" NNP_Master3_nids=arr__3_wjybfvpk__1

        wbia rsync pull --db NNP_Master3_nids=arr__3_wjybfvpk__1 --user jonc  --remote pachy --dryrun
        wbia rsync pull --db NNP_Master3_nids=arr__3_wjybfvpk__1 --user jonc  --remote pachy
        wbia rsync pull --db NNP_Master3 --user jonc --remote pachy
        wbia rsync pull --db testdb3 --user joncrall --remote hyrule
        wbia rsync pull --db NNP_MasterGIRM_core --user jonc --remote pachy

        #wbia rsync push --db lewa_grevys --user joncrall --remote hyrule --port 1022 --workdir=/data/wbia --dryrun
        wbia rsync pull --db lewa_grevys --user jonathan --remote lewa --port 1022 --dryrun

        wbia rsync push --db ELEPH_Master --user jonc --remote pachy --workdir=/raid/work2/Turk --dryrun
        wbia rsync push --db ELPH_Master --user jonc --remote pachy --workdir=/raid/work2/Turk

        wbia rsync pull --db PZ_ViewPoints --user joncrall --remote hyrule --dryrun

        wbia rsync push --db RotanTurtles,GZ_Master1,humpbacks_fb,PZ_Master1,PZ_MTEST --user jon.crall --remote aretha:data/wbia

        wbia rsync push --db PZ_Master1 --user joncrall --remote lev
        wbia rsync push --db GZ_Master1 --user joncrall --remote lev
        wbia rsync push --db NNP_MasterGIRM_core --user joncrall --remote lev --dryrun
        wbia rsync push --db PZ_PB_RF_TRAIN --user joncrall --remote lev --dryrun
        wbia rsync push --db WS_ALL --user joncrall --remote lev --dryrun
        wbia rsync push --db humpbacks_fb --user joncrall --remote lev

        wbia rsync pull --db GZ_Master1 --user joncrall --remote hyrule

        wbia rsync pull --db WS_ALL --user joncrall --remote hyrule --dryrun

        wbia rsync pull --db PZ_PB_RF_TRAIN --user joncrall --remote hyrule --dryrun
        wbia rsync pull --db PZ_Master1 --user joncrall --remote lev

        wbia rsync push --db lynx2 --user joncrall --remote lev --dryrun

        wbia rsync push --user joncrall --remote lev --db Oxford --dryrun


        stty -echo; ssh jonc@pachy.cs.uic.edu sudo -v; stty echo
        rsync -avhzP -e "ssh -p 22" --rsync-path="sudo rsync" jonc@pachy.cs.uic.edu:/home/wbia-repos/snow-leopards /raid/raw_rsync
        rsync -avhzP -e "ssh -p 22" jonc@pachy.cs.uic.edu:snow-leopards /raid/raw_rsync
        rsync -avhzP -e "ssh -p 22" jonc@pachy.cs.uic.edu:iberian-lynx /raid/raw_rsync
        rsync -avhzP -e "ssh -p 22" --rsync-path="sudo rsync" jonc@pachy.cs.uic.edu:/home/wbia-repos/african-dogs /raid/raw_rsync

        # make sure group read bits are set
        ssh -t jonc@pachy.cs.uic.edu "sudo chown -R apache:wbia /home/wbia-repos/"
        ssh -t jonc@pachy.cs.uic.edu "sudo chmod -R g+r /home/wbia-repos"
        rsync -avhzP -e "ssh -p 22" jonc@pachy.cs.uic.edu:/home/wbia-repos/african-dogs /raid/raw_rsync
        rsync -avhzP -e "ssh -p 22" joncrall@hyrule.cs.rpi.edu/raid/raw_rsync/iberian-lynx .
        rsync -avhzP joncrall@hyrule.cs.rpi.edu:/raid/raw_rsync/iberian-lynx .

        wbia rsync pull --db humpbacks --user joncrall --remote lev:/home/zach/data/IBEIS/ --dryrun
        wbia rsync pull --db humpbacks --user joncrall --remote lev:/home/zach/data/IBEIS/

        wbia rsync pull --db humpbacks_fb --user joncrall --remote lev:/media/hdd/zach/data/IBEIS/

        /home/zach/data/IBEIS/humpbacks_fb

        wbia rsync pull --db seaturtles2 --user 'ubuntu' --remote drewami:/data/wbia

        wbia rsync pull --db testdb3 --user joncrall --remote hyrule

    Fix Patchy
        pachy
        cd /home/wbia-repos
        sudo chmod -R g+r *


    Feasibility Testing Example:

        # --- GET DATA ---
        ssh -t jonc@pachy.cs.uic.edu "sudo chmod -R g+r /home/wbia-repos"
        rsync -avhzP jonc@pachy.cs.uic.edu:/home/wbia-repos/african-dogs /raid/raw_rsync
        rsync -avhzP drewami:turtles .


    WildDog Example:

        # --- GET DATA ---
        # make sure group read bits are set
        ssh -t jonc@pachy.cs.uic.edu "sudo chown -R apache:wbia /home/wbia-repos/"
        ssh -t jonc@pachy.cs.uic.edu "sudo chmod -R g+r /home/wbia-repos"
        rsync -avhzP jonc@pachy.cs.uic.edu:/home/wbia-repos/african-dogs /raid/raw_rsync

        # --- GET DATA ---
        # Get the data via rsync, pydio. (I always have issues doing this with
        # rsync on pachy, so I usually just do it manually)

        rsync -avhzP <user>@<host>:<remotedir>  <path-to-raw-imgs>

        # --- RUN INGEST SCRIPT ---
        # May have to massage folder names things to make everything work. Can
        # also specify fmtkey to use the python parse module to find the name
        # within the folder names.
        python -m wbia --tf ingest_rawdata --db <new-wbia-db-name> --imgdir <path-to-raw-imgs> --ingest-type=named_folders --species=<optional> --fmtkey=<optional>

        # --- OPEN DATABASE / FIX PROBLEMS ---
        wbia --db <new-wbia-db-name>

        # You will probably need to fix some bounding boxes.

        # --- LAUNCH IPYTHON NOTEBOOK ---
        # Then click Dev -> Launch IPython Notebook and run it
        # OR RUN
        wbia --tf autogen_ipynb --db <new-wbia-db-name> --ipynb


        Here is what I did for wild dogs
        # --- GET DATA ---
        # Download raw data to /raid/raw_rsync/african-dogs
        rsync -avhzP jonc@pachy.cs.uic.edu:/home/wbia-repos/african-dogs /raid/raw_rsync

        # --- RUN INGEST SCRIPT ---
        python -m wbia --tf ingest_rawdata --db wd_peter2 --imgdir /raid/raw_rsync/african-dogs --ingest-type=named_folders --species=wild_dog --fmtkey='African Wild Dog: {name}'

        # --- OPEN DATABASE / FIX PROBLEMS ---
        wbia --db wd_peter2
        # Fixed some bounding boxes

        # --- LAUNCH IPYTHON NOTEBOOK ---
        # I actually made two notebooks for this species to account for timedeltas

        # The first is the default notebook
        wbia --tf autogen_ipynb --db wd_peter --ipynb

        # The second removes images without timestamps and annotations that are too close together in time
        wbia --tf autogen_ipynb --db wd_peter --ipynb -t default:is_known=True,min_timedelta=3600,require_timestamp=True,min_pername=2

        # I then click download as html in the notebook. Although I'm sure there is a way to automate this

    """
    rsync_ibsdb_main()
