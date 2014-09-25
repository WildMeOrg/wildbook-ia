#!/usr/bin/env python2.7
"""
Runs IBIES demo
"""
from __future__ import absolute_import, division, print_function
import multiprocessing
import ibeis
import utool
from os.path import join


if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32

    # Create a directory for the demo database
    workdir = ibeis.get_workdir()
    demodir = join(workdir, 'demo')

    if utool.get_argval('--reset'):
        # Remove the previous demo if it exists
        utool.delete(demodir)

    # Start a new database there
    main_locals = ibeis.main(dbdir=demodir)

    # Get a handle to the GUIBackend Control
    back = main_locals['back']

    # Get a directory with some images in it

    testurl = 'https://www.dropbox.com/s/s4gkjyxjgghr18c/testdata_detect.zip'
    testdir = utool.grab_zipped_url(testurl)

    execstr = ibeis.main_loop(main_locals)
    exec(execstr)

    script = """
    back.import_images_from_dir(testdir)
    back.detect_grevys_quick()
    back.compute_encounters()
    """

    #execstr = ibeis.main_loop(main_locals)
    #exec(execstr)
