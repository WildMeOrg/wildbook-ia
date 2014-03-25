#!usr/bin/env python
from __future__ import division, print_function
from ibs.control import SQLite3Control
import os
import numpy as np

if __name__ == '__main__':
    try:
        os.remove('temp.sqlite3')
    except Exception as e:
        print(1)

    db = SQLite3Control.SQLite3Control('.', database_file='temp.sqlite3')

    db.schema('temp',    {
        'temp_id':      'INTEGER PRIMARY KEY',
        'temp_hash':    'NUMPY',
    })

    # list of 10,000 chips with 3,000 features apeice.
    table_list = [np.empty((3 * 1e3, 128), dtype=np.uint8) for i in xrange(10000)]
    for table in iter(table_list):
        db.query('INSERT INTO temp (temp_hash) VALUES (?)', [table])

    db.commit()

    db.query('SELECT temp_hash FROM temp', [])
    for result in db.results():
        pass

    db.dump(dump_file='temp.dump.txt')
