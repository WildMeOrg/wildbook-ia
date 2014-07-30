#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool
from six.moves import range
from ibeis.control import __SQLITE3__ as lite
from os.path import join, realpath
import random
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_SQL_REVERT]')


def _val1(i):
    return i * 2


def _val2(i):
    return str(i * 2) + "_string"


def test_query(connection, cur, _type, alter_callback, isolation_level, bound=10, transaction=True, commit=False):
    # Offset bound by 1
    bound += 1
    retval = True

    # Clear test data
    operation = '''
        DELETE FROM test
        WHERE rowid > 0
    '''
    cur.execute(operation, [])

    # Add dummy data and commit it
    for i in range(1, bound):
        operation = '''
            INSERT INTO test
            (
                rowid,
                test_int,
                test_string
            )
            VALUES (NULL, ?, ?)
        '''
        cur.execute(operation, [_val1(i), _val2(i)])
    connection.commit()

    # Begin the transaction
    if transaction:
        cur.execute('BEGIN', ())

    # Alter the data
    try:
        expected = alter_callback(connection, cur, bound)
        print("%s (isolation=%r, trans=%r, commit=%r) Passed Alter" % (_type, isolation_level, transaction, commit))
    except Exception as e:
        retval = False
        expected = None
        print("%s (isolation=%r, trans=%r, commit=%r) Failed Alter: %r" % (_type, isolation_level, transaction, commit, e))

    # Commit change
    if commit:
        connection.commit()

    try:
        # Rollback to previous state
        connection.rollback()

        # Check for consistency
        operation = '''
            SELECT *
            FROM test
            ORDER BY rowid ASC
        '''
        cur.execute(operation, [])
        rows = cur.fetchall()
        indices = []
        # Check for UPDATE and Rollback
        for row in rows:
            i = row[0]
            indices.append(i)
            # Must check extreme case where there are no transactions, no commiting and the isolation is None
            if commit or (isolation_level is None and not transaction and not commit):
                if _type == 'UPDATE' and (row[1] == _val1(i) or row[2] == _val2(i)):
                    raise IOError("ERROR, DATA SHOULD HAVE BEEN COMMITED AND CANNOT ROLLBACK")
            else:
                if row[1] != _val1(i) or row[2] != _val2(i):
                    raise IOError("ERROR, DATA SHOULD HAVE BEEN ROLLED BACK")

        # Check for INSERT / DELETE
        original = 0
        missing = 0
        added = 0
        originals = list(range(1, bound))
        for i in set(originals + indices):
            if i not in indices:
                missing += 1
            elif i in indices and i not in originals:
                added += 1
            else:
                original += 1

        if commit and _type == 'DELETE' and missing != expected:
            raise IOError("ERROR, DATA MISMATCH MISSING %r - %r" % (missing, expected))

        if commit and _type == 'INSERT' and added != expected:
            raise IOError("ERROR, DATA MISMATCH ADDED %r - %r" % (added, expected))

        print("%s (isolation=%r, trans=%r, commit=%r) Passed Rollback" % (_type, isolation_level, transaction, commit))
        print('Original: %r, Missing: %r, Added: %r, Expected: %r' % (original, missing, added, expected))
    except IOError as e:
        retval = False
        print("%s (isolation=%r, trans=%r, commit=%r) Failed Rollback: %r" % (_type, isolation_level, transaction, commit, e))

    print('')
    return retval


def alter_update(connection, cur, bound):
    # Modify Data
    for i in range(1, bound):
        operation = '''
            UPDATE test
            SET test_int=?, test_string=?
            WHERE rowid=?
        '''
        cur.execute(operation, [i * 5, str(i * 5) + "_string", i])

    # Check for change
    operation = '''
        SELECT *
        FROM test
        ORDER BY rowid ASC
    '''
    cur.execute(operation, [])
    rows = cur.fetchall()
    for row in rows:
        i = row[0]
        if row[1] == _val1(i) or row[2] == _val2(i):
            raise IOError("ERROR, DATA SHOULD HAVE BEEN ALTERED")

    return 0


def alter_delete(connection, cur, bound):
    # Get random indices
    randoms = sorted(set([random.randint(1, bound - 1) for x in range( int(bound * 0.10) )]))

    # Modify Data
    for i in randoms:
        operation = '''
            DELETE FROM test
            WHERE rowid=?
        '''
        cur.execute(operation, [i])

    # Check for change
    operation = '''
        SELECT *
        FROM test
        ORDER BY rowid ASC
    '''
    cur.execute(operation, [])
    rows = cur.fetchall()
    for row in rows:
        i = row[0]
        if (i in randoms and row is not None) or (i not in randoms and row is None):
            raise IOError("ERROR, DATA SHOULD HAVE BEEN ALTERED")

    return len(randoms)


def alter_insert(connection, cur, bound):
    added = int(bound * 0.10)

    # Modify Data
    for i in range(bound, bound + added):
        operation = '''
            INSERT INTO test
            (
                rowid,
                test_int,
                test_string
            )
            VALUES (NULL, ?, ?)
        '''
        cur.execute(operation, [_val1(i), _val2(i)])

    # Check for change
    operation = '''
        SELECT *
        FROM test
        ORDER BY rowid ASC
    '''
    cur.execute(operation, [])
    rows = cur.fetchall()
    for row in rows:
        i = row[0]
        if row is None or row[1] != _val1(i) or row[2] != _val2(i):
            raise IOError("ERROR, DATA SHOULD HAVE BEEN ALTERED")

    return added


def TEST_SQL_REVERT(isolation_level=None):
    base = realpath('.')

    # Create SQLITE3 object
    connection = lite.connect(
        join(base, 'test_sql_revert.sqlite3'),
        detect_types=lite.PARSE_DECLTYPES,
        isolation_level=isolation_level
    )
    cur = connection.cursor()

    # Clear the database and drop all current test data
    operation = '''
        DROP TABLE IF EXISTS test
    '''
    cur.execute(operation, [])

    # Create the test database because it will not exist
    operation = '''
        CREATE TABLE IF NOT EXISTS test
        (
            test_id INTEGER PRIMARY KEY,
            test_int INTEGER NOT NULL,
            test_string TEXT
        )
    '''
    cur.execute(operation, [])
    connection.commit()

    status = []
    for trans in [True, False]:
        status.append(test_query(connection, cur, 'UPDATE', alter_update, isolation_level, transaction=trans))
        status.append(test_query(connection, cur, 'UPDATE', alter_update, isolation_level, transaction=trans, commit=True))
        status.append(test_query(connection, cur, 'DELETE', alter_delete, isolation_level, transaction=trans))
        status.append(test_query(connection, cur, 'DELETE', alter_delete, isolation_level, transaction=trans, commit=True))
        status.append(test_query(connection, cur, 'INSERT', alter_insert, isolation_level, transaction=trans))
        status.append(test_query(connection, cur, 'INSERT', alter_insert, isolation_level, transaction=trans, commit=True))

    if not all(status):
        raise Exception("Tests failed for %r" % (isolation_level))
        # This will never happen unless SQLite 3 freaks out

    return locals()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    test_locals = utool.run_test(TEST_SQL_REVERT)
    test_locals = utool.run_test(TEST_SQL_REVERT, isolation_level='DEFERRED')
    test_locals = utool.run_test(TEST_SQL_REVERT, isolation_level='IMMEDIATE')
    test_locals = utool.run_test(TEST_SQL_REVERT, isolation_level='EXCLUSIVE')
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
