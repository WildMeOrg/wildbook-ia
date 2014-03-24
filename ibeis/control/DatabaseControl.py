from __future__ import division, print_function
import io
import os
import numpy as np
import sqlite3 as lite


def _NUMPY_TO_SQLITE3(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return buffer(out.read())


def _SQLITE3_TO_NUMPY(blob):
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)

     
class DatabaseControl(object):
    def __init__(db, database_path, database_file='database.sqlite3'):
        '''
            SQLite3 Documentation: http://www.sqlite.org/docs.html

            SQL INSERT: http://www.w3schools.com/sql/sql_insert.asp
            SQL UPDATE: http://www.w3schools.com/sql/sql_update.asp
            SQL SELECT: http://www.w3schools.com/sql/sql_select.asp
            SQL DELETE: http://www.w3schools.com/sql/sql_delete.asp
            
            --------------------------------------------------------------------

            Init the SQLite3 database connection and the query execution object.
            If the database does not exist, it will be automatically created
            upon this object's instantiation.
        '''

        db.database_path = database_path
        db.database_file = database_file

        db.connection = lite.connect(os.path.join(database_path, database_file), 
            detect_types=lite.PARSE_DECLTYPES) # Need this param for numpy arrays

        db.querier = db.connection.cursor()
        
        # Converts numpy array object to sqlite3 blob when insert querying
        lite.register_adapter(np.ndarray, _NUMPY_TO_SQLITE3)

        # Converts sqlite3 blob to numpy array object when select querying
        lite.register_converter('NUMPY', _SQLITE3_TO_NUMPY)


    def schema(db, table, schemas):
        '''
            schemas    - dictionary of table columns
                {
                    column_1_name: column_1_type,
                    column_2_name: column_2_type,
                    ...
                    column_N_name: column_N_type,
                }

            column_n_name - string name of column heading
            column_n_type - NULL | INTEGER | REAL | TEXT | BLOB
                The column type can be appended with ' PRIMARY KEY' to indicate
                the unique id for the table.  It can also specify a default
                value for the column with ' DEFAULT [VALUE]'.  It can also
                specify ' NOT NULL' to indicate the column cannot be empty.

            The table will only be created if it does not exist.  Therefore,
            this can be done on every table without the fear of deleting old
            data.

            TODO: Add handling for column addition between software versions.
            Column deletions will not be removed from the database schema.
        '''

        # Technically insecure call, but all entries are statically inputted by
        # the database's owner, who could delete or alter the entire database
        # anyway.
        sql = 'CREATE TABLE IF NOT EXISTS ' + table + '('
        for column_name, column_type in schemas.items():
            sql += column_name + ' ' + column_type + ', '
        sql = sql[:-2] + ')'
        db.query(sql, [])

    def query(db, sql, parameters, auto_commit=False):
        '''
            sql - parameterized SQL query string.
                Parameterized prevents SQL injection attacks by using an ordered
                representation ( ? ) or by using an ordered, text representation
                name ( :value )

            parameters - list of values or a dictionary of representations and
                         corresponding values
                * Ordered Representation -
                    List of values in the order the question marks appear in the
                    sql string
                * Unordered Representation -
                    Dictionary of (text representation name -> value) in an
                    arbirtary order that will be filled into the cooresponging
                    slots of the sql string
        '''
        status = 0
        try:
            status = db.querier.execute(sql, parameters)
            
            if auto_commit:
                db.commit()
        except Exception as e:
            status = 1

        return status

    def result(db, all=False):
        return db.querier.fetchone()

    def results(db, all=False):
        while True:
            result = db.result()
            if not result:
                break
            yield result[0]

    def commit(db, error_text="Generic database error", query_results=[]):
        '''
            Commits staged changes to the database and saves the binary
            representation of the database to disk.  All staged changes can be
            commited one at a time or after a batch - which allows for batch
            error handling without comprimising the integrity of the database.
        '''

        if sum(query_results) > 0:
            raise ValueError(error_text)
        else:
            db.connection.commit()

    def dump(db, dump_path=None, dump_file='database.dump.txt'):
        '''
            Same output as shell command below
            > sqlite3 database.sqlite3 .dump > database.dump.txt

            This saves the current database schema structure and data into a
            text dump.  The entire database can be recovered from this dump
            file.  The default will store a dump parallel to the current
            database file.
        '''
        db.commit()
        if dump_path is None:
            dump_path = db.database_path
        dump = open(os.path.join(dump_path, dump_file), 'w')
        for line in db.connection.iterdump():
            dump.write('%s\n' % line)
        dump.close()


if __name__ == '__main__':

    try:
        os.remove('temp.sqlite3')
    except Exception as e:
        print(1)

    db = DatabaseControl('.', database_file='temp.sqlite3')

    db.schema('temp',	{
        'temp_id':      'INTEGER PRIMARY KEY',
        'temp_hash':    'NUMPY',
    })    

    # list of 10,000 chips with 3,000 features apeice. 
    table_list = [np.empty((3 * 10^3, 128), dtype=np.uint8) for i in xrange(10000)]
    for table in iter(table_list):
        db.query('INSERT INTO temp (temp_hash) VALUES (?)', [table])
    
    db.commit()
    
    db.query('SELECT temp_hash FROM temp',[])
    for result in db.results():
        pass

    db.dump(dump_file='temp.dump.txt')

