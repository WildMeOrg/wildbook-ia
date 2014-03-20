
import os
import sqlite3 as lite


class Database(object):


	def __init__(db, database_path, database_file="database.sqlite3"):
		'''
			Init the SQLite3 database connection and the query execution object.

			If the database does not exist, it will be automatically created upon this object's instantiation.
		'''
		
		db.database_path = database_path
		db.database_file = database_file

		db.connection = lite.connect(os.path.join(database_path, database_file))
		db.querier = connection.cursor()


	def schema(db, table, schemas):
		''' 
			schemas	- dictionary of table columns
				{ 
					column_1_name: column_1_type, 
					column_2_name: column_2_type,
					...  
					column_N_name: column_N_type,
				}

			column_n_name - string name of column heading
			column_n_type - NULL | INTEGER | REAL | TEXT | BLOB
				The column type can be appended with " PRIMARY KEY" to indicate the unique id for the table


			The table will only be created if it does not exist.  Therefore, this can be done on every table without the
			fear of deleting old data.  

			TODO: Add handling for column addition between software versions.  Column deletions will not be removed from the database schema.
		'''

		# Technically insecure call, but all entries are statically inputted by the database's owner, who could delete or alter the entire database anywaya.
		sql = "CREATE TABLE IF NOT EXISTS " + table + "("
		for column_name, column_type in schemas.items():
			sql += column_name + " " + column_type + ", "
		sql += ")"

		db.query(sql, [])


	def query(db, sql, parameters, auto_commit=False):
		'''
			sql - parameterized SQL query string.  
				Parameterized prevents SQL injection attacks by using an ordered representation ( ? ) or 
				by using an ordered, text representation name ( :value )

			parameters - list of values or a dictionary of representations and corresponding values
				Ordered Representation - List of values in the order the question marks appear in the sql string
				Unordered Representation - Dictionary of (text representation name -> value) in an arbirtary order that will 
											be filled into the cooresponging slots of the sql string
		'''
		db.querier.execute(sql, parameters)
		
		if auto_commit:
			db.commit()


	def commit(db):
		'''
			Commits staged changes to the database and saves the binary representation of the database to disk.
			All staged changes can be commited one at a time or after a batch - which allows for batch error handling 
			without comprimising the integrity of the database.
		'''
		db.connection.commit()


	def dump(db, database_path=None, database_file="database.dump.text"):
		'''
			Same output as shell command below
			> sqlite3 database.sqlite3 .dump > database.dump.txt

			This saves the current database schema structure and data into a text dump.  The entire database can be 
			recovered from this dump file.  The default will store a dump parallel to the current database file.
		'''

		if database_path == None:
			database_path = db.database_path

		dump = open(os.path.join(database_path, database_file), 'w')

	    for line in db.connection.iterdump():
	       dump.write('%s\n' % line)

	    dump.close()

