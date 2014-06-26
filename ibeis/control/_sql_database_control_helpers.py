from __future__ import absolute_import, division, print_function
# Python
# Tools
import utool
from . import __SQLITE3__ as lite


def default_decorator(func):
    return func
    #return utool.indent_func('[sql.' + func.func_name + ']')(func)

# =======================
# Helper Functions
# =======================
DEBUG = False
VERBOSE = utool.VERBOSE
PRINT_SQL = utool.get_flag('--print-sql')
AUTODUMP = utool.get_flag('--auto-dump')
QUIET = utool.QUIET or utool.get_flag('--quiet-sql')
PRINT_SQL = utool.get_flag('--print-sql')


def _executor(executor, opeartion, params):
    """ HELPER: Send command to SQL (all other results are invalided)
    Execute an SQL Command
    """
    executor.execute(opeartion, params)


def _results_gen(executor, verbose=VERBOSE, get_last_id=False):
    """ HELPER - Returns as many results as there are.
    Careful. Overwrites the results once you call it.
    Basically: Dont call this twice.
    """
    if get_last_id:
        # The sqlite3_last_insert_rowid(D) interface returns the
        # <b> rowid of the most recent successful INSERT </b>
        # into a rowid table in D
        _executor(executor, 'SELECT last_insert_rowid()', ())
    # Wraping fetchone in a generator for some pretty tight calls.
    while True:
        result = executor.fetchone()
        if not result:
            raise StopIteration()
        else:
            # Results are always returned wraped in a tuple
            result = result[0] if len(result) == 1 else result
            #if get_last_id and result == 0:
            #    result = None
            yield result


def _unpacker(results_):
    """ HELPER: Unpacks results if unpack_scalars is True """
    results = None if len(results_) == 0 else results_[0]
    assert len(results_) < 2, 'throwing away results! { %r }' % (results_)
    return results


# =======================
# SQL Context Class
# =======================


class SQLExecutionContext(object):
    """ A good with context to use around direct sql calls
    """
    def __init__(context, db, operation, num_params=None, auto_commit=True,
                 start_transaction=False):
        context.auto_commit = auto_commit
        context.db = db  # Reference to sqldb
        context.operation = operation
        context.num_params = num_params
        context.start_transaction = start_transaction
        #context.__dict__.update(locals())  # Too mystic?
        context.operation_type = get_operation_type(operation)  # Parse the optype

    def __enter__(context):
        """ Checks to see if the operating will change the database """
        #utool.printif(lambda: '[sql] Callers: ' + utool.get_caller_name(range(3, 6)), DEBUG)
        if context.num_params is None:
            context.operation_label = ('[sql] execute num_params=%d optype=%s: '
                                       % (context.num_params, context.operation_type))
        else:
            context.operation_label = '[sql] executeone optype=%s: ' % (context.operation_type)
        # Start SQL Transaction
        if context.start_transaction:
            _executor(context.db.executor, 'BEGIN', ())
        if PRINT_SQL:
            print(context.operation_label)
        # Comment out timeing code
        #if not QUIET:
        #    context.tt = utool.tic(context.operation_label)
        return context

    # --- with SQLExecutionContext: statment code happens here ---

    def execute_and_generate_results(context, params):
        """ HELPER FOR CONTEXT STATMENT """
        executor = context.db.executor
        operation = context.operation
        try:
            _executor(context.db.executor, operation, params)
            is_insert = context.operation_type.upper().startswith('INSERT')
            return _results_gen(executor, get_last_id=is_insert)
        except lite.IntegrityError:
            raise

    def __exit__(context, type_, value, trace):
        #if not QUIET:
        #    utool.tic(context.tt)
        if trace is None:
            # Commit the transaction
            if context.auto_commit:
                context.db.commit(verbose=False)
        else:
            # An SQLError is a serious offence.
            # Dump on error
            print('[sql] FATAL ERROR IN QUERY')
            context.db.dump()
            # utool.sys.exit(1)


def get_operation_type(operation):
    """
    Parses the operation_type from an SQL operation
    """
    operation = ' '.join(operation.split('\n')).strip()
    operation_type = operation.split(' ')[0].strip()
    if operation_type.startswith('SELECT'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('INSERT'):
        operation_args = utool.str_between(operation, operation_type, '(').strip()
    elif operation_type.startswith('UPDATE'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('DELETE'):
        operation_args = utool.str_between(operation, operation_type, 'FROM').strip()
    elif operation_type.startswith('CREATE'):
        operation_args = utool.str_between(operation, operation_type, '(').strip()
    else:
        operation_args = None
    operation_type += ' ' + operation_args.replace('\n', ' ')
    return operation_type
