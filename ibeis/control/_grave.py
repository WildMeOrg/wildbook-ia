## Sanity check
#num_results = len(result_list)
#if num_results != 0 and num_results != num_params:
#    raise lite.Error('num_params=%r != num_results=%r'
#                     % (num_params, num_results))
# Transactions halve query time
# list comprehension cuts time by 10x
#result_list = [_unpacker(results_) for results_ in result_list]


            #if verbose:
                #caller_name = utool.util_dbg.get_caller_name()
                #print('[sql.commit] caller_name=%r' % caller_name)


#printDBG('<ACTUAL COMMIT>')
#printDBG('</ACTUAL COMMIT>')


#if verbose:
#    caller_name = utool.util_dbg.get_caller_name()
#    print('[sql.result] caller_name=%r' % caller_name)
