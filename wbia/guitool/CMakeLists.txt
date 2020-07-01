add_cython_target(_api_tree_node_cython "_api_tree_node_cython.pyx" C
                  OUTPUT_VAR cython_out_fpath)
message(STATUS "!!! cython_out_fpath = ${cython_out_fpath}")
add_library(_api_tree_node_cython MODULE ${cython_out_fpath})
python_extension_module(_api_tree_node_cython)
install(TARGETS _api_tree_node_cython LIBRARY DESTINATION guitool/lib)
