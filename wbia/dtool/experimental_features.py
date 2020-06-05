# -*- coding: utf-8 -*-
# @ut.accepts_scalar_input2(argx_list=[1])
# def get_obj(depc, tablename, root_rowids, config=None, ensure=True):
#     """ Convinience function. Gets data in `tablename` as a list of
#     objects. """
#     print('WARNING EXPERIMENTAL')
#     try:
#         if tablename == depc.root:
#             obj_list = list(depc._root_asobject(root_rowids))
#         else:
#             def make_property_getter(rowid, colname):
#                 def wrapper():
#                     return depc.get_property(
#                         tablename, rowid, colnames=colname, config=config,
#                         ensure=ensure)
#                 return wrapper
#             colnames = depc[tablename].data_colnames
#             obj_list = [
#                 ut.LazyDict({colname: make_property_getter(rowid, colname)
#                              for colname in colnames})
#                 for rowid in root_rowids
#             ]
#         return obj_list
#         # data_list = depc.get_property(tablename, root_rowids, config)
#         # # TODO: lazy dict
#         # return [dict(zip(colnames, data)) for data in data_list]
#     except Exception as ex:
#         ut.printex(ex, 'error in getobj', keys=['tablename', 'root_rowids',
#                                                 'colnames'])
#         raise


#     def root_asobject(aid_list):
#         """ Convinience for writing preproc funcs """
#         for aid in aid_list:
#             gpath = gpath_list[aid]
#             root_obj = ut.LazyDict({
#                 'aid': aid,
#                 'gpath': gpath,
#                 'image': lambda: vt.imread(gpath)
#             })
#             yield root_obj


#     @depc.register_preproc(
#         tablename='chip', parents=[dummy_root], colnames=['size', 'chip'],
#         coltypes=[(int, int), vt.imread], configclass=DummyChipConfig,
#         asobject=True)
#     def dummy_preproc_chip(depc, annot_list, config=None):
#         """
#         TODO: Infer properties from docstr

#         Args:
#             annot_list (list): list of annot objects
#             config (dict): config dictionary

#         Returns:
#             tuple : ((int, int), ('extern', vt.imread))
#         """
#         if config is None:
#             config = {}
#         # Demonstates using asobject to get input to function as a dictionary
#         # of properties
#         for annot in annot_list:
#             print('[preproc] Computing chips of annot=%r' % (annot,))
#             chip_fpath = annot['gpath']
#             w, h = vt.image.open_image_size(chip_fpath)
#             size = (w, h)
#             print('* chip_fpath = %r' % (chip_fpath,))
#             print('* size = %r' % (size,))
#             yield size, chip_fpath

# #config_hashid = config.get('feat_cfgstr')
# #assert config_hashid is not None
# # TODO store config_rowid in qparams
# #else:
# #    config_hashid = db.cfg.feat_cfg.get_cfgstr()
# if False:
#     if config is not None:
#         try:
#             #config_hashid = 'none'
#             config_hashid = config.get(table.tablename + '_hashid')
#         except KeyError:
#             try:
#                 subconfig = config.get(table.tablename + '_config')
#                 config_hashid = ut.hashstr27(ut.to_json(subconfig))
#             except KeyError:
#                 print('[deptbl.config] Warning: Config must either'
#                       'contain a string <tablename>_hashid or a dict'
#                       '<tablename>_config')
#                 raise
#     else:
#         config_hashid = 'none'
