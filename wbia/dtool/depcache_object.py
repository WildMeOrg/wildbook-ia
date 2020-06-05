# -*- coding: utf-8 -*-
# import utool as ut
# import ubelt as ub


# class DepcacheObject(ub.NiceRepr):

#     def __init__(self):
#         self._rowid
#         self._uuid
#         self._depc

#     def __nice__(self):
#         pass

#     def __hash__(self):
#         pass

#     @classmethod
#     def testdata(cls):
#         pass

#     def get_tags(self):
#         pass

#     def add_tags(self):
#         """ useful for generating manual labels on the fly """
#         pass

#     def remove_tags(self):
#         pass

#     # def delete(self):
#     #     pass

#     # def precompute(self):
#     #     pass


# class DepcacheObjectList(ub.NiceRepr):
#     """ behaves as a list of depcache objects, but implicitly represent
#     properties for efficiency

#     Example:
#         annots = AnnotationList()
#         chip_list = annots.chip()
#         feat_list = annots.feat()
#         indexer = annots.nnindexer()

#         query_annots = AnnotationList()
#         query_feats = query_annots.feat()
#         [indexer.knn(feat, K) for feat in query_feats]
#     """

#     pass


# class Property(ub.NiceRepr):
#     pass


# class Annotation(DepcacheObject):

#     _properties = [
#         Property('name'),
#         Property('bbox'),
#         Property('image'),
#         # Property('species'),
#     ]

#     def compute_chip(self, config):
#         pass

#     def compute_probchip(self, config):
#         pass

#     def compute_feat(self, config):
#         pass

#     def compute_feat_weights(self, config):
#         pass


# class VsoneConfig(object):
#     _param_info_list = [
#         ut.ParamInfo('distinctiveness_model'),
#         ut.ParamInfo('score_norm_model'),
#     ]

#     pass


# class AnnotationAlgorithms(object):

#     def query(query_annot, others, config):
#         pass

#     def query_vsmany(query_annot, others, config):
#         pass

#     def query_vsone(query_annot, other_annot, config):
#         pass

#     def query_smk(query_annot, others, config):
#         pass

#     def query_dtw(query_annot, other_annot, config):
#         pass


# class AnnotSubroutines(object):

#     def compute_vsmany_matches(query_annot, indexer, config):
#         pass

#     def compute_vsone_matches(annot1, annot2, config):
#         pass

#     def compute_many_matches(self, others):
#         pass

#     def compute_smk_matches(query_annot, indexer, config):
#         pass

#     def compute_smk_score(query_annot, nnindexer):
#         pass

#     def compute_feat_neighbors(query_annot, nnindexer):
#         pass

#     def weight_feat_matches(query_neighbors, config):
#         pass

#     def spatially_verify_matches(query_neighbors, config):
#         pass


# class AnnotationModels(object):

#     @staticmethod
#     def compute_nnindexer(others):
#         pass

#     @staticmethod
#     def compute_smk_indexer(others):
#         pass

#     @staticmethod
#     def compute_smk_repr(others):
#         pass

#     @staticmethod
#     def compute_smk_vocab(others):
#         pass


# class FlukeAnnotationExtention(Annotation):

#     def compute_notch_tips(self):
#         pass

#     def compute_trailing_edge(self):
#         pass

#     def compute_block_curvature(self):
#         pass

#     def compute_dt_bcw(self, other):
#         pass
