import ibeis
import vtool
import utool
import numpy as np
import numpy.linalg as npl  # NOQA
import pandas as pd
np.set_printoptions(precision=2)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('isplay.notebook_repr_html', True)
ibeis.ensure_pz_mtest()
ibs = ibeis.opendb('PZ_MTEST')

taids = ibs.get_valid_aids()
tvecs_list = ibs.get_annot_desc(taids)
tkpts_list = ibs.get_annot_kpts(taids)
tvec_list = np.vstack(tvecs_list)
print(idx2_vec)

labels, words = vtool.clustering.precompute_akmeans(tvec_list, 1000, 30, cache_dir='.')
tvecdf_list = [pd.DataFrame(vecs) for vecs in  tvecs_list]
tvecs_df = pd.DataFrame(tvecdf_list, index=taids)
kpts_col = pd.DataFrame(tkpts_list, index=taids, columns=['kpts'])
vecs_col = pd.DataFrame(tvecs_list, index=taids, columns=['vecs'])
tvecs_dflist = [pd.DataFrame(vecs, index=np.arange(len(vecs))) for vecs in tvecs_list]
pd.concat(tvecs_dflist)
# Bui


taids = ibs.get_valid_aids()
tvecs_list = ibs.get_annot_desc(taids)
tkpts_list = ibs.get_annot_kpts(taids)

orig_idx2_vec, orig_idx2_ax, orig_idx2_fx = vtool.nearest_neighbors.invertable_stack(tvecs_list, taids)
annots_df = pd.concat([vecs_col, kpts_col], axis=1)
annots_df

idx2_vec = np.vstack(annots_df['vecs'].values)
#idx2_ax =
idx2_vec, idx2_ax, idx2_fx = vtool.nearest_neighbors.invertable_stack(tvecs_list, taids)

wordflann = vtool.nearest_neighbors.flann_cache(words, cache_dir=utool.get_app_resource_dir('vtool'), flann_params={})
indexes, dists = wordflann.nn_index(tvec_list, 1)
wx2_tvec = utool.group_items(tvec_list.tolist(), indexes.tolist())
word_index = list(wx2_tvec.keys())[0]
vecs = np.array(wx2_tvec[word_index], dtype=np.float64)
word = np.array(words[word_index], dtype=np.float64)
residuals = np.array([word - vec for vec in vecs])
residuals_n = vtool.linalg.normalize_rows(residuals)
rvecs = residuals_n
similarity_matrix = (rvecs.dot(rvecs.T))
