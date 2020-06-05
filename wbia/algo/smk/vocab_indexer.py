# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from wbia import dtool
import utool as ut
import vtool as vt
from vtool._pyflann_backend import pyflann as pyflann
from wbia.algo.smk import pickle_flann
import numpy as np
import warnings
from wbia.control.controller_inject import register_preprocs

(print, rrr, profile) = ut.inject2(__name__)


derived_attribute = register_preprocs['annot']


class VocabConfig(dtool.Config):
    _param_info_list = [
        ut.ParamInfo('algorithm', 'minibatch', 'alg'),
        ut.ParamInfo('random_seed', 42, 'seed'),
        ut.ParamInfo('num_words', 1000, 'n'),
        ut.ParamInfo('version', 2),
        ut.ParamInfo('n_init', 1),
    ]


@ut.reloadable_class
class VisualVocab(ut.NiceRepr):
    """
    Class that maintains a list of visual words (cluster centers)
    Also maintains a nearest neighbor index structure for finding words.
    This class is build using the depcache
    """

    def __init__(vocab, words=None):
        vocab.wx_to_word = words
        vocab.wordflann = None

        vocab.flann_params = vt.get_flann_params(random_seed=42)
        vocab.flann_params['checks'] = 1024
        vocab.flann_params['trees'] = 8
        # TODO: grab the depcache rowid and maybe config?
        # make a dtool.Computable

    def __nice__(vocab):
        return 'nW=%r' % (ut.safelen(vocab.wx_to_word))

    def __len__(vocab):
        return len(vocab.wx_to_word)

    @property
    def shape(vocab):
        return vocab.wx_to_word.shape

    def __getstate__(vocab):
        state = vocab.__dict__.copy()
        if 'wx2_word' in state:
            state['wx_to_word'] = state.pop('wx2_word')
        state['wordindex_bytes'] = vocab.wordflann.dumps()
        del state['wordflann']
        return state

    def __setstate__(vocab, state):
        wordindex_bytes = state.pop('wordindex_bytes')
        vocab.__dict__.update(state)
        flannclass = pickle_flann.PickleFLANN
        vocab.wordflann = flannclass()
        try:
            vocab.wordflann.loads(wordindex_bytes, vocab.wx_to_word)
        except Exception:
            print('Fixing vocab problem')
            vocab.build()

    def build(vocab, verbose=True):
        num_vecs = len(vocab.wx_to_word)
        if vocab.wordflann is None:
            flannclass = pickle_flann.PickleFLANN
            vocab.wordflann = flannclass()
        if verbose:
            print(' ...build kdtree with %d points (may take a sec).' % num_vecs)
            tt = ut.tic(msg='Building vocab index')
        if num_vecs == 0:
            print('WARNING: CANNOT BUILD FLANN INDEX OVER 0 POINTS.')
            print('THIS MAY BE A SIGN OF A DEEPER ISSUE')
        else:
            vocab.wordflann.build_index(vocab.wx_to_word, **vocab.flann_params)
        if verbose:
            ut.toc(tt)

    def nn_index(vocab, idx_to_vec, nAssign, checks=None):
        """
            >>> idx_to_vec = depc.d.get_feat_vecs(aid_list)[0]
            >>> vocab = vocab
            >>> nAssign = 1
        """
        # Assign each vector to the nearest visual words
        assert nAssign > 0, 'cannot assign to 0 neighbors'
        if checks is None:
            checks = vocab.flann_params['checks']
        try:
            idx_to_vec = idx_to_vec.astype(vocab.wordflann._FLANN__curindex_data.dtype)
            _idx_to_wx, _idx_to_wdist = vocab.wordflann.nn_index(
                idx_to_vec, nAssign, checks=checks
            )
        except pyflann.FLANNException as ex:
            ut.printex(
                ex,
                'probably misread the cached flann_fpath=%r'
                % (getattr(vocab.wordflann, 'flann_fpath', None),),
            )
            raise
        else:
            _idx_to_wx = vt.atleast_nd(_idx_to_wx, 2)
            _idx_to_wdist = vt.atleast_nd(_idx_to_wdist, 2)
            return _idx_to_wx, _idx_to_wdist

    def render_vocab(vocab):
        """
        Renders the average patch of each word.
        This is a quick visualization of the entire vocabulary.

        CommandLine:
            python -m wbia.algo.smk.vocab_indexer render_vocab --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.smk.vocab_indexer import *  # NOQA
            >>> vocab = testdata_vocab('PZ_MTEST', num_words=64)
            >>> all_words = vocab.render_vocab()
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> pt.qt4ensure()
            >>> pt.imshow(all_words)
            >>> ut.show_if_requested()
        """
        import wbia.plottool as pt

        wx_list = list(range(len(vocab)))
        # wx_list = ut.strided_sample(wx_list, 64)
        wx_list = ut.strided_sample(wx_list, 64)

        word_patch_list = []
        for wx in ut.ProgIter(wx_list, bs=True, lbl='building patches'):
            word = vocab.wx_to_word[wx]
            word_patch = vt.inverted_sift_patch(word, 64)
            word_patch = pt.render_sift_on_patch(word_patch, word)
            word_patch_list.append(word_patch)

        all_words = vt.stack_square_images(word_patch_list)
        return all_words


@derived_attribute(
    tablename='vocab',
    parents=['feat*'],
    colnames=['words'],
    coltypes=[VisualVocab],
    configclass=VocabConfig,
    chunksize=1,
    fname='visual_vocab',
    taggable=True,
    vectorized=False,
)
def compute_vocab(depc, fid_list, config):
    r"""
    Depcache method for computing a new visual vocab

    CommandLine:
        python -m wbia.core_annots --exec-compute_neighbor_index --show
        python -m wbia show_depc_annot_table_input --show --tablename=neighbor_index

        python -m wbia.algo.smk.vocab_indexer --exec-compute_vocab:0
        python -m wbia.algo.smk.vocab_indexer --exec-compute_vocab:1

        # FIXME make util_tests register
        python -m wbia.algo.smk.vocab_indexer compute_vocab:0

    Ignore:
        >>> # Lev Oxford Debug Example
        >>> import wbia
        >>> ibs = wbia.opendb('Oxford')
        >>> depc = ibs.depc
        >>> table = depc['vocab']
        >>> # Check what currently exists in vocab table
        >>> table.print_configs()
        >>> table.print_table()
        >>> table.print_internal_info()
        >>> # Grab aids used to compute vocab
        >>> from wbia.expt.experiment_helpers import get_annotcfg_list
        >>> expanded_aids_list = get_annotcfg_list(ibs, ['oxford'])[1]
        >>> qaids, daids = expanded_aids_list[0]
        >>> vocab_aids = daids
        >>> config = {'num_words': 64000}
        >>> exists = depc.check_rowids('vocab', [vocab_aids], config=config)
        >>> print('exists = %r' % (exists,))
        >>> vocab_rowid = depc.get_rowids('vocab', [vocab_aids], config=config)[0]
        >>> print('vocab_rowid = %r' % (vocab_rowid,))
        >>> vocab = table.get_row_data([vocab_rowid], 'words')[0]
        >>> print('vocab = %r' % (vocab,))

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.smk.vocab_indexer import *  # NOQA
        >>> # Test depcache access
        >>> import wbia
        >>> ibs, aid_list = wbia.testdata_aids('testdb1')
        >>> depc = ibs.depc_annot
        >>> input_tuple = [aid_list]
        >>> rowid_kw = {}
        >>> tablename = 'vocab'
        >>> vocabid_list = depc.get_rowids(tablename, input_tuple, **rowid_kw)
        >>> vocab = depc.get(tablename, input_tuple, 'words')[0]
        >>> assert vocab.wordflann is not None
        >>> assert vocab.wordflann._FLANN__curindex_data is not None
        >>> assert vocab.wordflann._FLANN__curindex_data is vocab.wx_to_word

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.smk.vocab_indexer import *  # NOQA
        >>> import wbia
        >>> ibs, aid_list = wbia.testdata_aids('testdb1')
        >>> depc = ibs.depc_annot
        >>> fid_list = depc.get_rowids('feat', aid_list)
        >>> config = VocabConfig()
        >>> vocab, train_vecs = ut.exec_func_src(compute_vocab, keys=['vocab', 'train_vecs'])
        >>> idx_to_vec = depc.d.get_feat_vecs(aid_list)[0]
        >>> self = vocab
        >>> ut.quit_if_noshow()
        >>> data = train_vecs
        >>> centroids = vocab.wx_to_word
        >>> import wbia.plottool as pt
        >>> vt.plot_centroids(data, centroids, num_pca_dims=2)
        >>> ut.show_if_requested()
        >>> #config = ibs.depc_annot['vocab'].configclass()

    """
    print('[IBEIS] COMPUTE_VOCAB:')
    vecs_list = depc.get_native('feat', fid_list, 'vecs')
    train_vecs = np.vstack(vecs_list).astype(np.float32)
    num_words = config['num_words']
    print(
        '[smk_index] Train Vocab(nWords=%d) using %d annots and %d descriptors'
        % (num_words, len(fid_list), len(train_vecs))
    )
    if config['algorithm'] == 'kdtree':
        flann_params = vt.get_flann_params(random_seed=42)
        kwds = dict(max_iters=20, flann_params=flann_params)
        words = vt.akmeans(train_vecs, num_words, **kwds)
    elif config['algorithm'] == 'minibatch':
        print('Using minibatch kmeans')
        import sklearn.cluster

        rng = np.random.RandomState(config['random_seed'])
        n_init = config['n_init']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            init_size = int(num_words * 4)
            batch_size = 1000
            n_batches = ut.get_num_chunks(train_vecs.shape[0], batch_size)
            minibatch_params = dict(
                n_clusters=num_words,
                init='k-means++',
                init_size=init_size,
                n_init=n_init,
                max_iter=30000 // n_batches,
                batch_size=batch_size,
                tol=0.0,
                max_no_improvement=10,
                reassignment_ratio=0.01,
            )
            print('minibatch_params = %s' % (ut.repr4(minibatch_params),))
            clusterer = sklearn.cluster.MiniBatchKMeans(
                compute_labels=False, random_state=rng, verbose=2, **minibatch_params
            )
            try:
                clusterer.fit(train_vecs)
            except (Exception, KeyboardInterrupt) as ex:
                ut.printex(ex, tb=True)
                if ut.is_developer():
                    ut.embed()
                else:
                    raise
        words = clusterer.cluster_centers_
        print('Finished clustering')
    # if False:
    #     flann_params['checks'] = 64
    #     flann_params['trees'] = 4
    #     num_words = 128
    #     centroids = vt.initialize_centroids(num_words, train_vecs, 'akmeans++')
    #     words, hist = vt.akmeans_iterations(
    #         train_vecs, centroids, max_iters=1000, monitor=True,
    #         flann_params=flann_params)

    print('Constructing vocab')
    vocab = VisualVocab(words)
    print('Building vocab index')
    vocab.build()
    print('Returning vocab')
    return (vocab,)


def testdata_vocab(defaultdb='testdb1', **kwargs):
    """
    >>> from wbia.algo.smk.vocab_indexer import *  # NOQA
    >>> defaultdb='testdb1'
    >>> kwargs = {'num_words': 1000}
    """
    import wbia

    ibs, aids = wbia.testdata_aids(defaultdb=defaultdb)
    config = kwargs
    # vocab = new_load_vocab(ibs, aid_list, kwargs)
    # Hack in depcache info to the loaded vocab class
    # (maybe this becomes part of the depcache)
    rowid = ibs.depc.get_rowids('vocab', [aids], config=config)[0]
    # rowid = 1
    table = ibs.depc['vocab']
    vocab = table.get_row_data([rowid], 'words')[0]
    vocab.rowid = rowid
    vocab.config_history = table.get_config_history([vocab.rowid])[0]
    vocab.config = table.get_row_configs([vocab.rowid])[0]
    return vocab


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/wbia/wbia/algo/smk
        python ~/code/wbia/wbia/algo/smk/vocab_indexer.py
        python ~/code/wbia/wbia/algo/smk/vocab_indexer.py --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
