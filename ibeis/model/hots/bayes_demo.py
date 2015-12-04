from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
from six.moves import map
from ibeis.model.hots.bayes import make_name_model, test_model
print, rrr, profile = ut.inject2(__name__, '[bayes_demo]')


def make_bayes_notebook():
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes_demo --exec-make_bayes_notebook

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes_demo import *  # NOQA
        >>> result = make_bayes_notebook()
        >>> print(result)
    """
    from ibeis.templates import generate_notebook
    initialize = ut.codeblock(
        r'''
        # STARTBLOCK
        import os
        os.environ['UTOOL_NO_CNN'] = 'True'
        from ibeis.model.hots.bayes_demo import *  # NOQA
        # Matplotlib stuff
        import matplotlib as mpl
        %matplotlib inline
        %load_ext autoreload
        %autoreload
        from IPython.core.display import HTML
        HTML("<style>body .container { width:99% !important; }</style>")
        # ENDBLOCK
        '''
    )
    cell_list_def = [
        initialize,
        #show_model_templates,
        #demo_name_annot_complexity,
        ###demo_model_idependencies1,
        ###demo_model_idependencies2,
        #demo_single_add,
        #demo_single_add_soft,
        #demo_conflicting_evidence,
        demo_different_names,
        demo_modes,
    ]
    def format_cell(cell):
        if ut.is_funclike(cell):
            header = '# ' + ut.to_title_caps(ut.get_funcname(cell))
            code = (header, ut.get_func_sourcecode(cell, stripdef=True, stripret=True))
        else:
            code = (None, cell)
        return generate_notebook.format_cells(code)

    cell_list = ut.flatten([format_cell(cell) for cell in cell_list_def])
    nbstr = generate_notebook.make_notebook(cell_list)
    print('nbstr = %s' % (nbstr,))
    fpath = 'bayes_demo.ipynb'
    ut.writeto(fpath, nbstr)
    ut.startfile(fpath)


def show_model_templates():
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes_demo --exec-show_model_templates

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes_demo import *  # NOQA
        >>> result = show_model_templates()
        >>> ut.show_if_requested()
    """
    make_name_model(2, 2, verbose=True)


def demo_single_add():
    """
    This demo shows how a name is assigned to a new annotation.

    CommandLine:
        python -m ibeis.model.hots.bayes_demo --exec-demo_single_add --show --present --mode=1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes_demo import *  # NOQA
        >>> demo_single_add()
        >>> ut.show_if_requested()
    """
    # Initially there are only two annotations that have a strong match
    test_model(num_annots=2, num_names=5, score_evidence=['high'], name_evidence=[0])
    # Adding a new annotation does not change the original probabilites
    test_model(num_annots=3, num_names=5, score_evidence=['high'], name_evidence=[0])
    # Adding evidence that Na matches Nc does not influence the probability
    # that Na matches Nb. However the probability that Nb matches Nc goes up.
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high'], name_evidence=[0])
    # However, once Nb is scored against Nb that does increase the likelihood
    # that all 3 are fred goes up significantly.
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high', 'high'],
               name_evidence=[0])


def demo_single_add_soft():
    """
    This is the same as demo_single_add, but soft labels are used.

    CommandLine:
        python -m ibeis.model.hots.bayes_demo --exec-demo_single_add_soft --show --verbose

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes_demo import *  # NOQA
        >>> demo_single_add_soft()
        >>> ut.show_if_requested()
    """
    # Initially there are only two annotations that have a strong match
    #test_model(num_annots=2, num_names=5, score_evidence=['high'], name_evidence=[{0: .9}])
    # Adding a new annotation does not change the original probabilites
    #test_model(num_annots=3, num_names=5, score_evidence=['high'], name_evidence=[{0: .9}])
    # Adding evidence that Na matches Nc does not influence the probability
    # that Na matches Nb
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high'],
               name_evidence=[{0: .9}])
    # However, once Nb is scored against Nb that does increase the likelihood
    # that all 3 are fred goes up significantly.
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high', 'high'],
               name_evidence=[{0: .9}])


def demo_conflicting_evidence():
    """
    Notice that the number of annotations in the graph does not affect the
    probability of names.
    """
    # Initialized with two annots. Each are pretty sure they are someone else
    test_model(num_annots=2, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, {1: .9}])
    # Having evidence that they are different increases this confidence.
    test_model(num_annots=2, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, {1: .9}], other_evidence={'Sab': 'low'})
    # However,, confusion is introduced if there is evidence that they are the same
    test_model(num_annots=2, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, {1: .9}], other_evidence={'Sab': 'high'})

    # When Na is forced to be fred, this doesnt change Nbs evaulatation by more
    # than a few points
    test_model(num_annots=2, num_names=5, score_evidence=[],
               name_evidence=[0, {1: .9}], other_evidence={'Sab': 'high'})

    #test_model(num_annots=3, num_names=5, score_evidence=[],
    #           name_evidence=[{0: .9}, None, {1: .9}])
    #test_model(num_annots=3, num_names=5, score_evidence=[],
    #           name_evidence=[{0: .9}, None, {1: .9}], other_evidence={'Sac': 'low'})
    #test_model(num_annots=3, num_names=5, score_evidence=[],
    #           name_evidence=[{0: .9}, None, {1: .9}], other_evidence={'Sac': 'high'})


def demo_different_names():
    r"""
    Test what happens when an annotation need to choose between one of two
    names

    CommandLine:
        python -m ibeis.model.hots.bayes_demo --exec-demo_different_names --show --verbose --present --mode=2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes_demo import *  # NOQA
        >>> result = demo_different_names()
        >>> ut.show_if_requested()
    """
    # We will end up making annots a and b fred and c and d sue
    test_model(num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}])
    test_model(num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}],
               other_evidence={
                   'Sad': 'low'
               })
    test_model(num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}],
               other_evidence={
                   'Sad': 'low',
                   'Sab': 'high',
               })
    test_model(num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}],
               other_evidence={
                   'Sad': 'low',
                   'Sab': 'high',
                   'Scd': 'high',
               })
    test_model(num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}],
               other_evidence={
                   'Sad': 'low',
                   'Sab': 'high',
                   'Scd': 'high',
                   'Sac': 'low',
               })
    test_model(num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}],
               other_evidence={
                   'Sad': 'low',
                   'Sab': 'high',
                   'Scd': 'high',
                   'Sac': 'low',
                   'Sbc': 'low',
               })
    test_model(num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}],
               other_evidence={
                   'Sad': 'low',
                   'Sab': 'high',
                   'Scd': 'high',
                   'Sac': 'low',
                   'Sbc': 'low',
                   'Sbd': 'low',
               })


def demo_modes():
    """
    Look at the last result of the different names demo under differet modes

    """
    test_model(mode=1, num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}],
               other_evidence={
                   'Sad': 'low',
                   'Sab': 'high',
                   'Scd': 'high',
                   'Sac': 'low',
                   'Sbc': 'low',
                   'Sbd': 'low',
               })
    test_model(mode=2, num_annots=4, num_names=5, score_evidence=[],
               name_evidence=[{0: .9}, None, None, {1: .9}],
               other_evidence={
                   'Sad': 'low',
                   'Sab': 'high',
                   'Scd': 'high',
                   'Sac': 'low',
                   'Sbc': 'low',
                   'Sbd': 'low',
               })


def demo_name_annot_complexity():
    """
    This demo is meant to show the structure of the graph as more annotations
    and names are added.

    CommandLine:
        python -m ibeis.model.hots.bayes_demo --exec-demo_name_annot_complexity --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes_demo import *  # NOQA
        >>> demo_name_annot_complexity()
        >>> ut.show_if_requested()
    """
    # Initially there are 2 annots and 4 names
    test_model(num_annots=2, num_names=4, score_evidence=[], name_evidence=[])
    # Adding a name causes the probability of the other names to go down
    test_model(num_annots=2, num_names=5, score_evidence=[], name_evidence=[])
    # Adding an annotation wihtout matches does not effect probabilities of
    # names
    test_model(num_annots=3, num_names=5, score_evidence=[], name_evidence=[])
    test_model(num_annots=4, num_names=10, score_evidence=[], name_evidence=[])


def demo_model_idependencies1():
    """
    Independences of the 2 annot 2 name model
    """
    import re
    model = test_model(num_annots=2, num_names=2, score_evidence=[], name_evidence=[])[0]
    # This model has the following independenceis
    idens = model.get_independencies()
    # Might not be valid, try and collapse S and M
    xs = list(map(str, idens.independencies))
    xs = [re.sub(', M..', '', x) for x in xs]
    xs = [re.sub('M..,?', '', x) for x in xs]
    xs = [x for x in xs if not x.startswith('( _')]
    xs = [x for x in xs if not x.endswith('| )')]
    print('\n'.join(sorted(list(set(xs)))))


def demo_model_idependencies2():
    """
    Independences of the 3 annot 3 name model

    CommandLine:
        python -m ibeis.model.hots.bayes_demo --exec-demo_model_idependencies2 --mode=1
        python -m ibeis.model.hots.bayes_demo --exec-demo_model_idependencies2 --mode=2

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes_demo import *  # NOQA
        >>> result = demo_model_idependencies2()
        >>> print(result)
    """
    model = test_model(num_annots=3, num_names=3, score_evidence=[], name_evidence=[])[0]
    # This model has the following independenceis
    idens = model.get_independencies()
    print(idens)

# Might not be valid, try and collapse S and M
#xs = list(map(str, idens.independencies))
#import re
#xs = [re.sub(', M..', '', x) for x in xs]
#xs = [re.sub('M..,?', '', x) for x in xs]
#xs = [x for x in xs if not x.startswith('( _')]
#xs = [x for x in xs if not x.endswith('| )')]
#print('\n'.join(sorted(list(set(xs)))))


def bayesnet_cases():
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes_demo --exec-bayesnet_cases

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes_demo import *  # NOQA
        >>> bayesnet_cases()
    """
    from functools import partial
    import itertools
    count = partial(six.next, itertools.count(1))

    test_model(count(), num_annots=2, num_names=4, high_idx=[],
               name_evidence=[])  # init
    test_model(count(), num_annots=2, num_names=4, high_idx=[0],
               name_evidence=['n0'])  # Start with 4 names.
    test_model(count(), num_annots=2, num_names=5, high_idx=[0],
               name_evidence=['n0'])  # Add a name, Causes probability of match to go down
    test_model(count(), num_annots=3, num_names=5, high_idx=[0],
               name_evidence=['n0'])  # Add Annotation
    test_model(count(), num_annots=3, num_names=5, high_idx=[0, 2],
               name_evidence=['n0'])

    test_model(count(), num_annots=3, num_names=5, high_idx=[0, 2],
               name_evidence=['n0', {'n0': .9}])
    test_model(count(), num_annots=3, num_names=5, high_idx=[0],
               name_evidence=['n0', {'n0': .9}])
    test_model(count(), num_annots=3, num_names=5, high_idx=[0],
               name_evidence=[{'n0': .99}, {'n0': .9}])
    test_model(count(), num_annots=3, num_names=10, high_idx=[0],
               name_evidence=[{'n0': .99}, {'n0': .9}])
    test_model(count(), num_annots=3, num_names=10, high_idx=[0],
                       name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}])

    #fpath = test_model(count(), (3, 10), high_idx=[0, 1],
    #                   name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}])
    #fpath = test_model(count(), (3, 10), high_idx=[0, 1],
    #                   name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}, {'n0': .32}])
    test_model(count(), num_annots=3, num_names=10, high_idx=[0, 1, 2],
                       name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}, {'n0': .32}])
    # Fix indexing to move in diagonal order as opposed to row order
    test_model(count(), num_annots=4, num_names=10, high_idx=[0, 1, 2],
                       name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}, {'n0': .32}])
    #fpath = test_model(count(), (4, 10),
    #                   high_idx=[0, 1, 3], low_idx=[2], name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}, {'n0': .32}])

    #fpath = test_model(count(), (4, 10))

    #ut.startfile(fpath)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes_demo
        python -m ibeis.model.hots.bayes_demo --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
