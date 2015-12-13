from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
from ibeis.model.hots.bayes import make_name_model, test_model, draw_tree_model
print, rrr, profile = ut.inject2(__name__, '[demobayes]')


def one_test():
    """
    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-one_test --verbose
        python -m ibeis.model.hots.demobayes --exec-one_test --show --verbose --present

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = one_test()
        >>> ut.show_if_requested()
    """
    constkw = dict(
        num_annots=4, num_names=4,
        name_evidence=[0, None, None, None],
    )
    test_model(score_evidence=['veryhigh', 'high', 'high', None, None, 'high'], mode=4, **constkw)
    test_model(score_evidence=['veryhigh', 'high', 'high', None, None, 'high'], mode=1, **constkw)
    # test_model(score_evidence=['veryhigh', 'high', 'high', None, None, None], **constkw)
    # test_model(score_evidence=['veryhigh', 'high', 'high', None, 'high', None], **constkw)
    # constkw = dict(score_evidence=[], name_evidence=[], mode=1)
    # model, = test_model(num_annots=4, num_names=10, **constkw)
    # draw_tree_model(model)


def chuckchallenge():
    r"""
    Make a model that knows who the previous annots are and tries to classify a new annot

    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-chuckchallenge --diskshow --verbose --present --save demo4.png --dpath . --figsize=20,10 --dpi=128 --clipwhite

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = chuckchallenge()
        >>> ut.show_if_requested()
    """

    constkw = dict(
        num_annots=3, num_names=3,
        # name_evidence=[None, None, {0: .99}],
        # name_evidence=[1, 0, None],
    )
    model, evidence = test_model(
        mode=1,
        other_evidence={'Sab': 'high', 'Sac': 'high', 'Sbc': 'low'},
        **constkw)
    # model, evidence = test_model(
    #     mode=1,
    #     other_evidence={ 'Sab': 0, 'Sac': 0 },
    #     **constkw)


def classify_one_new_unknown():
    r"""
    Make a model that knows who the previous annots are and tries to classify a new annot

    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-classify_one_new_unknown --verbose
        python -m ibeis.model.hots.demobayes --exec-classify_one_new_unknown --show --verbose --present
        python3 -m ibeis.model.hots.demobayes --exec-classify_one_new_unknown --verbose
        python3 -m ibeis.model.hots.demobayes --exec-classify_one_new_unknown --verbose --diskshow --verbose --present --save demo5.png --dpath . --figsize=20,10 --dpi=128 --clipwhite

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = classify_one_new_unknown()
        >>> ut.show_if_requested()
    """
    if False:
        constkw = dict(
            num_annots=5, num_names=3,
            name_evidence=[0]
            #name_evidence=[0, 0, 1, 1, None],
            #name_evidence=[{0: .99}, {0: .99}, {1: .99}, {1: .99}, None],
            #name_evidence=[0, {0: .99}, {1: .99}, 1, None],
        )
        test_model(score_evidence=['high', 'low', 'low', 'low', 'low', 'high'], mode=1, show_prior=True, **constkw)
        #test_model(score_evidence=['high', 'low', 'low', 'low', 'low', 'high', 'high'], mode=1, show_prior=True, **constkw)
        #test_model(score_evidence=['high', 'low', 'low', 'low', 'low', 'high', 'high', 'high'], mode=1, show_prior=True, **constkw)
        #test_model(score_evidence=['high', 'low', 'low', 'low', 'low', 'high', 'high', 'high', 'high', 'high'], mode=1, show_prior=True, **constkw)
        #test_model(score_evidence=['high', 'low', 'low', 'low', 'low', 'high', 'low', 'low', 'low', 'low'], mode=1, show_prior=True, **constkw)

    #from ibeis.model.hots.demobayes import *
    constkw = dict(
        num_annots=4, num_names=4,
    )
    model, evidence = test_model(
        mode=1, show_prior=False,
        # lll and llh have strikingly different
        # probability of M marginals
        score_evidence=['low', 'low', 'high'],
        other_evidence={
        },
        **constkw)

    # if False:
    #     #from ibeis.model.hots import pgm_ext
    #     import pgmpy
    #     import pgmpy.models
    #     import pgmpy.inference
    #     assert isinstance(model, pgmpy.models.BayesianModel)

    #     cpds = model.ttype2_cpds['score']
    #     ut.embed()
    #     #model1 = model.copy()
    #     model.remove_cpds(*cpds)
    #     model.remove_cpds(*cpds)
    #     variables = [cpd.variable for cpd in cpds]
    #     self.delete_variables(variables)

    #     query_vars = ut.setdiff_ordered(model.nodes(), list(evidence.keys()))
    #     query_vars = ut.setdiff_ordered(query_vars, ['Sab', 'Sbc', 'Sac'])

    #     infr = pgmpy.inference.VariableElimination(model)
    #     #infr = pgmpy.inference.BeliefPropagation(model)
    #     #variables = query_vars
    #     map_assign = infr.map_query(query_vars, evidence)
    #     print('map_assign = %r' % (map_assign,))
    #     #infr.max_marginal(query_vars)
    #     infr.map_query(query_vars)

    #     #model

    #     #from pgmpy.factors.Factor import factor_product
    #     #self = infr
    #     #elimination_order = None

    #     # MAP estimate
    #     # argmax_y P(Y=y | E=e)
    #     joint = model.joint_distribution()
    #     j1 = joint.evidence_based_reduction(evidence=evidence)
    #     print(j1._str(tablefmt='psql', sort=-1))

    #     ut.embed()

    #     new_rows = j1._row_labels()
    #     new_vals = j1.values.ravel()
    #     new_vals2 = new_vals.compress(new_vals > 0)
    #     new_row2 = ut.compress(new_rows, new_vals > 0)
    #     print('new_vals2 = %r' % (new_vals2,))
    #     print('new_row2 = %r' % (new_row2,))
    #     print(j1.marginalize(['Na', 'Nb'], inplace=False))
    #     print(j1.marginalize(['Nc', 'Na'], inplace=False))
    #     print(j1.marginalize(['Nb', 'Nc'], inplace=False))


def test_triangle_property():
    r"""
    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-test_triangle_property --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = test_triangle_property()
        >>> ut.show_if_requested()
    """
    constkw = dict(
        num_annots=3, num_names=3,
        name_evidence=[],
    )
    test_model(
        mode=1, show_prior=True,
        other_evidence={
            'Mab': False,
            'Mac': False,
            #'Na': 'fred',
            #'Nb': 'sue',
        },
        **constkw)


def demo_structure():
    r"""
    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-demo_structure --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = demo_structure()
        >>> ut.show_if_requested()
    """
    constkw = dict(score_evidence=[], name_evidence=[], mode=3)
    model, = test_model(num_annots=4, num_names=4, **constkw)
    draw_tree_model(model)


def make_bayes_notebook():
    r"""
    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-make_bayes_notebook

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = make_bayes_notebook()
        >>> print(result)
    """
    from ibeis.templates import generate_notebook
    initialize = ut.codeblock(
        r'''
        # STARTBLOCK
        import os
        os.environ['UTOOL_NO_CNN'] = 'True'
        from ibeis.model.hots.demobayes import *  # NOQA
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
        show_model_templates,
        demo_modes,
        demo_name_annot_complexity,
        ###demo_model_idependencies,
        demo_single_add,
        demo_ambiguity,
        demo_conflicting_evidence,
        demo_annot_idependence_overlap,
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
    fpath = 'demobayes.ipynb'
    ut.writeto(fpath, nbstr)
    ut.startfile(fpath)


def show_model_templates():
    r"""
    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-show_model_templates

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = show_model_templates()
        >>> ut.show_if_requested()
    """
    make_name_model(2, 2, verbose=True, mode=1)
    print('-------------')
    make_name_model(2, 2, verbose=True, mode=2)


def demo_single_add():
    """
    This demo shows how a name is assigned to a new annotation.

    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-demo_single_add --show --present --mode=1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> demo_single_add()
        >>> ut.show_if_requested()
    """
    # Initially there are only two annotations that have a strong match
    name_evidence = [{0: .9}]  # Soft label
    name_evidence = [0]  # Hard label
    test_model(num_annots=2, num_names=5, score_evidence=['high'], name_evidence=name_evidence)
    # Adding a new annotation does not change the original probabilites
    test_model(num_annots=3, num_names=5, score_evidence=['high'], name_evidence=name_evidence)
    # Adding evidence that Na matches Nc does not influence the probability
    # that Na matches Nb. However the probability that Nb matches Nc goes up.
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high'], name_evidence=name_evidence)
    # However, once Nb is scored against Nb that does increase the likelihood
    # that all 3 are fred goes up significantly.
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high', 'high'],
               name_evidence=name_evidence)


def demo_conflicting_evidence():
    """
    Notice that the number of annotations in the graph does not affect the
    probability of names.
    """
    # Initialized with two annots. Each are pretty sure they are someone else
    constkw = dict(num_annots=2, num_names=5, score_evidence=[])
    test_model(name_evidence=[{0: .9}, {1: .9}], **constkw)
    # Having evidence that they are different increases this confidence.
    test_model(name_evidence=[{0: .9}, {1: .9}], other_evidence={'Sab': 'low'}, **constkw)
    # However,, confusion is introduced if there is evidence that they are the same
    test_model(name_evidence=[{0: .9}, {1: .9}], other_evidence={'Sab': 'high'}, **constkw)
    # When Na is forced to be fred, this doesnt change Nbs evaulatation by more
    # than a few points
    test_model(name_evidence=[0, {1: .9}], other_evidence={'Sab': 'high'}, **constkw)


def demo_ambiguity():
    r"""
    Test what happens when an annotation need to choose between one of two
    names

    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-demo_ambiguity --show --verbose --present

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = demo_ambiguity()
        >>> ut.show_if_requested()
    """
    constkw = dict(
        num_annots=3, num_names=3,
        name_evidence=[0],
        #name_evidence=[],
        #name_evidence=[{0: '+eps'}, {1: '+eps'}, {2: '+eps'}],
    )
    test_model(score_evidence=['low', 'low', 'high'], mode=1, show_prior=True, **constkw)

    ## We will end up making annots a and b fred and c and d sue
    #constkw = dict(
    #    num_annots=4, num_names=5,
    #    #name_evidence=[{0: .9}, None, None, {1: .9}]
    #    name_evidence=[0, None, None, None]
    #    #name_evidence=[0, None, None, None]
    #)
    #test_model(score_evidence=[None, None, None, None, None, None], show_prior=True, **constkw)
    #test_model(score_evidence=['high', None, None, None, None, None], **constkw)
    #test_model(score_evidence=['high', 'low', None, None, None, None], **constkw)
    #test_model(score_evidence=['high', 'low', 'low', None, None, None], **constkw)
    #test_model(score_evidence=['high', 'low', 'low', 'low', None, None], **constkw)
    #test_model(score_evidence=['high', 'low', 'low', 'low', 'low', None], **constkw)
    #test_model(score_evidence=['high', 'low', 'low', 'low', 'low', 'high'], **constkw)
    ## Resolve ambiguity
    #constkw['name_evidence'][-1] = 1
    #test_model(score_evidence=['high', 'low', 'low', 'low', 'low', 'high'], **constkw)


def demo_annot_idependence_overlap():
    r"""
    Given:
        * an unknown annotation \d
        * three annots with the same name (Fred) \a, \b, and \c
        * \a and \b are near duplicates
        * (\a and \c) / (\b and \c) are novel views

    Goal:
        * If \d matches to \a and \b the probably that \d is Fred should not be
          much more than if \d matched only \a or only \b.

        * The probability that \d is Fred given it matches to any of the 3 annots
           alone should be equal

            P(\d is Fred | Mad=1) = P(\d is Fred | Mbd=1) = P(\d is Fred | Mcd=1)

        * The probability that \d is fred given two matches to any of those two annots
          should be greater than the probability given only one.

            P(\d is Fred | Mad=1, Mbd=1) > P(\d is Fred | Mad=1)
            P(\d is Fred | Mad=1, Mcd=1) > P(\d is Fred | Mad=1)

        * The probability that \d is fred given matches to two near duplicate
          matches should be less than
          if \d matches two non-duplicate matches.

            P(\d is Fred | Mad=1, Mcd=1) > P(\d is Fred | Mad=1, Mbd=1)

        * The probability that \d is fred given two near duplicates should be only epsilon greater than
          a match to either one individually.

            P(\d is Fred | Mad=1, Mbd=1) = P(\d is Fred | Mad=1) + \epsilon

    Method:

        We need to model the fact that there are other causes that create the
        effect of a high score.  Namely, near duplicates.
        This can be done by adding an extra conditional that score depends on
        if they match as well as if they are near duplicates.

        P(S_ij | Mij) --> P(S_ij | Mij, Dij)

        where

        Dij is a random variable indicating if the image is a near duplicate.

        We can model this as an independant variable

        P(Dij) = {True: .5, False: .5}

        or as depending on if the names match.

        P(Dij | Mij) = {'same': {True: .5, False: .5} diff: {True: 0, False 1}}



    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-demo_annot_idependence_overlap --verbose --present --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = demo_annot_idependence_overlap()
        >>> ut.show_if_requested()
    """
    # We will end up making annots a and b fred and c and d sue
    constkw = dict(
        num_annots=4, num_names=4,
        name_evidence=[{0: '+eps'}, {1: '+eps'}, {2: '+eps'}, {3: '+eps'}],
        #name_evidence=[{0: .9}, None, None, {1: .9}]
        #name_evidence=[0, None, None, None]
        #name_evidence=[0, None, None, None]
    )
    test_model(score_evidence=['high', 'high', 'high', None, None, None], **constkw)
    test_model(score_evidence=['high', 'high', 'low', None, None, None], **constkw)
    test_model(score_evidence=['high', 'low', 'low', None, None, None], **constkw)


def demo_modes():
    """
    Look at the last result of the different names demo under differet modes
    """
    constkw = dict(
        num_annots=4, num_names=8,
        score_evidence=['high', 'low', 'low', 'low', 'low', 'high'],
        #name_evidence=[{0: .9}, None, None, {1: .9}],
        #name_evidence=[0, None, None, 1],
        name_evidence=[0, None, None, None],
        #other_evidence={
        #    'Sad': 'low',
        #    'Sab': 'high',
        #    'Scd': 'high',
        #    'Sac': 'low',
        #    'Sbc': 'low',
        #    'Sbd': 'low',
        #}
    )
    # The first mode uses a hidden Match layer
    test_model(mode=1, **constkw)
    # The second mode directly maps names to scores
    test_model(mode=2, **constkw)
    test_model(mode=3, noquery=True, **constkw)
    test_model(mode=4, noquery=True, **constkw)


def demo_name_annot_complexity():
    """
    This demo is meant to show the structure of the graph as more annotations
    and names are added.

    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-demo_name_annot_complexity --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> demo_name_annot_complexity()
        >>> ut.show_if_requested()
    """
    constkw = dict(score_evidence=[], name_evidence=[], mode=1)
    # Initially there are 2 annots and 4 names
    model, = test_model(num_annots=2, num_names=4, **constkw)
    draw_tree_model(model)
    # Adding a name causes the probability of the other names to go down
    model, = test_model(num_annots=2, num_names=5, **constkw)
    draw_tree_model(model)
    # Adding an annotation wihtout matches dos not effect probabilities of
    # names
    model, = test_model(num_annots=3, num_names=5, **constkw)
    draw_tree_model(model)
    model, = test_model(num_annots=4, num_names=10, **constkw)
    draw_tree_model(model)
    # Given A annots, the number of score nodes is (A ** 2 - A) / 2
    model, = test_model(num_annots=5, num_names=5, **constkw)
    draw_tree_model(model)
    #model, = test_model(num_annots=6, num_names=5, score_evidence=[], name_evidence=[], mode=1)
    #draw_tree_model(model)


def demo_model_idependencies():
    """
    Independences of the 3 annot 3 name model

    CommandLine:
        python -m ibeis.model.hots.demobayes --exec-demo_model_idependencies --mode=1 --num-names=2 --show
        python -m ibeis.model.hots.demobayes --exec-demo_model_idependencies --mode=2

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.demobayes import *  # NOQA
        >>> result = demo_model_idependencies()
        >>> print(result)
        >>> ut.show_if_requested()
    """
    num_names = ut.get_argval('--num-names', default=3)
    model = test_model(num_annots=num_names, num_names=num_names, score_evidence=[], name_evidence=[])[0]
    # This model has the following independenceis
    idens = model.get_independencies()

    iden_strs = [', '.join(sorted(iden.event1)) +
                 ' _L ' +
                 ','.join(sorted(iden.event2)) +
                 ' | ' +
                 ', '.join(sorted(iden.event3))
                 for iden in idens.independencies]
    print('general idependencies')
    print(ut.align(ut.align('\n'.join(sorted(iden_strs)), '_'), '|'))
    #ut.embed()
    #model.is_active_trail('Na', 'Nb', 'Sab')

# Might not be valid, try and collapse S and M
#xs = list(map(str, idens.independencies))
#import re
#xs = [re.sub(', M..', '', x) for x in xs]
#xs = [re.sub('M..,?', '', x) for x in xs]
#xs = [x for x in xs if not x.startswith('( _')]
#xs = [x for x in xs if not x.endswith('| )')]
#print('\n'.join(sorted(list(set(xs)))))


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.model.hots.demobayes
        python -m ibeis.model.hots.demobayes --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
