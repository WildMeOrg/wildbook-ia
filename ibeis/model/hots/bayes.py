# -*- coding: utf-8 -*-
"""

1) Ambiguity / num names
2) independence of annotations
3) continuous
4) exponential case
5) speicifc examples of our prob
6) human in loop

Arc reversal
http://www.cs.toronto.edu/~cebly/Papers/simulation.pdf

TODO:
    Need to find faster more mature libraries
    http://dlib.net/bayes.html
    http://www.cs.waikato.ac.nz/ml/weka/
    http://www.cs.waikato.ac.nz/~remco/weka.bn.pdf
    https://code.google.com/p/pebl-project/
    https://github.com/abhik/pebl
    http://www.cs.ubc.ca/~murphyk/Software/bnsoft.html

    Demo case where we think we know the labels of others.  Only one unknown
    name. Need to classify it as one of the other known names.

References:
    https://en.wikipedia.org/wiki/Bayesian_network
    https://class.coursera.org/pgm-003/lecture/17
    http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
    http://www3.cs.stonybrook.edu/~sael/teaching/cse537/Slides/chapter14d_BP.pdf
    http://www.cse.unsw.edu.au/~cs9417ml/Bayes/Pages/PearlPropagation.html
    https://github.com/pgmpy/pgmpy.git
    http://pgmpy.readthedocs.org/en/latest/
    http://nipy.bic.berkeley.edu:5000/download/11
    http://pgmpy.readthedocs.org/en/latest/wiki.html#add-feature-to-accept-and-output-state-names-for-models
    http://www.csse.monash.edu.au/bai/book/BAI_Chapter2.pdf


Clustering with CRF:
    http://srl.informatik.uni-freiburg.de/publicationsdir/tipaldiIROS09.pdf
    http://www.dis.uniroma1.it/~dottoratoii/media/students/documents/thesis_tipaldi.pdf
    An Unsupervised Conditional Random Fields Approach for Clustering Gene Expression Time Series
    http://bioinformatics.oxfordjournals.org/content/24/21/2467.full


CRFs:
    http://homepages.inf.ed.ac.uk/csutton/publications/crftutv2.pdf

AlphaBeta Swap:
    https://github.com/amueller/gco_python
    https://github.com/pmneila/PyMaxflow
    http://www.cs.cornell.edu/rdz/papers/bvz-iccv99.pdf

    http://arxiv.org/pdf/1411.6340.pdf  Iteratively Reweighted Graph Cut for Multi-label MRFs with Non-convex Priors

Fusion Moves:
    http://www.robots.ox.ac.uk/~vilem/fusion.pdf
    http://hci.iwr.uni-heidelberg.de/publications/mip/techrep/beier_15_fusion.pdf

Consensus Clustering

Explaining Away

Course Notes:
    Tie breaking for MAP assignment.
    https://class.coursera.org/pgm-003/lecture/60
    * random perdibiation

    Correspondence Problem is discussed in
    https://class.coursera.org/pgm-003/lecture/68

    Sparse Pattern Factors

    Collective Inference:
    Plate Models / Aggragator CPD is used to define dependencies.


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
import numpy as np
from six.moves import zip
import pgmpy
import pgmpy.inference
import pgmpy.factors
import pgmpy.models
from ibeis.model.hots import pgm_ext
print, rrr, profile = ut.inject2(__name__, '[bayes]')

SPECIAL_BASIS_POOL = ['fred', 'sue', 'paul']


def test_model(num_annots, num_names, score_evidence=[], name_evidence=[],
               other_evidence={}, show_prior=False, noquery=False, **kwargs):
    verbose = ut.VERBOSE

    model = make_name_model(num_annots, num_names, verbose=verbose, **kwargs)

    if verbose:
        model.print_priors(ignore_ttypes=['match'])
        #ut.colorprint('\n --- Priors ---', 'darkblue')
        #for ttype, cpds in model.ttype2_cpds.items():
        #    if ttype != 'match':
        #        for fs_ in ut.ichunks(cpds, 4):
        #            ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]), 'purple')

    model, evidence, soft_evidence = update_model_evidence(
        model, name_evidence, score_evidence, other_evidence)

    if verbose:
        ut.colorprint('\n --- Soft Evidence ---', 'white')
        for ttype, cpds in model.ttype2_cpds.items():
            if ttype != 'match':
                for fs_ in ut.ichunks(cpds, 4):
                    ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]),
                                  'green')

    if verbose:
        ut.colorprint('\n --- Inference ---', 'red')

    if (len(evidence) > 0 or len(soft_evidence) > 0) and not noquery:
        interest_ttypes = ['name']
        infr = pgmpy.inference.VariableElimination(model)
        #infr = pgmpy.inference.BeliefPropagation(model)
        evidence = infr._ensure_internal_evidence(evidence, model)
        query_results = try_query(
            model, infr, evidence, interest_ttypes, verbose=verbose)
    else:
        query_results = {}

    factor_list = query_results['factor_list']

    if verbose:
        if verbose:
            print('+--------')
        semtypes = [model.var2_cpd[f.variables[0]].ttype
                    for f in factor_list]
        for type_, factors in ut.group_items(factor_list, semtypes).items():
            print('Result Factors (%r)' % (type_,))
            factors = ut.sortedby(factors, [f.variables[0] for f in factors])
            for fs_ in ut.ichunks(factors, 4):
                ut.colorprint(ut.hz_str([f._str('phi', 'psql') for f in fs_]),
                              'yellow')
        print('MAP assignments')
        top_assignments = query_results.get('top_assignments', [])
        tmp = []
        for lbl, val in top_assignments:
            tmp.append('%s : %.4f' % (ut.repr2(lbl), val))
        print(ut.align('\n'.join(tmp), ' :'))

        #print('Joint Factors')
        #for ttype, marginal in marginalized_joints.items():
        #    print('Marginal Joint %s Factors' % (ttype,))
        #    ut.colorprint(marginal._str('phi', 'psql', sort=-1, maxrows=4), 'white')
        print('L_____\n')

    showkw = dict(evidence=evidence,
                  soft_evidence=soft_evidence,
                  show_prior=show_prior,
                  **query_results)

    show_model(model, **showkw)
    return (model, evidence)
    # print_ascii_graph(model)


def name_model_mode5(num_annots, num_names=None, verbose=True, mode=1):
    mode = ut.get_argval('--mode', default=mode)
    annots = ut.chr_range(num_annots, base=ut.get_argval('--base', default='a'))
    # The indexes of match CPDs will not change if another annotation is added
    upper_diag_idxs = ut.colwise_diag_idxs(num_annots, 2)
    if num_names is None:
        num_names = num_annots

    # -- Define CPD Templates

    name_cpd_t = pgm_ext.TemplateCPD(
        'name', ('n', num_names), varpref='N',
        special_basis_pool=SPECIAL_BASIS_POOL)
    name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]

    def match_pmf(match_type, n1, n2):
        return {
            True: {'same': 1.0, 'diff': 0.0},
            False: {'same': 0.0, 'diff': 1.0},
        }[n1 == n2][match_type]
    match_cpd_t = pgm_ext.TemplateCPD(
        'match', ['diff', 'same'], varpref='M',
        evidence_ttypes=[name_cpd_t, name_cpd_t], pmf_func=match_pmf)
    namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
    match_cpds = [match_cpd_t.new_cpd(parents=cpds)
                  for cpds in namepair_cpds]

    def trimatch_pmf(match_ab, match_bc, match_ca):
        lookup = {'same': {'same': {'same': 1, 'diff': 0, },
                           'diff': {'same': 0, 'diff': 1, }, },
                  'diff': {'same': {'same': 0, 'diff': 1, },
                           'diff': {'same': .5, 'diff': .5, }, } }
        return lookup[match_ca][match_bc][match_ab]
    trimatch_cpd_t = pgm_ext.TemplateCPD(
        'tri_match', ['diff', 'same'], varpref='T',
        evidence_ttypes=[match_cpd_t, match_cpd_t],
        pmf_func=trimatch_pmf)
    #triple_idxs = ut.colwise_diag_idxs(num_annots, 3)
    tid2_match = {cpd._template_id: cpd for cpd in match_cpds}
    trimatch_cpds = []
    # such hack
    for cpd in match_cpds:
        parents = []
        this_ = list(cpd._template_id)
        for aid in annots:
            if aid in this_:
                continue
            for aid2 in this_:
                key = aid2 + aid
                if key not in tid2_match:
                    key = aid + aid2
                parents += [tid2_match[key]]
        trimatch_cpds += [trimatch_cpd_t.new_cpd(parents=parents)]

    def score_pmf(score_type, match_type):
        score_lookup = {
            'same': {'low': .1, 'high': .9, 'veryhigh': .9},
            'diff': {'low': .9, 'high': .09, 'veryhigh': .01}
        }
        val = score_lookup[match_type][score_type]
        return val
    score_cpd_t = pgm_ext.TemplateCPD(
        'score', ['low', 'high'],
        varpref='S',
        evidence_ttypes=[match_cpd_t], pmf_func=score_pmf)
    score_cpds = [score_cpd_t.new_cpd(parents=cpds)
                  for cpds in zip(match_cpds)]

    #score_cpds = [score_cpd_t.new_cpd(parents=cpds)
    #              for cpds in zip(trimatch_cpds)]

    cpd_list = name_cpds + score_cpds + match_cpds + trimatch_cpds
    print('score_cpds = %r' % (ut.list_getattr(score_cpds, 'variable'),))

    # Make Model
    model = pgm_ext.define_model(cpd_list)
    model.num_names = num_names

    if verbose:
        model.print_templates()
    return model


def name_model_mode1(num_annots, num_names=None, verbose=True):
    r"""
    spaghettii

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-name_model_mode1 --show
        python -m ibeis.model.hots.bayes --exec-name_model_mode1
        python -m ibeis.model.hots.bayes --exec-name_model_mode1 --num-annots=3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> defaults = dict(num_annots=2, num_names=2, verbose=True)
        >>> kw = ut.argparse_funckw(name_model_mode1, defaults)
        >>> model = name_model_mode1(**kw)
        >>> ut.quit_if_noshow()
        >>> show_model(model, show_prior=False, show_title=False)
        >>> ut.show_if_requested()

    Ignore:
        import nx2tikz
        print(nx2tikz.dumps_tikz(model, layout='layered', use_label=True))
    """
    annots = ut.chr_range(num_annots, base=ut.get_argval('--base', default='a'))
    # The indexes of match CPDs will not change if another annotation is added
    upper_diag_idxs = ut.colwise_diag_idxs(num_annots, 2)
    if num_names is None:
        num_names = num_annots

    # +--- Define CPD Templates ---

    # +-- Name Factor ---
    name_cpd_t = pgm_ext.TemplateCPD(
        'name', ('n', num_names), varpref='N',
        special_basis_pool=SPECIAL_BASIS_POOL)
    name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]

    # +-- Match Factor ---
    def match_pmf(match_type, n1, n2):
        return {
            True: {'same': 1.0, 'diff': 0.0},
            False: {'same': 0.0, 'diff': 1.0},
        }[n1 == n2][match_type]
    match_cpd_t = pgm_ext.TemplateCPD(
        'match', ['diff', 'same'], varpref='M',
        evidence_ttypes=[name_cpd_t, name_cpd_t], pmf_func=match_pmf)
    namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
    match_cpds = [match_cpd_t.new_cpd(parents=cpds)
                  for cpds in namepair_cpds]

    # +-- Score Factor ---
    def score_pmf(score_type, match_type):
        score_lookup = {
            'same': {'low': .1, 'high': .9, 'veryhigh': .9},
            'diff': {'low': .9, 'high': .09, 'veryhigh': .01}
        }
        val = score_lookup[match_type][score_type]
        return val
    score_cpd_t = pgm_ext.TemplateCPD(
        'score', ['low', 'high'],
        varpref='S',
        evidence_ttypes=[match_cpd_t], pmf_func=score_pmf)
    score_cpds = [score_cpd_t.new_cpd(parents=cpds)
                  for cpds in zip(match_cpds)]

    # L___ End CPD Definitions ___

    cpd_list = name_cpds + score_cpds + match_cpds
    print('score_cpds = %r' % (ut.list_getattr(score_cpds, 'variable'),))

    # Make Model
    model = pgm_ext.define_model(cpd_list)
    model.num_names = num_names

    if verbose:
        model.print_templates()
    return model


def make_name_model(num_annots, num_names=None, verbose=True, mode=1):
    """
    Defines the general name model

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-make_name_model --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> defaults = dict(num_annots=2, num_names=2, verbose=True, mode=2)
        >>> kw = ut.argparse_funckw(make_name_model, defaults)
        >>> model = make_name_model(**kw)
        >>> ut.quit_if_noshow()
        >>> show_model(model, show_prior=True)
        >>> ut.show_if_requested()
    """
    #annots = ut.chr_range(num_annots, base='a')
    mode = ut.get_argval('--mode', default=mode)
    annots = ut.chr_range(num_annots, base=ut.get_argval('--base', default='a'))
    # The indexes of match CPDs will not change if another annotation is added
    upper_diag_idxs = ut.colwise_diag_idxs(num_annots, 2)
    if num_names is None:
        num_names = num_annots

    # -- Define CPD Templates
    def match_pmf(match_type, n1, n2):
        if n1 == n2:
            val = 1.0 if match_type == 'same' else 0.0
            #val = .999 if match_type == 'same' else 0.001
        elif n1 != n2:
            #val = 0.01 if match_type == 'same' else .99
            val = 0.0 if match_type == 'same' else 1.0
        return val

    def score_pmf(score_type, match_type):
        score_lookup = {
            'same': {'low': .1, 'high': .9, 'veryhigh': .9},
            'diff': {'low': .9, 'high': .09, 'veryhigh': .01}
            #'same': {'low': .1, 'high': .9},
            #'diff': {'low': .9, 'high': .1}
        }
        val = score_lookup[match_type][score_type]
        return val

    def score_pmf3(score_type, match_type, isdup='False'):
        score_lookup = {
            'False': {
                'same': {'low': .1, 'high': .5, 'veryhigh': .4},
                'diff': {'low': .9, 'high': .09, 'veryhigh': .01}
            },
            'True': {
                'same': {'low': .01, 'high': .2, 'veryhigh': .79},
                'diff': {'low': .4, 'high': .4, 'veryhigh': .2}
            }
        }
        val = score_lookup[isdup][match_type][score_type]
        return val

    def score_pmf2(score_type, n1, n2):
        score_lookup = {
            True: {'low': .1, 'high': .4, 'veryhigh': .5},
            False: {'low': .9, 'high': .09, 'veryhigh': .01}
        }
        val = score_lookup[n1 == n2][score_type]
        return val

    def dup_pmf(dupstate, match_type):
        lookup = {
            'same': {'True': 0.5, 'False': 0.5},
            'diff': {'True': 0.0, 'False': 1.0}
        }
        return lookup[match_type][dupstate]

    def check_pmf(n0, n1, match_type):
        pass

    def trimatch_pmf(match_ab, match_bc, match_ca):
        lookup = {
            'same': {
                'same': {'same': 1, 'diff': 0, },
                'diff': {'same': 0, 'diff': 1, }
            },
            'diff': {
                'same': {'same': 0, 'diff': 1, },
                'diff': {'same': .5, 'diff': .5, }
            }
        }
        return lookup[match_ca][match_bc][match_ab]

    name_cpd_t = pgm_ext.TemplateCPD(
        'name', ('n', num_names), varpref='N',
        special_basis_pool=SPECIAL_BASIS_POOL)

    if mode == 1 or mode == 5:
        match_cpd_t = pgm_ext.TemplateCPD(
            'match', ['diff', 'same'], varpref='M',
            evidence_ttypes=[name_cpd_t, name_cpd_t], pmf_func=match_pmf)

        if mode == 5:
            trimatch_cpd_t = pgm_ext.TemplateCPD(
                'tri_match', ['diff', 'same'], varpref='T',
                #evidence_ttypes=[match_cpd_t, match_cpd_t, match_cpd_t],
                evidence_ttypes=[match_cpd_t, match_cpd_t],
                pmf_func=trimatch_pmf)

            score_cpd_t = pgm_ext.TemplateCPD(
                #'score', ['low', 'high', 'veryhigh'],
                'score', ['low', 'high'],
                varpref='S',
                evidence_ttypes=[match_cpd_t], pmf_func=score_pmf)
        else:
            score_cpd_t = pgm_ext.TemplateCPD(
                #'score', ['low', 'high', 'veryhigh'],
                'score', ['low', 'high'],
                varpref='S',
                evidence_ttypes=[match_cpd_t], pmf_func=score_pmf)

    elif mode == 2:
        name_cpd_t = pgm_ext.TemplateCPD(
            'name', ('n', num_names), varpref='N',
            special_basis_pool=SPECIAL_BASIS_POOL)
        score_cpd_t = pgm_ext.TemplateCPD(
            #'score', ['low', 'high', 'veryhigh'],
            'score', ['low', 'high'],
            varpref='S',
            evidence_ttypes=[name_cpd_t, name_cpd_t],
            pmf_func=score_pmf2)
    elif mode == 3 or mode == 4:
        match_cpd_t = pgm_ext.TemplateCPD(
            'match', ['diff', 'same'], varpref='M',
            evidence_ttypes=[name_cpd_t, name_cpd_t], pmf_func=match_pmf)
        if mode == 3:
            dup_cpd_t = pgm_ext.TemplateCPD(
                'dup', ['False', 'True'], varpref='D',
            )
        else:
            dup_cpd_t = pgm_ext.TemplateCPD(
                'dup', ['False', 'True'], varpref='D',
                evidence_ttypes=[match_cpd_t], pmf_func=dup_pmf
            )
        score_cpd_t = pgm_ext.TemplateCPD(
            'score', ['low', 'high', 'veryhigh'], varpref='S',
            evidence_ttypes=[match_cpd_t, dup_cpd_t], pmf_func=score_pmf3)

    # Instanciate templates

    if mode == 1 or mode == 5:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        match_cpds = [match_cpd_t.new_cpd(parents=cpds)
                      for cpds in namepair_cpds]
        score_cpds = [score_cpd_t.new_cpd(parents=cpds)
                      for cpds in zip(match_cpds)]
        if mode == 5:
            #triple_idxs = ut.colwise_diag_idxs(num_annots, 3)
            tid2_match = {cpd._template_id: cpd for cpd in match_cpds}
            trimatch_cpds = []
            # such hack
            for cpd in match_cpds:
                parents = []
                this_ = list(cpd._template_id)
                for aid in annots:
                    if aid in this_:
                        continue
                    for aid2 in this_:
                        key = aid2 + aid
                        if key not in tid2_match:
                            key = aid + aid2
                        parents += [tid2_match[key]]
                trimatch_cpds += [trimatch_cpd_t.new_cpd(parents=parents)]

            #score_cpds = [score_cpd_t.new_cpd(parents=cpds)
            #              for cpds in zip(trimatch_cpds)]

            cpd_list = name_cpds + score_cpds + match_cpds + trimatch_cpds
        else:
            cpd_list = name_cpds + score_cpds + match_cpds
    elif mode == 2:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        score_cpds = [score_cpd_t.new_cpd(parents=cpds)
                      for cpds in namepair_cpds]
        cpd_list = name_cpds + score_cpds
    elif mode == 3 or mode == 4:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        match_cpds = [match_cpd_t.new_cpd(parents=cpds)
                      for cpds in namepair_cpds]
        if mode == 3:
            dup_cpds = [dup_cpd_t.new_cpd(parents=''.join(map(str, aids))) for aids
                        in ut.list_unflat_take(annots, upper_diag_idxs)]
        else:
            dup_cpds = [dup_cpd_t.new_cpd(parents=[mcpds]) for mcpds
                        in match_cpds]
        score_cpds = [score_cpd_t.new_cpd(parents=([mcpds] + [dcpd]))
                      for mcpds, dcpd in zip(match_cpds, dup_cpds)]
        cpd_list = name_cpds + score_cpds + match_cpds + dup_cpds

    #print('upper_diag_idxs = %r' % (upper_diag_idxs,))
    print('score_cpds = %r' % (ut.list_getattr(score_cpds, 'variable'),))
    # import sys
    # sys.exit(1)

    # Make Model
    model = pgm_ext.define_model(cpd_list)
    model.num_names = num_names

    if verbose:
        model.print_templates()
        #ut.colorprint('\n --- CPD Templates ---', 'blue')
        #for temp_cpd in templates:
        #    ut.colorprint(temp_cpd._cpdstr('psql'), 'turquoise')
    #print_ascii_graph(model)
    return model


def update_model_evidence(model, name_evidence, score_evidence, other_evidence):
    r"""

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-update_model_evidence

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> verbose = True
        >>> other_evidence = {}
        >>> name_evidence = [0, 0, 1, 1, None]
        >>> score_evidence = ['high', 'low', 'low', 'low', 'low', 'high']
        >>> model = make_name_model(num_annots=5, num_names=3, verbose=True, mode=1)
        >>> model, evidence, soft_evidence = update_model_evidence(model, name_evidence, score_evidence, other_evidence)
    """
    name_cpds = model.ttype2_cpds['name']
    score_cpds = model.ttype2_cpds['score']

    evidence = {}
    evidence.update(other_evidence)
    soft_evidence = {}

    def apply_hard_soft_evidence(cpd_list, evidence_list):
        for cpd, ev in zip(cpd_list, evidence_list):
            if isinstance(ev, int):
                # hard internal evidence
                evidence[cpd.variable] = ev
            if isinstance(ev, six.string_types):
                # hard external evidence
                evidence[cpd.variable] = cpd._internal_varindex(
                    cpd.variable, ev)
            if isinstance(ev, dict):
                # soft external evidence
                # HACK THAT MODIFIES CPD IN PLACE
                def rectify_evidence_val(_v, card=cpd.variable_card):
                    # rectify hacky string structures
                    #tmp = 1
                    tmp = (1 / (2 * card ** 2))
                    return (1 + tmp) / (card + tmp) if _v == '+eps' else _v
                ev_ = ut.map_dict_vals(rectify_evidence_val, ev)
                fill = (1 - sum(ev_.values())) / (cpd.variable_card - len(ev_))
                assert fill >= 0
                row_labels = list(ut.iprod(*cpd.statenames))

                for i, lbl in enumerate(row_labels):
                    if lbl in ev_:
                        # external case1
                        cpd.values[i] = ev_[lbl]
                    elif len(lbl) == 1 and lbl[0] in ev_:
                        # external case2
                        cpd.values[i] = ev_[lbl[0]]
                    elif i in ev_:
                        # internal case
                        cpd.values[i] = ev_[i]
                    else:
                        cpd.values[i] = fill
                soft_evidence[cpd.variable] = True

    apply_hard_soft_evidence(name_cpds, name_evidence)
    apply_hard_soft_evidence(score_cpds, score_evidence)
    return model, evidence, soft_evidence


def try_query(model, infr, evidence, interest_ttypes=[], verbose=True):
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes --exec-try_query --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> verbose = True
        >>> other_evidence = {}
        >>> name_evidence = [0, 0, 1, 1, None]
        >>> score_evidence = ['high', 'low', 'low', 'low', 'low', 'high']
        >>> model = make_name_model(num_annots=5, num_names=3, verbose=True, mode=1)
        >>> model, evidence, soft_evidence = update_model_evidence(model, name_evidence, score_evidence, other_evidence)
        >>> interest_ttypes = ['name']
        >>> infr = pgmpy.inference.BeliefPropagation(model)
        >>> evidence = infr._ensure_internal_evidence(evidence, model)
        >>> query_results = try_query(model, infr, evidence, interest_ttypes, verbose)
        >>> result = ('query_results = %s' % (str(query_results),))
        >>> ut.quit_if_noshow()
        >>> show_model(model, show_prior=True, **query_results)
        >>> ut.show_if_requested()

    Ignore:
        query_vars = ut.setdiff_ordered(model.nodes(), list(evidence.keys()))
        probs = infr.query(query_vars, evidence)
        map_assignment = infr.map_query(query_vars, evidence)
    """
    import vtool as vt
    query_vars = ut.setdiff_ordered(model.nodes(), list(evidence.keys()))
    # hack
    query_vars = ut.setdiff_ordered(query_vars, ut.list_getattr(model.ttype2_cpds['score'], 'variable'))
    if verbose:
        evidence_str = ', '.join(model.pretty_evidence(evidence))
        print('P(' + ', '.join(query_vars) + ' | ' + evidence_str + ') = ')
    # Compute all marginals
    probs = infr.query(query_vars, evidence)
    #probs = infr.query(query_vars, evidence)
    factor_list = probs.values()
    # Compute MAP joints
    # There is a bug here.
    #map_assign = infr.map_query(query_vars, evidence)
    # (probably an invalid thing to do)
    #joint_factor = pgmpy.factors.factor_product(*factor_list)
    # Brute force MAP
    query_vars2 = ut.list_getattr(model.ttype2_cpds['name'], 'variable')
    query_vars2 = ut.setdiff_ordered(query_vars2, list(evidence.keys()))
    # TODO: incorporate case where Na is assigned to Fred
    #evidence_h = ut.delete_keys(evidence.copy(), ['Na'])
    joint = model.joint_distribution()
    joint.evidence_based_reduction(
        query_vars2, evidence, inplace=True)
    # Relabel rows based on the knowledge that everything
    # is the same, only the names have changed.
    new_rows = joint._row_labels()
    new_vals = joint.values.ravel()
    # HACK
    new_rows = [('fred',) + x for x in new_rows]
    #new_vals = [('fred',) + x for x in new_rows]
    cpd_t = model.ttype2_cpds['name'][0]._template_
    basis = cpd_t.basis
    def relabel_names(names, basis=basis):
        names = list(map(six.text_type, names))
        mapping = {}
        for n in names:
            if n not in mapping:
                mapping[n] = len(mapping)
        new_names = tuple([basis[mapping[n]] for n in names])
        return new_names
    relabeled_rows = list(map(relabel_names, new_rows))
    import utool
    with utool.embed_on_exception_context:
        data_ids = np.array(vt.other.compute_unique_data_ids_(relabeled_rows))
        unique_ids, groupxs = vt.group_indices(data_ids)
        reduced_row_lbls = ut.take(relabeled_rows, ut.get_list_column(groupxs, 0))
        reduced_values = np.array([
            g.sum() for g in vt.apply_grouping(new_vals, groupxs)
        ])
    sortx = reduced_values.argsort()[::-1]
    reduced_row_lbls = ut.take(reduced_row_lbls, sortx.tolist())
    reduced_values = reduced_values[sortx]
    # Better map assignment based on knowledge of labels
    map_assign = reduced_row_lbls[0]

    reduced_row_lbls = [','.join(x) for x in reduced_row_lbls]

    top_assignments = list(zip(reduced_row_lbls[:3], reduced_values))
    if len(reduced_values) > 3:
        top_assignments += [('other', 1 - sum(reduced_values[:3]))]

    #map_assign = joint.map_bruteforce(query_vars, evidence)
    #joint.reduce(evidence)
    ## Marginalize over non-query, non-evidence
    #irrelevant_vars = ut.setdiff_ordered(joint.variables, list(evidence.keys()) + query_vars)
    #joint.marginalize(irrelevant_vars)
    #joint.normalize()
    #new_rows = joint._row_labels()
    #new_vals = joint.values.ravel()
    #map_vals = new_rows[new_vals.argmax()]
    #map_assign = dict(zip(joint.variables, map_vals))
    # Compute Marginalized MAP joints
    #marginalized_joints = {}
    #for ttype in interest_ttypes:
    #    other_vars = [v for v in joint_factor.scope()
    #                  if model.var2_cpd[v].ttype != ttype]
    #    marginal = joint_factor.marginalize(other_vars, inplace=False)
    #    marginalized_joints[ttype] = marginal
    query_results = {
        'factor_list': factor_list,
        'top_assignments': top_assignments,
        'map_assign': map_assign,
        'marginalized_joints': None,
    }
    return query_results


def draw_tree_model(model, **kwargs):
    import plottool as pt
    import networkx as netx
    if not ut.get_argval('--hackjunc'):
        fnum = pt.ensure_fnum(None)
        fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
        ax = pt.gca()
        #name_nodes = sorted(ut.list_getattr(model.ttype2_cpds['name'], 'variable'))
        netx_graph = model.to_markov_model()
        #pos = netx.pygraphviz_layout(netx_graph)
        #pos = netx.graphviz_layout(netx_graph)
        #pos = get_hacked_pos(netx_graph, name_nodes, prog='neato')
        pos = netx.pydot_layout(netx_graph)
        node_color = [pt.WHITE] * len(pos)
        drawkw = dict(pos=pos, ax=ax, with_labels=True, node_color=node_color,
                      node_size=1100)
        netx.draw(netx_graph, **drawkw)
        if kwargs.get('show_title', True):
            pt.set_figtitle('Markov Model')

    if not ut.get_argval('--hackmarkov'):
        fnum = pt.ensure_fnum(None)
        fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
        ax = pt.gca()
        netx_graph = model.to_junction_tree()
        # prettify nodes
        def fixtupkeys(dict_):
            return {
                ', '.join(k) if isinstance(k, tuple) else k: fixtupkeys(v)
                for k, v in dict_.items()
            }
        n = fixtupkeys(netx_graph.node)
        e = fixtupkeys(netx_graph.edge)
        a = fixtupkeys(netx_graph.adj)
        netx_graph.node = n
        netx_graph.edge = e
        netx_graph.adj = a
        #netx_graph = model.to_markov_model()
        #pos = netx.pygraphviz_layout(netx_graph)
        #pos = netx.graphviz_layout(netx_graph)
        pos = netx.pydot_layout(netx_graph)
        node_color = [pt.WHITE] * len(pos)
        drawkw = dict(pos=pos, ax=ax, with_labels=True, node_color=node_color,
                      node_size=2000)
        netx.draw(netx_graph, **drawkw)
        if kwargs.get('show_title', True):
            pt.set_figtitle('Junction/Clique Tree / Cluster Graph')


def get_hacked_pos(netx_graph, name_nodes=None, prog='dot'):
    import pygraphviz
    import networkx as netx
    # Add "invisible" edges to induce an ordering
    # Hack for layout (ordering of top level nodes)
    netx_graph2 = netx_graph.copy()
    if getattr(netx_graph, 'ttype2_cpds', None) is not None:
        grouped_nodes = []
        for ttype in netx_graph.ttype2_cpds.keys():
            # if ttype not in ['match', 'name']:
            #     continue
            ttype_cpds = netx_graph.ttype2_cpds[ttype]
            # use defined ordering
            ttype_nodes = ut.list_getattr(ttype_cpds, 'variable')
            # ttype_nodes = sorted(ttype_nodes)
            invis_edges = list(ut.itertwo(ttype_nodes))
            netx_graph2.add_edges_from(invis_edges)
            grouped_nodes.append(ttype_nodes)

        A = netx.to_agraph(netx_graph2)
        for nodes in grouped_nodes:
            A.add_subgraph(nodes, rank='same')
    else:
        A = netx.to_agraph(netx_graph2)

    #if name_nodes is not None:
    #    #netx.set_node_attributes(netx_graph, 'label', {n: {'label': n} for n in all_nodes})
    #    invis_edges = list(ut.itertwo(name_nodes))
    #    netx_graph2.add_edges_from(invis_edges)
    #    A.add_subgraph(name_nodes, rank='same')
    #else:
    #    A = netx.to_agraph(netx_graph2)
    args = ''
    G = netx_graph
    A.layout(prog=prog, args=args)
    #A.draw('example.png', prog='dot')
    node_pos = {}
    for n in G:
        node_ = pygraphviz.Node(A, n)
        try:
            xx, yy = node_.attr["pos"].split(',')
            node_pos[n] = (float(xx), float(yy))
        except:
            print("no position for node", n)
            node_pos[n] = (0.0, 0.0)
    return node_pos


def show_model(model, evidence=None, soft_evidence={}, **kwargs):
    """
    References:
        http://stackoverflow.com/questions/22207802/pygraphviz-networkx-set-node-level-or-layer

    Ignore:
        pkg-config --libs-only-L libcgraph
        sudo apt-get  install libgraphviz-dev -y
        sudo apt-get  install libgraphviz4 -y

        # sudo apt-get install pkg-config
        sudo apt-get install libgraphviz-dev
        # pip install git+git://github.com/pygraphviz/pygraphviz.git
        pip install pygraphviz
        python -c "import pygraphviz; print(pygraphviz.__file__)"

        sudo pip3 install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
        python3 -c "import pygraphviz; print(pygraphviz.__file__)"
    """
    if ut.get_argval('--hackmarkov') or ut.get_argval('--hackjunc'):
        draw_tree_model(model, **kwargs)
        return

    import plottool as pt
    import networkx as netx
    import matplotlib as mpl
    fnum = pt.ensure_fnum(None)
    fig = pt.figure(fnum=fnum, pnum=(3, 1, (slice(0, 2), 0)), doclf=True)  # NOQA
    #fig = pt.figure(fnum=fnum, pnum=(3, 2, (1, slice(1, 2))), doclf=True)  # NOQA
    ax = pt.gca()
    var2_post = {f.variables[0]: f for f in kwargs.get('factor_list', [])}

    netx_graph = (model)
    #netx_graph.graph.setdefault('graph', {})['size'] = '"10,5"'
    #netx_graph.graph.setdefault('graph', {})['rankdir'] = 'LR'

    pos = get_hacked_pos(netx_graph)
    #netx.pygraphviz_layout(netx_graph)
    #pos = netx.pydot_layout(netx_graph, prog='dot')
    #pos = netx.graphviz_layout(netx_graph)

    drawkw = dict(pos=pos, ax=ax, with_labels=True, node_size=1500)
    if evidence is not None:
        node_colors = [
            # (pt.TRUE_BLUE
            (pt.WHITE
             if node not in soft_evidence else
             pt.LIGHT_PINK)
            if node not in evidence
            else pt.FALSE_RED
            for node in netx_graph.nodes()]

        for node in netx_graph.nodes():
            cpd = model.var2_cpd[node]
            if cpd.ttype == 'score':
                pass
        drawkw['node_color'] = node_colors

    netx.draw(netx_graph, **drawkw)

    if True:
        textprops = {
            'family': 'monospace',
            'horizontalalignment': 'left',
            #'horizontalalignment': 'center',
            #'size': 12,
            'size': 8,
        }

        textkw = dict(
            xycoords='data', boxcoords='offset points', pad=0.25,
            frameon=True, arrowprops=dict(arrowstyle='->'),
            #bboxprops=dict(fc=node_attr['fillcolor']),
        )

        netx_nodes = model.nodes(data=True)
        node_key_list = ut.get_list_column(netx_nodes, 0)
        pos_list = ut.dict_take(pos, node_key_list)

        artist_list = []
        offset_box_list = []
        for pos_, node in zip(pos_list, netx_nodes):
            x, y = pos_
            variable = node[0]

            cpd = model.var2_cpd[variable]

            prior_marg = (cpd if cpd.evidence is None else
                          cpd.marginalize(cpd.evidence, inplace=False))

            prior_text = None

            text = None
            if variable in evidence:
                text = cpd.variable_statenames[evidence[variable]]
            elif variable in var2_post:
                post_marg = var2_post[variable]
                text = pgm_ext.make_factor_text(post_marg, 'post_marginal')
                prior_text = pgm_ext.make_factor_text(prior_marg, 'prior_marginal')
            else:
                if len(evidence) == 0 and len(soft_evidence) == 0:
                    prior_text = pgm_ext.make_factor_text(prior_marg, 'prior_marginal')

            show_post = kwargs.get('show_post', False)
            show_prior = kwargs.get('show_prior', False)
            show_ev = (evidence is not None and variable in evidence)
            if (show_post or show_ev) and text is not None:
                offset_box = mpl.offsetbox.TextArea(text, textprops)
                artist = mpl.offsetbox.AnnotationBbox(
                    # offset_box, (x + 5, y), xybox=(20., 5.),
                    offset_box, (x, y + 5), xybox=(0., 20.),
                    #box_alignment=(0, 0),
                    box_alignment=(.5, 0),
                    **textkw)
                offset_box_list.append(offset_box)
                artist_list.append(artist)

            if show_prior and prior_text is not None:
                offset_box2 = mpl.offsetbox.TextArea(prior_text, textprops)
                artist2 = mpl.offsetbox.AnnotationBbox(
                    # offset_box2, (x - 5, y), xybox=(-20., -15.),
                    offset_box2, (x, y - 5), xybox=(-15., -20.),
                    #box_alignment=(1, 1),
                    box_alignment=(.5, 1),
                    **textkw)
                offset_box_list.append(offset_box2)
                artist_list.append(artist2)

        for artist in artist_list:
            ax.add_artist(artist)

        xmin, ymin = np.array(pos_list).min(axis=0)
        xmax, ymax = np.array(pos_list).max(axis=0)
        num_annots = len(model.ttype2_cpds['name'])
        if num_annots > 4:
            ax.set_xlim((xmin - 40, xmax + 40))
            ax.set_ylim((ymin - 50, ymax + 50))
            fig.set_size_inches(30, 7)
        else:
            ax.set_xlim((xmin - 42, xmax + 42))
            ax.set_ylim((ymin - 50, ymax + 50))
            fig.set_size_inches(23, 7)
        fig = pt.gcf()

        title = 'num_names=%r, num_annots=%r' % (model.num_names, num_annots,)
        map_assign = kwargs.get('map_assign', None)
        #max_marginal_list = []
        #for name, marginal in marginalized_joints.items():
        #    states = list(ut.iprod(*marginal.statenames))
        #    vals = marginal.values.ravel()
        #    x = vals.argmax()
        #    max_marginal_list += ['P(' + ', '.join(states[x]) + ') = ' + str(vals[x])]
        # title += str(marginal)
        if map_assign is not None:
            title += '\nMAP=' + ut.repr2(map_assign, strvals=True)
        if kwargs.get('show_title', True):
            pt.set_figtitle(title, size=14)
        #pt.set_xlabel()

        def hack_fix_centeralign():
            if textprops['horizontalalignment'] == 'center':
                print('Fixing centeralign')
                fig = pt.gcf()
                fig.canvas.draw()

                # Superhack for centered text. Fix bug in
                # /usr/local/lib/python2.7/dist-packages/matplotlib/offsetbox.py
                # /usr/local/lib/python2.7/dist-packages/matplotlib/text.py
                for offset_box in offset_box_list:
                    offset_box.set_offset
                    z = offset_box._text.get_window_extent()
                    (z.x1 - z.x0) / 2
                    offset_box._text
                    T = offset_box._text.get_transform()
                    A = mpl.transforms.Affine2D()
                    A.clear()
                    A.translate((z.x1 - z.x0) / 2, 0)
                    offset_box._text.set_transform(T + A)
        hack_fix_centeralign()
    top_assignments = kwargs.get('top_assignments', None)
    if top_assignments is not None:
        bin_labels = ut.get_list_column(top_assignments, 0)
        bin_vals =  ut.get_list_column(top_assignments, 1)
        pt.draw_histogram(bin_labels, bin_vals, fnum=fnum, pnum=(3, 8, (2, slice(1, None))),
                          transpose=True,
                          use_darkbackground=False,
                          #xtick_rotation=-10,
                          ylabel='Prob', xlabel='assignment')
        pt.set_title('Assignment probabilities')

    #fpath = ('name_model_' + suff + '.png')
    #pt.plt.savefig(fpath)
    #return fpath


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes
        python -m ibeis.model.hots.bayes --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
