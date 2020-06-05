# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import print_function, division, absolute_import
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)

"""

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


def try_query(model, infr, evidence, interest_ttypes=[], verbose=True):
    r"""
    CommandLine:
        python -m wbia.algo.hots.bayes --exec-try_query --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.bayes import *  # NOQA
        >>> verbose = True
        >>> other_evidence = {}
        >>> name_evidence = [1, None, 0, None]
        >>> score_evidence = ['high', 'low', 'low']
        >>> query_vars = None
        >>> model = make_name_model(num_annots=4, num_names=4, verbose=True, mode=1)
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
    infr = pgmpy.inference.VariableElimination(model)
    # infr = pgmpy.inference.BeliefPropagation(model)
    if True:
        return bruteforce(model, query_vars=None, evidence=evidence)
    else:
        import vtool as vt

        query_vars = ut.setdiff_ordered(model.nodes(), list(evidence.keys()))
        # hack
        query_vars = ut.setdiff_ordered(
            query_vars, ut.list_getattr(model.ttype2_cpds['score'], 'variable')
        )
        if verbose:
            evidence_str = ', '.join(model.pretty_evidence(evidence))
            print('P(' + ', '.join(query_vars) + ' | ' + evidence_str + ') = ')
        # Compute MAP joints
        # There is a bug here.
        # map_assign = infr.map_query(query_vars, evidence)
        # (probably an invalid thing to do)
        # joint_factor = pgmpy.factors.factor_product(*factor_list)
        # Brute force MAP

        name_vars = ut.list_getattr(model.ttype2_cpds['name'], 'variable')
        query_name_vars = ut.setdiff_ordered(name_vars, list(evidence.keys()))
        # TODO: incorporate case where Na is assigned to Fred
        # evidence_h = ut.delete_keys(evidence.copy(), ['Na'])

        joint = model.joint_distribution()
        joint.evidence_based_reduction(query_name_vars, evidence, inplace=True)

        # Find static row labels in the evidence
        given_name_vars = [var for var in name_vars if var in evidence]
        given_name_idx = ut.dict_take(evidence, given_name_vars)
        given_name_val = [
            joint.statename_dict[var][idx]
            for var, idx in zip(given_name_vars, given_name_idx)
        ]
        new_vals = joint.values.ravel()
        # Add static evidence variables to the relabeled name states
        new_vars = given_name_vars + joint.variables
        new_rows = [tuple(given_name_val) + row for row in joint._row_labels()]
        # Relabel rows based on the knowledge that
        # everything is the same, only the names have changed.
        temp_basis = [i for i in range(model.num_names)]

        def relabel_names(names, temp_basis=temp_basis):
            names = list(map(six.text_type, names))
            mapping = {}
            for n in names:
                if n not in mapping:
                    mapping[n] = len(mapping)
            new_names = tuple([temp_basis[mapping[n]] for n in names])
            return new_names

        relabeled_rows = list(map(relabel_names, new_rows))
        # Combine probability of rows with the same (new) label
        data_ids = np.array(vt.other.compute_unique_data_ids_(relabeled_rows))
        unique_ids, groupxs = vt.group_indices(data_ids)
        reduced_row_lbls = ut.take(relabeled_rows, ut.get_list_column(groupxs, 0))
        reduced_row_lbls = list(map(list, reduced_row_lbls))
        reduced_values = np.array([g.sum() for g in vt.apply_grouping(new_vals, groupxs)])
        # Relabel the rows one more time to agree with initial constraints
        used_ = []
        replaced = []
        for colx, (var, val) in enumerate(zip(given_name_vars, given_name_val)):
            # All columns must be the same for this labeling
            alias = reduced_row_lbls[0][colx]
            reduced_row_lbls = ut.list_replace(reduced_row_lbls, alias, val)
            replaced.append(alias)
            used_.append(val)
        basis = model.ttype2_cpds['name'][0]._template_.basis
        find_remain_ = ut.setdiff_ordered(temp_basis, replaced)
        repl_remain_ = ut.setdiff_ordered(basis, used_)
        for find, repl in zip(find_remain_, repl_remain_):
            reduced_row_lbls = ut.list_replace(reduced_row_lbls, find, repl)

        # Now find the most likely state
        sortx = reduced_values.argsort()[::-1]
        sort_reduced_row_lbls = ut.take(reduced_row_lbls, sortx.tolist())
        sort_reduced_values = reduced_values[sortx]

        # Remove evidence based labels
        new_vars_ = new_vars[len(given_name_vars) :]
        sort_reduced_row_lbls_ = ut.get_list_column(
            sort_reduced_row_lbls, slice(len(given_name_vars), None)
        )

        sort_reduced_row_lbls_[0]

        # hack into a new joint factor
        var_states = ut.lmap(ut.unique_ordered, zip(*sort_reduced_row_lbls_))
        statename_dict = dict(zip(new_vars, var_states))
        cardinality = ut.lmap(len, var_states)
        val_lookup = dict(
            zip(ut.lmap(tuple, sort_reduced_row_lbls_), sort_reduced_values)
        )
        values = np.zeros(np.prod(cardinality))
        for idx, state in enumerate(ut.iprod(*var_states)):
            if state in val_lookup:
                values[idx] = val_lookup[state]
        joint2 = pgmpy.factors.Factor(
            new_vars_, cardinality, values, statename_dict=statename_dict
        )
        print(joint2)
        max_marginals = {}
        for i, var in enumerate(query_name_vars):
            one_out = query_name_vars[:i] + query_name_vars[i + 1 :]
            max_marginals[var] = joint2.marginalize(one_out, inplace=False)
            # max_marginals[var] = joint2.maximize(one_out, inplace=False)
        print(joint2.marginalize(['Nb', 'Nc'], inplace=False))
        factor_list = max_marginals.values()

        # Better map assignment based on knowledge of labels
        map_assign = dict(zip(new_vars_, sort_reduced_row_lbls_[0]))

        sort_reduced_rowstr_lbls = [
            ut.repr2(
                dict(zip(new_vars, lbls)), explicit=True, nobraces=True, strvals=True
            )
            for lbls in sort_reduced_row_lbls_
        ]

        top_assignments = list(zip(sort_reduced_rowstr_lbls[:3], sort_reduced_values))
        if len(sort_reduced_values) > 3:
            top_assignments += [('other', 1 - sum(sort_reduced_values[:3]))]

        # import utool
        # utool.embed()

        # Compute all marginals
        # probs = infr.query(query_vars, evidence)
        # probs = infr.query(query_vars, evidence)
        # factor_list = probs.values()

        ## Marginalize over non-query, non-evidence
        # irrelevant_vars = ut.setdiff_ordered(joint.variables, list(evidence.keys()) + query_vars)
        # joint.marginalize(irrelevant_vars)
        # joint.normalize()
        # new_rows = joint._row_labels()
        # new_vals = joint.values.ravel()
        # map_vals = new_rows[new_vals.argmax()]
        # map_assign = dict(zip(joint.variables, map_vals))
        # Compute Marginalized MAP joints
        # marginalized_joints = {}
        # for ttype in interest_ttypes:
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


def name_model_mode5(num_annots, num_names=None, verbose=True, mode=1):
    mode = ut.get_argval('--mode', default=mode)
    annots = ut.chr_range(num_annots, base=ut.get_argval('--base', default='a'))
    # The indexes of match CPDs will not change if another annotation is added
    upper_diag_idxs = ut.colwise_diag_idxs(num_annots, 2)
    if num_names is None:
        num_names = num_annots

    # -- Define CPD Templates

    name_cpd_t = pgm_ext.TemplateCPD(
        'name', ('n', num_names), varpref='N', special_basis_pool=SPECIAL_BASIS_POOL
    )
    name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]

    def match_pmf(match_type, n1, n2):
        return {True: {'same': 1.0, 'diff': 0.0}, False: {'same': 0.0, 'diff': 1.0},}[
            n1 == n2
        ][match_type]

    match_cpd_t = pgm_ext.TemplateCPD(
        'match',
        ['diff', 'same'],
        varpref='M',
        evidence_ttypes=[name_cpd_t, name_cpd_t],
        pmf_func=match_pmf,
    )
    namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
    match_cpds = [match_cpd_t.new_cpd(parents=cpds) for cpds in namepair_cpds]

    def trimatch_pmf(match_ab, match_bc, match_ca):
        lookup = {
            'same': {'same': {'same': 1, 'diff': 0,}, 'diff': {'same': 0, 'diff': 1,},},
            'diff': {
                'same': {'same': 0, 'diff': 1,},
                'diff': {'same': 0.5, 'diff': 0.5,},
            },
        }
        return lookup[match_ca][match_bc][match_ab]

    trimatch_cpd_t = pgm_ext.TemplateCPD(
        'tri_match',
        ['diff', 'same'],
        varpref='T',
        evidence_ttypes=[match_cpd_t, match_cpd_t],
        pmf_func=trimatch_pmf,
    )
    # triple_idxs = ut.colwise_diag_idxs(num_annots, 3)
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
            'same': {'low': 0.1, 'high': 0.9, 'veryhigh': 0.9},
            'diff': {'low': 0.9, 'high': 0.09, 'veryhigh': 0.01},
        }
        val = score_lookup[match_type][score_type]
        return val

    score_cpd_t = pgm_ext.TemplateCPD(
        'score',
        ['low', 'high'],
        varpref='S',
        evidence_ttypes=[match_cpd_t],
        pmf_func=score_pmf,
    )
    score_cpds = [score_cpd_t.new_cpd(parents=cpds) for cpds in zip(match_cpds)]

    # score_cpds = [score_cpd_t.new_cpd(parents=cpds)
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
        python -m wbia.algo.hots.bayes --exec-name_model_mode1 --show
        python -m wbia.algo.hots.bayes --exec-name_model_mode1
        python -m wbia.algo.hots.bayes --exec-name_model_mode1 --num-annots=3

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.bayes import *  # NOQA
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
        'name', ('n', num_names), varpref='N', special_basis_pool=SPECIAL_BASIS_POOL
    )
    name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]

    # +-- Match Factor ---
    def match_pmf(match_type, n1, n2):
        return {True: {'same': 1.0, 'diff': 0.0}, False: {'same': 0.0, 'diff': 1.0},}[
            n1 == n2
        ][match_type]

    match_cpd_t = pgm_ext.TemplateCPD(
        'match',
        ['diff', 'same'],
        varpref='M',
        evidence_ttypes=[name_cpd_t, name_cpd_t],
        pmf_func=match_pmf,
    )
    namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
    match_cpds = [match_cpd_t.new_cpd(parents=cpds) for cpds in namepair_cpds]

    # +-- Score Factor ---
    def score_pmf(score_type, match_type):
        score_lookup = {
            'same': {'low': 0.1, 'high': 0.9, 'veryhigh': 0.9},
            'diff': {'low': 0.9, 'high': 0.09, 'veryhigh': 0.01},
        }
        val = score_lookup[match_type][score_type]
        return val

    score_cpd_t = pgm_ext.TemplateCPD(
        'score',
        ['low', 'high'],
        varpref='S',
        evidence_ttypes=[match_cpd_t],
        pmf_func=score_pmf,
    )
    score_cpds = [score_cpd_t.new_cpd(parents=cpds) for cpds in zip(match_cpds)]

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
        python -m wbia.algo.hots.bayes --exec-make_name_model --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.bayes import *  # NOQA
        >>> defaults = dict(num_annots=2, num_names=2, verbose=True, mode=2)
        >>> kw = ut.argparse_funckw(make_name_model, defaults)
        >>> model = make_name_model(**kw)
        >>> ut.quit_if_noshow()
        >>> show_model(model, show_prior=True)
        >>> ut.show_if_requested()
    """
    # annots = ut.chr_range(num_annots, base='a')
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
            # val = .999 if match_type == 'same' else 0.001
        elif n1 != n2:
            # val = 0.01 if match_type == 'same' else .99
            val = 0.0 if match_type == 'same' else 1.0
        return val

    def score_pmf(score_type, match_type):
        score_lookup = {
            'same': {'low': 0.1, 'high': 0.9, 'veryhigh': 0.9},
            'diff': {'low': 0.9, 'high': 0.09, 'veryhigh': 0.01}
            #'same': {'low': .1, 'high': .9},
            #'diff': {'low': .9, 'high': .1}
        }
        val = score_lookup[match_type][score_type]
        return val

    def score_pmf3(score_type, match_type, isdup='False'):
        score_lookup = {
            'False': {
                'same': {'low': 0.1, 'high': 0.5, 'veryhigh': 0.4},
                'diff': {'low': 0.9, 'high': 0.09, 'veryhigh': 0.01},
            },
            'True': {
                'same': {'low': 0.01, 'high': 0.2, 'veryhigh': 0.79},
                'diff': {'low': 0.4, 'high': 0.4, 'veryhigh': 0.2},
            },
        }
        val = score_lookup[isdup][match_type][score_type]
        return val

    def score_pmf2(score_type, n1, n2):
        score_lookup = {
            True: {'low': 0.1, 'high': 0.4, 'veryhigh': 0.5},
            False: {'low': 0.9, 'high': 0.09, 'veryhigh': 0.01},
        }
        val = score_lookup[n1 == n2][score_type]
        return val

    def dup_pmf(dupstate, match_type):
        lookup = {
            'same': {'True': 0.5, 'False': 0.5},
            'diff': {'True': 0.0, 'False': 1.0},
        }
        return lookup[match_type][dupstate]

    def check_pmf(n0, n1, match_type):
        pass

    def trimatch_pmf(match_ab, match_bc, match_ca):
        lookup = {
            'same': {'same': {'same': 1, 'diff': 0,}, 'diff': {'same': 0, 'diff': 1,}},
            'diff': {
                'same': {'same': 0, 'diff': 1,},
                'diff': {'same': 0.5, 'diff': 0.5,},
            },
        }
        return lookup[match_ca][match_bc][match_ab]

    name_cpd_t = pgm_ext.TemplateCPD(
        'name', ('n', num_names), varpref='N', special_basis_pool=SPECIAL_BASIS_POOL
    )

    if mode == 1 or mode == 5:
        match_cpd_t = pgm_ext.TemplateCPD(
            'match',
            ['diff', 'same'],
            varpref='M',
            evidence_ttypes=[name_cpd_t, name_cpd_t],
            pmf_func=match_pmf,
        )

        if mode == 5:
            trimatch_cpd_t = pgm_ext.TemplateCPD(
                'tri_match',
                ['diff', 'same'],
                varpref='T',
                # evidence_ttypes=[match_cpd_t, match_cpd_t, match_cpd_t],
                evidence_ttypes=[match_cpd_t, match_cpd_t],
                pmf_func=trimatch_pmf,
            )

            score_cpd_t = pgm_ext.TemplateCPD(
                #'score', ['low', 'high', 'veryhigh'],
                'score',
                ['low', 'high'],
                varpref='S',
                evidence_ttypes=[match_cpd_t],
                pmf_func=score_pmf,
            )
        else:
            score_cpd_t = pgm_ext.TemplateCPD(
                #'score', ['low', 'high', 'veryhigh'],
                'score',
                ['low', 'high'],
                varpref='S',
                evidence_ttypes=[match_cpd_t],
                pmf_func=score_pmf,
            )

    elif mode == 2:
        name_cpd_t = pgm_ext.TemplateCPD(
            'name', ('n', num_names), varpref='N', special_basis_pool=SPECIAL_BASIS_POOL
        )
        score_cpd_t = pgm_ext.TemplateCPD(
            #'score', ['low', 'high', 'veryhigh'],
            'score',
            ['low', 'high'],
            varpref='S',
            evidence_ttypes=[name_cpd_t, name_cpd_t],
            pmf_func=score_pmf2,
        )
    elif mode == 3 or mode == 4:
        match_cpd_t = pgm_ext.TemplateCPD(
            'match',
            ['diff', 'same'],
            varpref='M',
            evidence_ttypes=[name_cpd_t, name_cpd_t],
            pmf_func=match_pmf,
        )
        if mode == 3:
            dup_cpd_t = pgm_ext.TemplateCPD('dup', ['False', 'True'], varpref='D',)
        else:
            dup_cpd_t = pgm_ext.TemplateCPD(
                'dup',
                ['False', 'True'],
                varpref='D',
                evidence_ttypes=[match_cpd_t],
                pmf_func=dup_pmf,
            )
        score_cpd_t = pgm_ext.TemplateCPD(
            'score',
            ['low', 'high', 'veryhigh'],
            varpref='S',
            evidence_ttypes=[match_cpd_t, dup_cpd_t],
            pmf_func=score_pmf3,
        )

    # Instanciate templates

    if mode == 1 or mode == 5:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        match_cpds = [match_cpd_t.new_cpd(parents=cpds) for cpds in namepair_cpds]
        score_cpds = [score_cpd_t.new_cpd(parents=cpds) for cpds in zip(match_cpds)]
        if mode == 5:
            # triple_idxs = ut.colwise_diag_idxs(num_annots, 3)
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

            # score_cpds = [score_cpd_t.new_cpd(parents=cpds)
            #              for cpds in zip(trimatch_cpds)]

            cpd_list = name_cpds + score_cpds + match_cpds + trimatch_cpds
        else:
            cpd_list = name_cpds + score_cpds + match_cpds
    elif mode == 2:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        score_cpds = [score_cpd_t.new_cpd(parents=cpds) for cpds in namepair_cpds]
        cpd_list = name_cpds + score_cpds
    elif mode == 3 or mode == 4:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        match_cpds = [match_cpd_t.new_cpd(parents=cpds) for cpds in namepair_cpds]
        if mode == 3:
            dup_cpds = [
                dup_cpd_t.new_cpd(parents=''.join(map(str, aids)))
                for aids in ut.list_unflat_take(annots, upper_diag_idxs)
            ]
        else:
            dup_cpds = [dup_cpd_t.new_cpd(parents=[mcpds]) for mcpds in match_cpds]
        score_cpds = [
            score_cpd_t.new_cpd(parents=([mcpds] + [dcpd]))
            for mcpds, dcpd in zip(match_cpds, dup_cpds)
        ]
        cpd_list = name_cpds + score_cpds + match_cpds + dup_cpds

    # print('upper_diag_idxs = %r' % (upper_diag_idxs,))
    print('score_cpds = %r' % (ut.list_getattr(score_cpds, 'variable'),))
    # import sys
    # sys.exit(1)

    # Make Model
    model = pgm_ext.define_model(cpd_list)
    model.num_names = num_names

    if verbose:
        model.print_templates()
        # ut.colorprint('\n --- CPD Templates ---', 'blue')
        # for temp_cpd in templates:
        #    ut.colorprint(temp_cpd._cpdstr('psql'), 'turquoise')
    # print_ascii_graph(model)
    return model


def show_model(model, evidence={}, soft_evidence={}, **kwargs):
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

    import wbia.plottool as pt
    import networkx as netx
    import matplotlib as mpl

    fnum = pt.ensure_fnum(None)
    fig = pt.figure(fnum=fnum, pnum=(3, 1, (slice(0, 2), 0)), doclf=True)  # NOQA
    # fig = pt.figure(fnum=fnum, pnum=(3, 2, (1, slice(1, 2))), doclf=True)  # NOQA
    ax = pt.gca()
    var2_post = {f.variables[0]: f for f in kwargs.get('factor_list', [])}

    netx_graph = model
    # netx_graph.graph.setdefault('graph', {})['size'] = '"10,5"'
    # netx_graph.graph.setdefault('graph', {})['rankdir'] = 'LR'

    pos = get_hacked_pos(netx_graph)
    # netx.nx_agraph.pygraphviz_layout(netx_graph)
    # pos = netx.nx_agraph.pydot_layout(netx_graph, prog='dot')
    # pos = netx.nx_agraph.graphviz_layout(netx_graph)

    drawkw = dict(pos=pos, ax=ax, with_labels=True, node_size=1500)
    if evidence is not None:
        node_colors = [
            # (pt.TRUE_BLUE
            (pt.WHITE if node not in soft_evidence else pt.LIGHT_PINK)
            if node not in evidence
            else pt.FALSE_RED
            for node in netx_graph.nodes()
        ]

        for node in netx_graph.nodes():
            cpd = model.var2_cpd[node]
            if cpd.ttype == 'score':
                pass
        drawkw['node_color'] = node_colors

    netx.draw(netx_graph, **drawkw)

    show_probs = True
    if show_probs:
        textprops = {
            'family': 'monospace',
            'horizontalalignment': 'left',
            #'horizontalalignment': 'center',
            #'size': 12,
            'size': 8,
        }

        textkw = dict(
            xycoords='data',
            boxcoords='offset points',
            pad=0.25,
            framewidth=True,
            arrowprops=dict(arrowstyle='->'),
            # bboxprops=dict(fc=node_attr['fillcolor']),
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

            prior_marg = (
                cpd
                if cpd.evidence is None
                else cpd.marginalize(cpd.evidence, inplace=False)
            )

            prior_text = None

            text = None
            if variable in evidence:
                text = cpd.variable_statenames[evidence[variable]]
            elif variable in var2_post:
                post_marg = var2_post[variable]
                text = pgm_ext.make_factor_text(post_marg, 'post')
                prior_text = pgm_ext.make_factor_text(prior_marg, 'prior')
            else:
                if len(evidence) == 0 and len(soft_evidence) == 0:
                    prior_text = pgm_ext.make_factor_text(prior_marg, 'prior')

            show_post = kwargs.get('show_post', False)
            show_prior = kwargs.get('show_prior', False)
            show_prior = True
            show_post = True

            show_ev = evidence is not None and variable in evidence
            if (show_post or show_ev) and text is not None:
                offset_box = mpl.offsetbox.TextArea(text, textprops)
                artist = mpl.offsetbox.AnnotationBbox(
                    # offset_box, (x + 5, y), xybox=(20., 5.),
                    offset_box,
                    (x, y + 5),
                    xybox=(4.0, 20.0),
                    # box_alignment=(0, 0),
                    box_alignment=(0.5, 0),
                    **textkw
                )
                offset_box_list.append(offset_box)
                artist_list.append(artist)

            if show_prior and prior_text is not None:
                offset_box2 = mpl.offsetbox.TextArea(prior_text, textprops)
                artist2 = mpl.offsetbox.AnnotationBbox(
                    # offset_box2, (x - 5, y), xybox=(-20., -15.),
                    # offset_box2, (x, y - 5), xybox=(-15., -20.),
                    offset_box2,
                    (x, y - 5),
                    xybox=(-4, -20.0),
                    # box_alignment=(1, 1),
                    box_alignment=(0.5, 1),
                    **textkw
                )
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
        # max_marginal_list = []
        # for name, marginal in marginalized_joints.items():
        #    states = list(ut.iprod(*marginal.statenames))
        #    vals = marginal.values.ravel()
        #    x = vals.argmax()
        #    max_marginal_list += ['P(' + ', '.join(states[x]) + ') = ' + str(vals[x])]
        # title += str(marginal)
        top_assignments = kwargs.get('top_assignments', None)
        if top_assignments is not None:
            map_assign, map_prob = top_assignments[0]
            if map_assign is not None:
                # title += '\nMAP=' + ut.repr2(map_assign, strvals=True)
                title += '\nMAP: ' + map_assign + ' @' + '%.2f%%' % (100 * map_prob,)
        if kwargs.get('show_title', True):
            pt.set_figtitle(title, size=14)
        # pt.set_xlabel()

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
        bin_vals = ut.get_list_column(top_assignments, 1)

        # bin_labels = ['\n'.join(ut.textwrap.wrap(_lbl, width=30)) for _lbl in bin_labels]

        pt.draw_histogram(
            bin_labels,
            bin_vals,
            fnum=fnum,
            pnum=(3, 8, (2, slice(4, None))),
            transpose=True,
            use_darkbackground=False,
            # xtick_rotation=-10,
            ylabel='Prob',
            xlabel='assignment',
        )
        pt.set_title('Assignment probabilities')

    # fpath = ('name_model_' + suff + '.png')
    # pt.plt.savefig(fpath)
    # return fpath


def flow():
    """
    http://pmneila.github.io/PyMaxflow/maxflow.html#maxflow-fastmin

    pip install PyMaxFlow
    pip install pystruct
    pip install hdbscan
    """
    # Toy problem representing attempting to discover names via annotation
    # scores

    import pystruct  # NOQA
    import pystruct.models  # NOQA
    import networkx as netx  # NOQA

    import vtool as vt

    num_annots = 10
    num_names = num_annots
    hidden_nids = np.random.randint(0, num_names, num_annots)
    unique_nids, groupxs = vt.group_indices(hidden_nids)

    toy_params = {True: {'mu': 1.0, 'sigma': 2.2}, False: {'mu': 7.0, 'sigma': 0.9}}

    if True:
        import vtool as vt
        import wbia.plottool as pt

        xdata = np.linspace(0, 100, 1000)
        tp_pdf = vt.gauss_func1d(xdata, **toy_params[True])
        fp_pdf = vt.gauss_func1d(xdata, **toy_params[False])
        pt.plot_probabilities([tp_pdf, fp_pdf], ['TP', 'TF'], xdata=xdata)

    def metric(aidx1, aidx2, hidden_nids=hidden_nids, toy_params=toy_params):
        if aidx1 == aidx2:
            return 0
        rng = np.random.RandomState(int(aidx1 + aidx2))
        same = hidden_nids[int(aidx1)] == hidden_nids[int(aidx2)]
        mu, sigma = ut.dict_take(toy_params[same], ['mu', 'sigma'])
        return np.clip(rng.normal(mu, sigma), 0, np.inf)

    pairwise_aidxs = list(ut.iprod(range(num_annots), range(num_annots)))
    pairwise_labels = np.array(
        [hidden_nids[a1] == hidden_nids[a2] for a1, a2 in pairwise_aidxs]
    )
    pairwise_scores = np.array([metric(*zz) for zz in pairwise_aidxs])
    pairwise_scores_mat = pairwise_scores.reshape(num_annots, num_annots)
    if num_annots <= 10:
        print(ut.repr2(pairwise_scores_mat, precision=1))

    # aids = list(range(num_annots))
    # g = netx.DiGraph()
    # g.add_nodes_from(aids)
    # g.add_edges_from([(tup[0], tup[1], {'weight': score}) for tup, score in zip(pairwise_aidxs, pairwise_scores) if tup[0] != tup[1]])
    # netx.draw_graphviz(g)
    # pr = netx.pagerank(g)

    X = pairwise_scores
    Y = pairwise_labels

    encoder = vt.ScoreNormalizer()
    encoder.fit(X, Y)
    encoder.visualize()

    # meanshift clustering
    import sklearn

    bandwidth = sklearn.cluster.estimate_bandwidth(
        X[:, None]
    )  # , quantile=quantile, n_samples=500)
    assert bandwidth != 0, '[] bandwidth is 0. Cannot cluster'
    # bandwidth is with respect to the RBF used in clustering
    # ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms = sklearn.cluster.MeanShift(
        bandwidth=bandwidth, bin_seeding=True, cluster_all=False
    )
    ms.fit(X[:, None])
    label_arr = ms.labels_
    unique_labels = np.unique(label_arr)
    max_label = max(0, unique_labels.max())
    num_orphans = (label_arr == -1).sum()
    label_arr[label_arr == -1] = np.arange(max_label + 1, max_label + 1 + num_orphans)

    X_data = np.arange(num_annots)[:, None].astype(np.int64)

    # graph = pystruct.models.GraphCRF(
    #    n_states=None,
    #    n_features=None,
    #    inference_method='lp',
    #    class_weight=None,
    #    directed=False,
    # )

    import scipy
    import scipy.cluster
    import scipy.cluster.hierarchy

    thresh = 2.0
    labels = scipy.cluster.hierarchy.fclusterdata(X_data, thresh, metric=metric)
    unique_lbls, lblgroupxs = vt.group_indices(labels)
    print(groupxs)
    print(lblgroupxs)
    print('groupdiff = %r' % (ut.compare_groupings(groupxs, lblgroupxs),))
    print('common groups = %r' % (ut.find_grouping_consistencies(groupxs, lblgroupxs),))
    # X_data, seconds_thresh, criterion='distance')

    # help(hdbscan.HDBSCAN)

    import hdbscan

    alg = hdbscan.HDBSCAN(
        metric=metric, min_cluster_size=1, p=1, gen_min_span_tree=1, min_samples=2
    )
    labels = alg.fit_predict(X_data)
    labels[labels == -1] = np.arange(np.sum(labels == -1)) + labels.max() + 1
    unique_lbls, lblgroupxs = vt.group_indices(labels)
    print(groupxs)
    print(lblgroupxs)
    print('groupdiff = %r' % (ut.compare_groupings(groupxs, lblgroupxs),))
    print('common groups = %r' % (ut.find_grouping_consistencies(groupxs, lblgroupxs),))

    # import ddbscan
    # help(ddbscan.DDBSCAN)
    # alg = ddbscan.DDBSCAN(2, 2)

    # D = np.zeros((len(aids), len(aids) + 1))
    # D.T[-1] = np.arange(len(aids))

    ## Can alpha-expansion be used when the pairwise potentials are not in a grid?

    # hidden_ut.group_items(aids, hidden_nids)
    if False:
        import maxflow

        # from maxflow import fastmin
        # Create a graph with integer capacities.
        g = maxflow.Graph[int](2, 2)
        # Add two (non-terminal) nodes. Get the index to the first one.
        nodes = g.add_nodes(2)
        # Create two edges (forwards and backwards) with the given capacities.
        # The indices of the nodes are always consecutive.
        g.add_edge(nodes[0], nodes[1], 1, 2)
        # Set the capacities of the terminal edges...
        # ...for the first node.
        g.add_tedge(nodes[0], 2, 5)
        # ...for the second node.
        g.add_tedge(nodes[1], 9, 4)
        g = maxflow.Graph[float](2, 2)
        g.maxflow()
        g.get_nx_graph()
        g.get_segment(nodes[0])


# seed = (start * 11)
# np.random.RandomState(seed)
# if initial_nids is None:
# else:
#    # HACK
#    #import utool
#    #utool.embed()
#    nid_is_avail = vt.index_to_boolmask(initial_nids, initial_nids.max() + 1)
#    next_ = np.where(~nid_is_avail)[0]
#    if len(next_) == 0:
#        nid_is_avail = np.append(nid_is_avail, [True])
#        idx = len(nid_is_avail) - 1
#    else:
#        idx = next_[0]
#        nid_is_avail[idx] = True
#    avail_nids = np.where(nid_is_avail)[0]
#    p = np.ones(len(avail_nids))
#    # chance to see a new annotation
#    chance = .4
#    # more or less
#    p[:] = (1 - chance) / (len(p) - 1)
#    p[idx] = chance
#    #print('p = %r' % (p,))
#    #nids = rng.randint(0, num_names, num_annots)
#    #print('avail_nids = %r' % (avail_nids,))
#    nids = rng.choice(avail_nids, num_annots, p=p)
#    #print('nids = %r' % (nids,))
