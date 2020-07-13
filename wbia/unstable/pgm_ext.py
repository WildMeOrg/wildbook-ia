# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
import numpy as np
from six.moves import zip, map
from wbia.unstable import pgm_viz

try:
    import pgmpy
    import pgmpy.inference
    from pgmpy.extern import tabulate

    HAS_PGMPY = True
except ImportError:
    HAS_PGMPY = False
    pass
# from wbia.algo.hots.pgm_viz import *  # NOQA
print, rrr, profile = ut.inject2(__name__)


def define_model(cpd_list):
    """
    Custom extensions of pgmpy modl
    """
    input_graph = ut.flatten(
        [
            [(evar, cpd.variable) for evar in cpd.evidence]
            for cpd in cpd_list
            if cpd.evidence is not None
        ]
    )
    model = pgmpy.models.BayesianModel(input_graph)
    model.add_cpds(*cpd_list)
    customize_model(model)
    return model


def customize_model(model):
    model.var2_cpd = {cpd.variable: cpd for cpd in model.cpds}
    model.ttype2_cpds = ut.groupby_attr(model.cpds, 'ttype')
    model._templates = list(set([cpd._template_ for cpd in model.var2_cpd.values()]))
    model.ttype2_template = {t.ttype: t for t in model._templates}

    def pretty_evidence(model, evidence):
        return [
            evar + '=' + str(model.var2_cpd[evar].variable_statenames[val])
            for evar, val in evidence.items()
        ]

    def print_templates(model, ignore_ttypes=[]):
        templates = model._templates
        ut.colorprint('\n --- CPD Templates ---', 'blue')
        for temp_cpd in templates:
            if temp_cpd.ttype not in ignore_ttypes:
                ut.colorprint(temp_cpd._cpdstr('psql'), 'turquoise')

    def print_priors(model, ignore_ttypes=[], title='Priors', color='darkblue'):
        ut.colorprint('\n --- %s ---' % (title,), color=color)
        for ttype, cpds in model.ttype2_cpds.items():
            if ttype not in ignore_ttypes:
                for fs_ in ut.ichunks(cpds, 4):
                    ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]), color)

    ut.inject_func_as_method(model, print_priors)
    ut.inject_func_as_method(model, print_templates)
    ut.inject_func_as_method(model, pretty_evidence)
    ut.inject_func_as_method(model, pgm_viz.show_model)
    ut.inject_func_as_method(model, pgm_viz.show_markov_model)
    ut.inject_func_as_method(model, pgm_viz.show_junction_tree)
    return model


class ApproximateFactor(object):
    """
    Instead of holding a weight for all possible states, an approximate factor
    simply lists a set of (potentially duplicate) states. Each state has a
    weight that is approximately proportional to the probability of that state.

    The main difference is that the cardinality are implicit and the row labels
    are explicit. In a normal factor it is reversed.

    Maybe rename to sparse factor?

    CommandLine:
        python -m wbia.algo.hots.pgm_ext --exec-ApproximateFactor --show

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.algo.hots.pgm_ext import *  # NOQA
        >>> state_idxs = [[1, 1, 1], [1, 0, 1], [2, 0, 2]]
        >>> weights = [.1, .2, .1]
        >>> variables = ['v1', 'v2', 'v3']
        >>> self = ApproximateFactor(state_idxs, weights, variables)
        >>> result = str(self)
        >>> print(result)
    """

    @classmethod
    def from_sampled(cls, sampled, variables=None, statename_dict=None):
        """
        convert sampled states into an approximate factor
        """
        if variables is None:
            variables = sampled.columns[:-1]
        state_idxs = np.array(
            [[item.state for item in row] for row in sampled[variables].values]
        )
        weights = sampled['_weight']
        phi = cls(state_idxs, weights, variables, statename_dict)
        return phi

    def __init__(self, state_idxs, weights, variables, statename_dict=None):
        self.variables = variables
        self.state_idxs = np.array(state_idxs)
        self.weights = np.array(weights)
        self.statename_dict = statename_dict

    def copy(self):
        """
        Returns a copy of the factor.
        """
        statename_dict = (
            self.statename_dict.copy() if self.statename_dict is not None else None
        )
        other = ApproximateFactor(
            self.state_idxs, self.weights, self.variables, statename_dict
        )
        return other

    @property
    def values(self):
        return self.weights

    def get_sparse_values(self):
        # http://stackoverflow.com/questions/20114194/sparse-array-in-python-cython
        # from scipy.sparse import coo_matrix
        # values = coo_matrix((self.weights, self.state_idxs.T), shape=self.cardinality)
        raise NotImplementedError('scipy only supports sparse 2D-arrays')

    def scope(self):
        return self.variables

    def marginalize(self, variables, inplace=True):
        r"""
        Modifies the factor with marginalized values.

        Args:
            variables (list, array-like):
                List of variables over which to marginalize.

            inplace (bool):
                If inplace=True it will modify the factor itself, else would
                return a new factor.

        Returns:
            Factor or None:
                if inplace=True (default) returns None
                if inplace=False returns a new `Factor` instance.

        CommandLine:
            python -m wbia.algo.hots.pgm_ext marginalize --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.hots.pgm_ext import *  # NOQA
            >>> state_idxs = [[1, 1, 1], [1, 0, 1], [2, 0, 2]]
            >>> weights = [.1, .2, .1]
            >>> variables = ['v1', 'v2', 'v3']
            >>> self = ApproximateFactor(state_idxs, weights, variables)
            >>> variables = ['v2']
            >>> inplace = False
            >>> phi = self.marginalize(variables, inplace)
            >>> print(phi)
            +------+------+--------------------+
            | v1   | v3   |   \hat{phi}(v1,v3) |
            |------+------+--------------------|
            | v1_1 | v3_1 |             0.3000 |
            | v1_2 | v3_2 |             0.1000 |
            +------+------+--------------------+
        """
        if isinstance(variables, six.string_types):
            raise TypeError('variables: Expected type list or array-like, got type str')

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError('{var} not in scope.'.format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = list(set(range(len(self.variables))) - set(var_indexes))
        index_to_keep = sorted(index_to_keep)
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.state_idxs = phi.state_idxs.T[index_to_keep].T

        if True:
            phi.consolidate()

        if not inplace:
            return phi

    def _compute_unique_state_ids(self):
        import vtool as vt

        # data_ids = vt.compute_ndarray_unique_rowids_unsafe(self.state_idxs)
        data_ids = np.array(
            vt.compute_unique_data_ids_(list(map(tuple, self.state_idxs)))
        )
        return data_ids

    def consolidate(self, inplace=False):
        r""" removes duplicate entries

        Example:
            >>> # UNSTABLE_DOCTEST
            >>> from wbia.algo.hots.pgm_ext import *  # NOQA
            >>> state_idxs = [[1, 0, 1], [1, 0, 1], [1, 0, 2]]
            >>> weights = [.1, .2, .1]
            >>> variables = ['v1', 'v2', 'v3']
            >>> self = ApproximateFactor(state_idxs, weights, variables)
            >>> inplace = False
            >>> phi = self.consolidate(inplace)
            >>> result = str(phi)
            >>> print(result)
            +------+------+------+-----------------------+
            | v1   | v2   | v3   |   \hat{phi}(v1,v2,v3) |
            |------+------+------+-----------------------|
            | v1_1 | v2_0 | v3_1 |                0.3000 |
            | v1_1 | v2_0 | v3_2 |                0.1000 |
            +------+------+------+-----------------------+
        """
        import vtool as vt

        phi = self.copy() if inplace else self
        # data_ids = vt.compute_ndarray_unique_rowids_unsafe(self.state_idxs)
        data_ids = self._compute_unique_state_ids()
        unique_ids, groupxs = vt.group_indices(data_ids)
        # assert len(unique_ids) == len(np.unique(vt.compute_unique_data_ids_(list(map(tuple, phi.state_idxs)))))
        if len(data_ids) != len(unique_ids):
            # Sum the values in the cpd to marginalize the duplicate probs
            # Take only the unique rows under this induced labeling
            unique_tmp_groupxs = np.array([gxs[0] for gxs in groupxs])
            self.state_idxs = self.state_idxs.take(unique_tmp_groupxs, axis=0)
            self.weights = np.array(
                [g.sum() for g in vt.apply_grouping(self.weights, groupxs)]
            )
            # print('[pgm] Consolidated %r states into %r states' % (len(data_ids), len(unique_ids),))
        # else:
        #    print('[pgm] Cannot consolidated %r unique states' % (len(data_ids),))
        if not inplace:
            return phi

    def normalize(self, inplace=True):
        r"""
        Normalizes the weights of factor so that they sum to 1.

        Args:
            inplace (bool): (default = True)

        CommandLine:
            python -m wbia.algo.hots.pgm_ext --exec-normalize

        Example:
            >>> # UNSTABLE_DOCTEST
            >>> from wbia.algo.hots.pgm_ext import *  # NOQA
            >>> state_idxs = [[0, 0, 1], [1, 0, 1], [2, 0, 2]]
            >>> weights = [.1, .2, .1]
            >>> variables = ['v1', 'v2', 'v3']
            >>> self = ApproximateFactor(state_idxs, weights, variables)
            >>> inplace = True
            >>> print(self)
            >>> self.normalize(inplace)
            >>> result = ('%s' % (self,))
            >>> print(result)
            +------+------+------+-----------------------+
            | v1   | v2   | v3   |   \hat{phi}(v1,v2,v3) |
            |------+------+------+-----------------------|
            | v1_0 | v2_0 | v3_1 |                0.2500 |
            | v1_1 | v2_0 | v3_1 |                0.5000 |
            | v1_2 | v2_0 | v3_2 |                0.2500 |
            +------+------+------+-----------------------+
        """
        phi = self if inplace else self.copy()
        phi.weights = phi.weights / phi.weights.sum()
        if not inplace:
            return phi

    def reorder(self, order=None, inplace=True):
        r"""
        Changes internal variable ordering

        CommandLine:
            python -m wbia.algo.hots.pgm_ext --exec-reorder

        Example:
            >>> # UNSTABLE_DOCTEST
            >>> from wbia.algo.hots.pgm_ext import *  # NOQA
            >>> state_idxs = [[0, 0, 1], [1, 0, 1], [2, 0, 2]]
            >>> weights = [.1, .2, .1]
            >>> variables = ['v1', 'v2', 'v3']
            >>> self = ApproximateFactor(state_idxs, weights, variables)
            >>> order = [2, 0, 1]
            >>> inplace = True
            >>> print(self)
            >>> self.reorder(order, inplace)
            >>> result = ('%s' % (self,))
            >>> print(result)
            +------+------+------+-----------------------+
            | v3   | v1   | v2   |   \hat{phi}(v3,v1,v2) |
            |------+------+------+-----------------------|
            | v3_1 | v1_0 | v2_0 |                0.1000 |
            | v3_1 | v1_1 | v2_0 |                0.2000 |
            | v3_2 | v1_2 | v2_0 |                0.1000 |
            +------+------+------+-----------------------+
        """
        phi = self if inplace else self.copy()
        if order is not None:
            if all(isinstance(x, int) for x in order):
                sortx = np.array(order)
            else:
                sortx = np.array([phi.variables.index(v) for v in order])
        else:
            sortx = np.lexsort((phi.variables,))
        phi.variables = [phi.variables[i] for i in sortx]
        phi.state_idxs = np.ascontiguousarray(phi.state_idxs[:, sortx])
        if not inplace:
            return phi

    def _row_labels(self, asindex=False):
        if asindex:
            row_labels = self.state_idxs
        else:
            row_labels = [
                [
                    '{var}_{i}'.format(var=var, i=i)
                    for var, i in zip(self.variables, state)
                ]
                for state in self.state_idxs
            ]
        return row_labels

    def _str(self, phi_or_p=None, tablefmt=None, sort=False, maxrows=None):
        """
        Generate the string from `__str__` method.
        """
        if phi_or_p is None:
            phi_or_p = r'\hat{phi}'
        if tablefmt is None:
            tablefmt = 'psql'
        string_header = list(self.scope())
        string_header.append(
            '{phi_or_p}({variables})'.format(
                phi_or_p=phi_or_p, variables=','.join(string_header)
            )
        )

        factor_table = []

        row_values = self.values.ravel()
        row_labels = self._row_labels(asindex=False)
        factor_table = [list(lbls) + [val] for lbls, val in zip(row_labels, row_values)]

        if sort:
            sortx = row_values.argsort()[::sort]
            factor_table = [factor_table[row] for row in sortx]

        if maxrows is not None and maxrows < len(factor_table):
            factor_table = factor_table[:maxrows]
            factor_table.append(['...'] * len(string_header))

        return tabulate(
            factor_table, headers=string_header, tablefmt=tablefmt, floatfmt='.4f'
        )

    def __str__(self):
        return self._str()

    @property
    def cardinality(self):
        cardinality = self.state_idxs.max(axis=0) + 1
        return cardinality

    def __repr__(self):
        var_card = ', '.join(
            [
                '{var}:{card}'.format(var=var, card=card)
                for var, card in zip(self.variables, self.cardinality)
            ]
        )
        return r'<ApproximateFactor representing phi({var_card}) at {address}>'.format(
            address=hex(id(self)), var_card=var_card
        )


def print_factors(model, factor_list):
    if hasattr(model, 'var2_cpd'):
        semtypes = [model.var2_cpd[f.variables[0]].ttype for f in factor_list]
    else:
        semtypes = [0] * len(factor_list)
    for type_, factors in ut.group_items(factor_list, semtypes).items():
        print('Result Factors (%r)' % (type_,))
        factors = ut.sortedby(factors, [f.variables[0] for f in factors])
        for fs_ in ut.ichunks(factors, 4):
            ut.colorprint(ut.hz_str([f._str('phi', 'psql') for f in fs_]), 'yellow')


class TemplateCPD(object):
    """
    Factory for templated cpds

    Args:
        ttype (?):
        basis (?):
        varpref (None): Letter to use as the random variable
        evidence_ttypes (None): (default = None)
        pmf_func (None): (default = None)
        special_basis_pool (None): (default = None)

    CommandLine:
        python -m wbia.algo.hots.pgm_ext TemplateCPD --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.pgm_ext import *  # NOQA
        >>> self = TemplateCPD('coin', ['fair', 'bias'], varpref='C')
        >>> cpd = self.new_cpd(0)
        >>> print(cpd)
    """

    def __init__(
        self,
        ttype,
        basis,
        varpref=None,
        evidence_ttypes=None,
        pmf_func=None,
        special_basis_pool=None,
    ):
        if isinstance(basis, tuple):
            state_pref, state_card = basis
            stop = state_card
            basis = []
            num_special = 0
            if special_basis_pool is not None:
                start = stop - len(special_basis_pool)
                num_special = min(len(special_basis_pool), state_card)
                basis = special_basis_pool[0:num_special]
            if (state_card - num_special) >= 0:
                start = num_special
                basis = basis + [state_pref + str(i) for i in range(start, stop)]
        if varpref is None:
            varpref = ttype[0].upper()
        self.basis = basis
        self.ttype = ttype
        self.varpref = varpref
        self.evidence_ttypes = evidence_ttypes
        self.pmf_func = pmf_func

    def __call__(self, *args, **kwargs):
        return self.new_cpd(*args, **kwargs)

    def _cpdstr(self, *args, **kwargs):
        example_cpd = self.example_cpd()
        return example_cpd._cpdstr(*args, **kwargs)

    @ut.memoize
    def example_cpd(self, id_=0):
        kw = dict()
        if self.evidence_ttypes is None:
            kw['parents'] = ut.chr_range(id_, id_ + 1)[0]
        else:
            kw['parents'] = [
                tcpd.example_cpd(i) for i, tcpd in enumerate(self.evidence_ttypes)
            ]
        example_cpd = self.new_cpd(**kw)
        return example_cpd

    def new_cpd(self, parents=None, pmf_func=None):
        """
        Makes a new random variable that is an instance of this tempalte

        parents : only used to define the name of this node.
        """
        if pmf_func is None:
            pmf_func = self.pmf_func

        # --- MAKE VARIABLE ID
        def _getid(obj):
            if isinstance(obj, int):
                return str(obj)
            elif isinstance(obj, six.string_types):
                return obj
            else:
                return obj._template_id

        if not ut.isiterable(parents):
            parents = [parents]

        template_ids = [_getid(cpd) for cpd in parents]
        HACK_SAME_IDS = True
        # TODO: keep track of parent index inheritence
        # then rectify uniqueness based on that
        if HACK_SAME_IDS and ut.allsame(template_ids):
            _id = template_ids[0]
        else:
            _id = ''.join(template_ids)
        variable = ''.join([self.varpref, _id])
        # variable = '_'.join([self.varpref, '{' + _id + '}'])
        # variable = '$%s$' % (variable,)

        evidence_cpds = [cpd for cpd in parents if hasattr(cpd, 'ttype')]
        if len(evidence_cpds) == 0:
            evidence_cpds = None

        variable_card = len(self.basis)
        statename_dict = {
            variable: self.basis,
        }
        if self.evidence_ttypes is not None:
            if any(
                cpd.ttype != tcpd.ttype for cpd, tcpd in zip(evidence_cpds, evidence_cpds)
            ):
                raise ValueError('Evidence is not of appropriate type')
            evidence_bases = [cpd.variable_statenames for cpd in evidence_cpds]
            evidence_card = list(map(len, evidence_bases))
            evidence_states = list(ut.iprod(*evidence_bases))

            for cpd in evidence_cpds:
                _dict = ut.dict_subset(cpd.statename_dict, [cpd.variable])
                statename_dict.update(_dict)

            evidence = [cpd.variable for cpd in evidence_cpds]
        else:
            if evidence_cpds is not None:
                raise ValueError('Gave evidence for evidence-less template')
            evidence = None
            evidence_card = None

        # --- MAKE TABLE VALUES
        if pmf_func is not None:
            if isinstance(pmf_func, list):
                values = np.array(pmf_func)
            else:
                values = np.array(
                    [
                        [pmf_func(vstate, *estates) for estates in evidence_states]
                        for vstate in self.basis
                    ]
                )
            ensure_normalized = True
            if ensure_normalized:
                values = values / values.sum(axis=0)
        else:
            # assume uniform
            fill_value = 1.0 / variable_card
            if evidence_card is None:
                values = np.full((1, variable_card), fill_value)
            else:
                values = np.full([variable_card] + list(evidence_card), fill_value)

        try:
            cpd = pgmpy.factors.TabularCPD(
                variable=variable,
                variable_card=variable_card,
                values=values,
                evidence=evidence,
                evidence_card=evidence_card,
                # statename_dict=statename_dict,
                state_names=statename_dict,
            )
        except Exception as ex:
            ut.printex(
                ex,
                'Failed to create TabularCPD',
                keys=[
                    'variable',
                    'variable_card',
                    'statename_dict',
                    'evidence_card',
                    'evidence',
                    'values.shape',
                ],
            )
            ut.embed()
            raise

        cpd.ttype = self.ttype
        cpd._template_ = self
        cpd._template_id = _id
        return cpd


def mustbe_example():
    """
    Simple example where observing F0 forces N0 to take on a value.

    CommandLine:
        python -m wbia.algo.hots.pgm_ext --exec-mustbe_example --show

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.algo.hots.pgm_ext import *  # NOQA
        >>> model = mustbe_example()
        >>> model.print_templates()
        >>> model.print_priors()
        >>> #infr = pgmpy.inference.VariableElimination(model)
        >>> infr = pgmpy.inference.BeliefPropagation(model)
        >>> print('Observe: ' + ','.join(model.pretty_evidence({})))
        >>> factor_list1 = infr.query(['N0'], {}).values()
        >>> map1 = infr.map_query(['N0'], evidence={})
        >>> print('map1 = %r' % (map1,))
        >>> print_factors(model, factor_list1)
        >>> #
        >>> evidence = model._ensure_internal_evidence({'F0': 'true'})
        >>> print('Observe: ' + ','.join(model.pretty_evidence(evidence)))
        >>> factor_list2 = infr.query(['N0'], evidence).values()
        >>> map2 = infr.map_query(['N0'], evidence)
        >>> print('map2 = %r' % (map2,))
        >>> print_factors(model, factor_list2)
        >>> #
        >>> evidence = model._ensure_internal_evidence({'F0': 'false'})
        >>> print('Observe: ' + ','.join(model.pretty_evidence(evidence)))
        >>> factor_list3 = infr.query(['N0'], evidence).values()
        >>> map3 = infr.map_query(['N0'], evidence)
        >>> print('map3 = %r' % (map3,))
        >>> print_factors(model, factor_list3)
        >>> #
        >>> phi1 = factor_list1[0]
        >>> phi2 = factor_list2[0]
        >>> assert phi1['fred'] == phi1['sue'], 'should be uniform'
        >>> assert phi2['fred'] == 1, 'should be 1'
        >>> ut.quit_if_noshow()
        >>> #netx.draw_graphviz(model, with_labels=True)
        >>> import wbia.plottool as pt
        >>> pgm_viz.show_model(model, fnum=1)
        >>> pgm_viz.show_model(model, fnum=2, evidence=evidence, factor_list=factor_list2)
        >>> ut.show_if_requested()

    Ignore:
        from wbia.algo.hots.pgm_ext import _debug_repr_model
        _debug_repr_model(model)
    """

    def isfred_pmf(isfred, name):
        return {
            'fred': {'true': 1, 'false': 0},
            'sue': {'true': 0, 'false': 1},
            'tom': {'true': 0, 'false': 1},
        }[name][isfred]

    name_cpd_t = TemplateCPD('name', ['fred', 'sue', 'tom'], varpref='N')
    isfred_cpd_t = TemplateCPD(
        'fred',
        ['true', 'false'],
        varpref='F',
        evidence_ttypes=[name_cpd_t],
        pmf_func=isfred_pmf,
    )
    name_cpd = name_cpd_t.new_cpd(0)
    isfred_cpd = isfred_cpd_t.new_cpd(parents=[name_cpd])
    model = define_model([name_cpd, isfred_cpd])
    return model


def map_example():
    """
    CommandLine:
        python -m wbia.algo.hots.pgm_ext --exec-map_example --show

    References:
        https://class.coursera.org/pgm-003/lecture/44

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.algo.hots.pgm_ext import *  # NOQA
        >>> model = map_example()
        >>> ut.quit_if_noshow()
        >>> #netx.draw_graphviz(model, with_labels=True)
        >>> pgm_viz.show_model(model, fnum=1)
        >>> ut.show_if_requested()

    Ignore:
        from wbia.algo.hots.pgm_ext import _debug_repr_model
        _debug_repr_model(model)
    """
    # https://class.coursera.org/pgm-003/lecture/44
    a_cpd_t = TemplateCPD('A', ['0', '1'], varpref='A', pmf_func=[[0.4], [0.6]])
    b_cpd_t = TemplateCPD(
        'B',
        ['0', '1'],
        varpref='B',
        evidence_ttypes=[a_cpd_t],
        pmf_func=[[0.1, 0.5], [0.9, 0.5]],
    )
    a_cpd = a_cpd_t.new_cpd(0)
    b_cpd = b_cpd_t.new_cpd(parents=[a_cpd])
    model = define_model([a_cpd, b_cpd])
    model.print_templates()
    model.print_priors()
    infr = pgmpy.inference.VariableElimination(model)
    marg_factors = infr.query(['A0', 'B0']).values()
    print_factors(model, marg_factors)
    map_res = infr.map_query()
    print('map_res = %r' % (map_res,))
    return model


def coin_example():
    """
    Simple example of conditional independence.

    Notes:
        We are given a coin.
        We do not know if it is fair or unfair.
        There is an equal chance of either.
        (If it is unfair it has a a 9-to-1 odds).
        Initially, the results a coin toss are initially conditionally
          independant of any other toss.
        However, if we observe a heads on the first toss the chance of heads
          on the second toss will increase.

    CommandLine:
        python -m wbia.algo.hots.pgm_ext --exec-coin_example
        python -m wbia.algo.hots.pgm_ext --exec-coin_example --show
        python -m wbia.algo.hots.pgm_ext --exec-coin_example --show --cmd

    Example:
        >>> # UNSTABLE_DOCTEST
        >>> from wbia.algo.hots.pgm_ext import *  # NOQA
        >>> model = coin_example()
        >>> model.print_templates()
        >>> model.print_priors()
        >>> query_vars = ['T02']
        >>> infr = pgmpy.inference.VariableElimination(model)
        >>> # Inference (1)
        >>> print('(1.a) Observe nothing')
        >>> evidence1 = {}
        >>> factor_list1 = infr.query(query_vars, evidence1).values()
        >>> print_factors(model, factor_list1)
        >>> print('(1.b)  nothing changes')
        >>> # Inference (2)
        >>> print('(2.a) Observe that toss 1 was heads')
        >>> evidence2 = model._ensure_internal_evidence({'T01': 'heads'})
        >>> factor_list2 = infr.query(query_vars, evidence2).values()
        >>> print_factors(model, factor_list2)
        >>> #
        >>> phi1 = factor_list1[0]
        >>> phi2 = factor_list2[0]
        >>> assert phi2['heads'] > phi1['heads']
        >>> print('(2.b) Slightly more likely to see heads in the second coin toss')
        >>> #
        >>> # print('Observe that toss 1 was tails')
        >>> # evidence = model._ensure_internal_evidence({'T01': 'tails'})
        >>> # factor_list2 = infr.query(query_vars, evidence).values()
        >>> # print_factors(model, factor_list2)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> from wbia.algo.hots import bayes
        >>> kw = bayes.cluster_query(model, query_vars,evidence2,
        >>>                          method='bp', operation='marginalize')
        >>> #model.show_model(fnum=1)
        >>> #model.show_model(fnum=2, evidence=evidence2, factor_list=factor_list2)
        >>> model.show_model(fnum=3, evidence=evidence2, **kw)
        >>> model.show_markov_model(fnum=4, evidence=evidence2, factor_list=factor_list2)
        >>> model.show_junction_tree(fnum=5, evidence=evidence2, factor_list=factor_list2)
        >>> #netx.draw_graphviz(model, with_labels=True)
        >>> ut.show_if_requested()
    """

    def toss_pmf(side, coin):
        toss_lookup = {
            'fair': {'heads': 0.5, 'tails': 0.5},
            # 'bias': {'heads': .6, 'tails': .4},
            'bias': {'heads': 0.9, 'tails': 0.1},
        }
        return toss_lookup[coin][side]

    coin_cpd_t = TemplateCPD('coin', ['fair', 'bias'], varpref='C')
    toss_cpd_t = TemplateCPD(
        'toss',
        ['heads', 'tails'],
        varpref='T',
        evidence_ttypes=[coin_cpd_t],
        pmf_func=toss_pmf,
    )
    coin_cpd = coin_cpd_t.new_cpd(0)
    toss1_cpd = toss_cpd_t.new_cpd(parents=[coin_cpd, 1])
    toss2_cpd = toss_cpd_t.new_cpd(parents=[coin_cpd, 2])
    model = define_model([coin_cpd, toss1_cpd, toss2_cpd])
    return model


def markovmodel_test():
    """
    >>> from wbia.algo.hots.pgm_ext import *  # NOQA
    """
    from pgmpy.models import MarkovModel
    from pgmpy.factors import Factor

    markovmodel = MarkovModel([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
    factor_a_b = Factor(variables=['A', 'B'], cardinality=[2, 2], values=[100, 5, 5, 100])
    factor_b_c = Factor(variables=['B', 'C'], cardinality=[2, 2], values=[100, 3, 2, 4])
    factor_c_d = Factor(variables=['C', 'D'], cardinality=[2, 2], values=[3, 5, 1, 6])
    factor_d_a = Factor(variables=['D', 'A'], cardinality=[2, 2], values=[6, 2, 56, 2])
    markovmodel.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)

    pgm_viz.show_markov_model(markovmodel)
    pgm_viz.show_junction_tree(markovmodel)
    # model = markovmodel.to_bayesian_model()
    # customize_model(model)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.hots.pgm_ext
        python -m wbia.algo.hots.pgm_ext --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    if HAS_PGMPY:
        ut.doctest_funcs()
