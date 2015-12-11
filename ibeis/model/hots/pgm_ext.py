# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import networkx as netx
import six  # NOQA
import utool as ut
import numpy as np
from six.moves import zip, map
import pgmpy
import pgmpy.inference
import pgmpy.factors
import pgmpy.models
print, rrr, profile = ut.inject2(__name__, '[pgmext]')


def print_factors(model, factor_list):
    if hasattr(model, 'var2_cpd'):
        semtypes = [model.var2_cpd[f.variables[0]].ttype
                    for f in factor_list]
    else:
        semtypes = [0] * len(factor_list)
    for type_, factors in ut.group_items(factor_list, semtypes).items():
        print('Result Factors (%r)' % (type_,))
        factors = ut.sortedby(factors, [f.variables[0] for f in factors])
        for fs_ in ut.ichunks(factors, 4):
            ut.colorprint(ut.hz_str([f._str('phi', 'psql') for f in fs_]),
                          'yellow')


def define_model(cpd_list):
    """
    Custom extensions of pgmpy modl
    """

    input_graph = ut.flatten([
        [(evar, cpd.variable) for evar in cpd.evidence]
        for cpd in cpd_list if cpd.evidence is not None
    ])
    model = pgmpy.models.BayesianModel(input_graph)
    model.add_cpds(*cpd_list)
    model.var2_cpd = {cpd.variable: cpd for cpd in model.cpds}
    model.ttype2_cpds = ut.groupby_attr(model.cpds, 'ttype')
    model._templates = list(set([cpd._template_
                                 for cpd in model.var2_cpd.values()]))

    def pretty_evidence(model, evidence):
        return [evar + '=' + str(model.var2_cpd[evar].variable_statenames[val])
                for evar, val in evidence.items()]

    def print_templates(model):
        templates = model._templates
        ut.colorprint('\n --- CPD Templates ---', 'blue')
        for temp_cpd in templates:
            ut.colorprint(temp_cpd._cpdstr('psql'), 'turquoise')

    def print_priors(model, ignore_ttypes=[], title='Priors'):
        ut.colorprint('\n --- %s ---' % (title,), 'darkblue')
        for ttype, cpds in model.ttype2_cpds.items():
            if ttype not in ignore_ttypes:
                for fs_ in ut.ichunks(cpds, 4):
                    ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]), 'darkblue')

    ut.inject_func_as_method(model, print_priors)
    ut.inject_func_as_method(model, print_templates)
    ut.inject_func_as_method(model, pretty_evidence)
    return model


class TemplateCPD(object):
    """
    Factory for templated cpds
    """
    def __init__(self, ttype, basis, varpref, evidence_ttypes=None,
                 pmf_func=None, special_basis_pool=None):
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
            varpref = ttype
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
                tcpd.example_cpd(i)
                for i, tcpd in enumerate(self.evidence_ttypes)
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
        if HACK_SAME_IDS and ut.list_allsame(template_ids):
            _id = template_ids[0]
        else:
            _id = ''.join(template_ids)
        variable = ''.join([self.varpref, _id])
        #variable = '_'.join([self.varpref, '{' + _id + '}'])
        #variable = '$%s$' % (variable,)

        evidence_cpds = [cpd for cpd in parents if hasattr(cpd, 'ttype')]
        if len(evidence_cpds) == 0:
            evidence_cpds = None
        variable_card = len(self.basis)
        statename_dict = {
            variable: self.basis,
        }
        if self.evidence_ttypes is not None:
            if any(cpd.ttype != tcpd.ttype
                   for cpd, tcpd in zip(evidence_cpds, evidence_cpds)):
                raise ValueError('Evidence is not of appropriate type')
            evidence_bases = [cpd.variable_statenames for cpd in evidence_cpds]
            evidence_card = list(map(len, evidence_bases))
            evidence_states = list(ut.iprod(*evidence_bases))

            for cpd in evidence_cpds:
                statename_dict.update(cpd.statename_dict)

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
                values = np.array([
                    [pmf_func(vstate, *estates) for estates in evidence_states]
                    for vstate in self.basis
                ])
            ensure_normalized = True
            if ensure_normalized:
                values = values / values.sum(axis=0)
        else:
            # assume uniform
            values = [[1.0 / variable_card] * variable_card]

        cpd = pgmpy.factors.TabularCPD(
            variable=variable,
            variable_card=variable_card,
            values=values,
            evidence=evidence,
            evidence_card=evidence_card,
            statename_dict=statename_dict,
        )
        cpd.ttype = self.ttype
        cpd._template_ = self
        cpd._template_id = _id
        return cpd


def print_ascii_graph(model_):
    """
    pip install img2txt.py

    python -c
    """
    from PIL import Image
    from six.moves import StringIO
    #import networkx as netx
    import copy
    model = copy.deepcopy(model_)
    assert model is not model_
    # model.graph.setdefault('graph', {})['size'] = '".4,.4"'
    model.graph.setdefault('graph', {})['size'] = '".3,.3"'
    model.graph.setdefault('graph', {})['height'] = '".3,.3"'
    pydot_graph = netx.to_pydot(model)
    png_str = pydot_graph.create_png(prog='dot')
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    pil_img = Image.open(sio)
    print('pil_img.size = %r' % (pil_img.size,))
    #def print_ascii_image(pil_img):
    #    img2txt = ut.import_module_from_fpath('/home/joncrall/venv/bin/img2txt.py')
    #    import sys
    #    pixel = pil_img.load()
    #    width, height = pil_img.size
    #    bgcolor = None
    #    #fill_string =
    #    img2txt.getANSIbgstring_for_ANSIcolor(img2txt.getANSIcolor_for_rgb(bgcolor))
    #    fill_string = "\x1b[49m"
    #    fill_string += "\x1b[K"          # does not move the cursor
    #    sys.stdout.write(fill_string)
    #    img_ansii_str = img2txt.generate_ANSI_from_pixels(pixel, width, height, bgcolor)
    #    sys.stdout.write(img_ansii_str)
    def print_ascii_image(pil_img):
        #https://gist.github.com/cdiener/10491632
        SC = 1.0
        GCF = 1.0
        WCF = 1.0
        img = pil_img
        S = (int(round(img.size[0] * SC * WCF * 3)), int(round(img.size[1] * SC)))
        img = np.sum( np.asarray( img.resize(S) ), axis=2)
        print('img.shape = %r' % (img.shape,))
        img -= img.min()
        chars = np.asarray(list(' .,:;irsXA253hMHGS#9B&@'))
        img = (1.0 - img / img.max()) ** GCF * (chars.size - 1)
        print( "\n".join( ("".join(r) for r in chars[img.astype(int)]) ) )
    print_ascii_image(pil_img)
    pil_img.close()
    pass


def _debug_repr_model(model):
    cpd_code_list = [_debug_repr_cpd(cpd) for cpd in model.cpds]
    code_fmt = ut.codeblock(
        '''
        import numpy as np
        import pgmpy
        import pgmpy.inference
        import pgmpy.factors
        import pgmpy.models

        {cpds}

        cpd_list = {nodes}
        input_graph = {edges}
        model = pgmpy.models.BayesianModel(input_graph)
        model.add_cpds(*cpd_list)
        infr = pgmpy.inference.BeliefPropagation(model)
        ''')

    code = code_fmt.format(
        cpds='\n'.join(cpd_code_list),
        nodes=ut.repr2(sorted(model.nodes()), strvals=True),
        edges=ut.repr2(sorted(model.edges()), nl=1),
    )
    ut.print_code(code)
    ut.copy_text_to_clipboard(code)


def _debug_repr_cpd(cpd):
    import re
    import utool as ut
    code_fmt = ut.codeblock(
        '''
        {variable} = pgmpy.factors.TabularCPD(
            variable={variable_repr},
            variable_card={variable_card_repr},
            values={get_cpd_repr},
            evidence={evidence_repr},
            evidence_card={evidence_card_repr},
        )
        ''')
    keys = ['variable', 'variable_card', 'values', 'evidence', 'evidence_card']
    dict_ = ut.odict(zip(keys, [getattr(cpd, key) for key in keys]))
    # HACK
    dict_['values'] = cpd.get_cpd()
    r = ut.repr2(dict_, explicit=True, nobraces=True, nl=True)
    print(r)

    # Parse props that are needed for this fmtstr
    fmt_keys = [match.groups()[0] for match in re.finditer('{(.*?)}', code_fmt)]
    need_reprs = [key[:-5] for key in fmt_keys if key.endswith('_repr')]
    need_keys = [key for key in fmt_keys if not key.endswith('_repr')]
    # Get corresponding props
    # Call methods if needbe
    tmp = [(prop, getattr(cpd, prop)) for prop in need_reprs]
    tmp = [(x, y()) if ut.is_funclike(y) else (x, y) for (x, y) in tmp]
    fmtdict = dict(tmp)
    fmtdict = ut.map_dict_vals(ut.repr2, fmtdict)
    fmtdict = ut.map_dict_keys(lambda x: x + '_repr', fmtdict)
    tmp2 = [(prop, getattr(cpd, prop)) for prop in need_keys]
    fmtdict.update(dict(tmp2))
    code = code_fmt.format(**fmtdict)
    return code


def make_factor_text(factor, name):
    collapse_uniform = True
    if collapse_uniform and ut.almost_allsame(factor.values):
        # Reduce uniform text
        ftext = name + ':\nuniform(%.3f)' % (factor.values[0],)
    else:
        values = factor.values
        rowstrs = ['p(%s)=%.3f' % (','.join(n), v,)
                   for n, v in zip(zip(*factor.statenames), values)]
        idxs = ut.list_argmaxima(values)
        for idx in idxs:
            rowstrs[idx] += '*'
        thresh = 4
        if len(rowstrs) > thresh:
            sortx = factor.values.argsort()[::-1]
            rowstrs = ut.take(rowstrs, sortx[0:(thresh - 1)])
            rowstrs += ['... %d more' % ((len(values) - len(rowstrs)),)]
        ftext = name + ': \n' + '\n'.join(rowstrs)
    return ftext


def coin_example():
    """
    Simple example of conditional independence

    CommandLine:
        python -m ibeis.model.hots.pgm_ext --exec-coin_example

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pgm_ext import *  # NOQA
        >>> model = coin_example()
        >>> model.print_templates()
        >>> model.print_priors()
        >>> infr = pgmpy.inference.VariableElimination(model)
        >>> print('Observe nothing')
        >>> factor_list1 = infr.query(['T02'], {}).values()
        >>> print_factors(model, factor_list1)
        >>> #
        >>> print('Observe that toss 1 was heads')
        >>> evidence = infr._ensure_internal_evidence({'T01': 'heads'}, model)
        >>> factor_list2 = infr.query(['T02'], evidence).values()
        >>> print_factors(model, factor_list2)
        >>> #
        >>> phi1 = factor_list1[0]
        >>> phi2 = factor_list2[0]
        >>> assert phi2['heads'] > phi1['heads']
        >>> print('Slightly more likely that you will see heads in the second coin toss')
        >>> #
        >>> print('Observe nothing')
        >>> factor_list1 = infr.query(['T02'], {}).values()
        >>> print_factors(model, factor_list1)
        >>> #
        >>> print('Observe that toss 1 was tails')
        >>> evidence = infr._ensure_internal_evidence({'T01': 'tails'}, model)
        >>> factor_list2 = infr.query(['T02'], evidence).values()
        >>> print_factors(model, factor_list2)
        >>> ut.quit_if_noshow()
        >>> netx.draw_graphviz(model, with_labels=True)
        >>> ut.show_if_requested()
    """
    def toss_pmf(side, coin):
        toss_lookup = {
            'fair': {'heads': .5, 'tails': .5},
            #'bias': {'heads': .6, 'tails': .4},
            'bias': {'heads': .9, 'tails': 1},
        }
        return toss_lookup[coin][side]
    coin_cpd_t = TemplateCPD(
        'coin', ['fair', 'bias'], varpref='C')
    toss_cpd_t = TemplateCPD(
        'toss', ['heads', 'tails'], varpref='T',
        evidence_ttypes=[coin_cpd_t], pmf_func=toss_pmf)
    coin_cpd = coin_cpd_t.new_cpd(0)
    toss1_cpd = toss_cpd_t.new_cpd(parents=[coin_cpd, 1])
    toss2_cpd = toss_cpd_t.new_cpd(parents=[coin_cpd, 2])
    model = define_model([coin_cpd, toss2_cpd, toss1_cpd])
    return model


def mustbe_example():
    """
    Simple example where observing F0 forces N0 to take on a value.

    CommandLine:
        python -m ibeis.model.hots.pgm_ext --exec-mustbe_example

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pgm_ext import *  # NOQA
        >>> model = mustbe_example()
        >>> model.print_templates()
        >>> model.print_priors()
        >>> #infr = pgmpy.inference.VariableElimination(model)
        >>> infr = pgmpy.inference.BeliefPropagation(model)
        >>> print('Observe: ' + ','.join(model.pretty_evidence({})))
        >>> factor_list1 = infr.query(['N0'], {}).values()
        >>> map1 = infr.map_query(['N0'], evidence)
        >>> print('map1 = %r' % (map1,))
        >>> print_factors(model, factor_list1)
        >>> #
        >>> evidence = infr._ensure_internal_evidence({'F0': 'true'}, model)
        >>> print('Observe: ' + ','.join(model.pretty_evidence(evidence)))
        >>> factor_list2 = infr.query(['N0'], evidence).values()
        >>> map2 = infr.map_query(['N0'], evidence)
        >>> print('map2 = %r' % (map2,))
        >>> print_factors(model, factor_list2)
        >>> #
        >>> evidence = infr._ensure_internal_evidence({'F0': 'false'}, model)
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
        >>> netx.draw_graphviz(model, with_labels=True)
        >>> ut.show_if_requested()

    Ignore:
        from ibeis.model.hots.pgm_ext import _debug_repr_model
        _debug_repr_model(model)
    """
    def isfred_pmf(isfred, name):
        toss_lookup = {
            'fred': {'true': 1, 'false': 0},
            'sue': {'true': 0, 'false': 1},
            'tom': {'true': 0, 'false': 1},
        }
        return toss_lookup[name][isfred]
    name_cpd_t = TemplateCPD(
        'name', ['fred', 'sue', 'tom'], varpref='N')
    isfred_cpd_t = TemplateCPD(
        'fred', ['true', 'false'], varpref='F',
        evidence_ttypes=[name_cpd_t], pmf_func=isfred_pmf)
    name_cpd = name_cpd_t.new_cpd(0)
    isfred_cpd = isfred_cpd_t.new_cpd(parents=[name_cpd])
    model = define_model([name_cpd, isfred_cpd])
    return model


def map_example():
    """
    Simple example where observing F0 forces N0 to take on a value.

    CommandLine:
        python -m ibeis.model.hots.pgm_ext --exec-map_example

    References:
        https://class.coursera.org/pgm-003/lecture/44

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.pgm_ext import *  # NOQA
        >>> model = map_example()
        >>> ut.quit_if_noshow()
        >>> netx.draw_graphviz(model, with_labels=True)
        >>> ut.show_if_requested()

    Ignore:
        from ibeis.model.hots.pgm_ext import _debug_repr_model
        _debug_repr_model(model)
    """
    # https://class.coursera.org/pgm-003/lecture/44
    a_cpd_t = TemplateCPD(
        'A', ['0', '1'], varpref='A', pmf_func=[[.4], [.6]])
    b_cpd_t = TemplateCPD(
        'B', ['0', '1'], varpref='B',
        evidence_ttypes=[a_cpd_t], pmf_func=[[.1, .5], [.9, .5]])
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.model.hots.pgm_ext
        python -m ibeis.model.hots.pgm_ext --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
