# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
import numpy as np
from six.moves import zip, map
import pgmpy
import pgmpy.inference
import pgmpy.factors
import pgmpy.models
print, rrr, profile = ut.inject2(__name__, '[pgmext]')


def define_model(cpd_list):
    input_graph = ut.flatten([
        [(evar, cpd.variable) for evar in cpd.evidence]
        for cpd in cpd_list if cpd.evidence is not None
    ])
    model = pgmpy.models.BayesianModel(input_graph)
    model.add_cpds(*cpd_list)
    model.var2_cpd = {cpd.variable: cpd for cpd in model.cpds}
    model.ttype2_cpds = ut.groupby_attr(model.cpds, 'ttype')

    def pretty_evidence(model, evidence):
        return [evar + '=' + str(model.var2_cpd[evar].variable_statenames[val])
                for evar, val in evidence.items()]
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
            if (state_card - num_special) > 0:
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
        Makes a new instance of this tempalte
        """
        if pmf_func is None:
            pmf_func = self.pmf_func
        if not ut.isiterable(parents):
            _id = parents
            evidence_cpds = None
        else:
            evidence_cpds = parents
            _id = ''.join([cpd._template_id for cpd in parents])
        variable = ''.join([self.varpref, _id])
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

            # print('evidence_states = %r' % (evidence_states,))
            # print('self.basis = %r' % (self.basis,))
            values = np.array([
                [pmf_func(vstate, *estates) for estates in evidence_states]
                for vstate in self.basis])
            if False:
                # ensure normalized
                values = values / values.sum(axis=0)
            evidence = [cpd.variable for cpd in evidence_cpds]
        else:
            if evidence_cpds is not None:
                raise ValueError('Gave evidence for evidence-less template')
            evidence = None
            evidence_card = None

        if pmf_func is None:
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
        cpd._template_id = _id
        return cpd


def print_ascii_graph(model_):
    """
    pip install img2txt.py

    python -c
    """
    from PIL import Image
    from six.moves import StringIO
    import networkx as netx
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
    #    #fill_string = img2txt.getANSIbgstring_for_ANSIcolor(img2txt.getANSIcolor_for_rgb(bgcolor))
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
        model_inference = pgmpy.inference.BeliefPropagation(model)
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
