# -*- coding: utf-8 -*-
import torch
import torchvision
import utool as ut

print, rrr, profile = ut.inject2(__name__)


class Siamese(torch.nn.Module):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.verif.siamese import *
        >>> self = Siamese()
    """

    def __init__(self):
        ut.super2(Siamese, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.num_fcin = self.resnet.fc.in_features
        # replace the last layer of resnet
        self.resnet.fc = torch.nn.Linear(self.num_fcin, 500)
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def forward(self, input1, input2):
        """
        Compute a resnet50 vector for each input and look at the L2 distance
        between the vectors.
        """
        output1 = self.resnet(input1)
        output2 = self.resnet(input2)
        output = self.pdist(output1, output2)

        return output


def visualize():
    import networkx as nx
    import torch
    from torch.autograd import Variable

    def make_nx(var, params):
        param_map = {id(v): k for k, v in params.items()}
        print(param_map)
        node_attr = dict(
            style='filled',
            shape='box',
            align='left',
            fontsize='12',
            ranksep='0.1',
            height='0.2',
        )
        seen = set()
        G = nx.DiGraph()

        def size_to_str(size):
            return '(' + (', ').join(['%d' % v for v in size]) + ')'

        def build_graph(var):
            if var not in seen:
                if torch.is_tensor(var):
                    G.add_node(
                        id(var),
                        label=size_to_str(var.size()),
                        fillcolor='orange',
                        **node_attr,
                    )
                elif hasattr(var, 'variable'):
                    u = var.variable
                    node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()),)
                    G.add_node(
                        id(var), label=node_name, fillcolor='lightblue', **node_attr
                    )
                else:
                    G.add_node(id(var), label=str(type(var).__name__), **node_attr)
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            G.add_edge(id(u[0]), id(var))
                            build_graph(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        G.add_edge(id(t), id(var))
                        build_graph(t)

        build_graph(var.grad_fn)
        return G

    # inputs = torch.randn(1, 3, 224, 224)
    # resnet18 = models.resnet18()
    # y = resnet18(Variable(inputs))

    inputs = torch.randn(1, 3, 224, 224)
    # model = torchvision.models.resnet18()
    model = torchvision.models.resnet50()

    model = Siamese()

    # y = model(Variable(inputs))
    y = model(Variable(inputs), Variable(inputs))

    params = model.state_dict()
    G = make_nx(y, params)

    import wbia.plottool as pt

    pt.dump_nx_ondisk(G, './pytorch_network.png')
    ut.startfile('./pytorch_network.png')
    # pt.show_nx(G, arrow_width=1)
    # pt.zoom_factory()
    # pt.pan_factory()
