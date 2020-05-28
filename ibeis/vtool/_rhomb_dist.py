# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import collections

UNKNOWN = np.nan
R  = 1
FR = 2
F  = 3
FL = 4
L  = 5
BL = 6
B  = 7
BR = 8

U   = 9
UF  = 10
UB  = 11
UL  = 12
UR  = 13
UFL = 14
UFR = 15
UBL = 16
UBR = 17

D   = 18
DF  = 19
DB  = 20
DL  = 21
DR  = 22
DFL = 23
DFR = 24
DBL = 25
DBR = 26

INT_TO_CODE = collections.OrderedDict([
    (UNKNOWN, 'unknown'),
    (R,  'right'),
    (FR, 'frontright'),
    (F,  'front'),
    (FL, 'frontleft'),
    (L,  'left'),
    (BL, 'backleft'),
    (B,  'back'),
    (BR, 'backright'),

    (U,    'up'),
    (UF,   'upfront'),
    (UB,   'upback'),
    (UL,   'upleft'),
    (UR,   'upright'),
    (UFL,  'upfrontleft'),
    (UFR,  'upfrontright'),
    (UBL,  'upbackleft'),
    (UBR,  'upbackright'),

    (D,    'down'),
    (DF,   'downfront'),
    (DB,   'downback'),
    (DL,   'downleft'),
    (DR,   'downright'),
    (DFL,  'downfrontleft'),
    (DFR,  'downfrontright'),
    (DBL,  'downbackleft'),
    (DBR,  'downbackright'),
])


# HACK: mirrors ibeis viewpoint int distance encoding

VIEW_INT_DIST = {
    # DIST 0 PAIRS
    (B, B): 0, (BL, BL): 0, (BR, BR): 0, (D, D): 0, (DB, DB): 0,
    (DBL, DBL): 0, (DBR, DBR): 0, (DF, DF): 0, (DFL, DFL): 0,
    (DFR, DFR): 0, (DL, DL): 0, (DR, DR): 0, (F, F): 0, (FL, FL): 0,
    (FR, FR): 0, (L, L): 0, (R, R): 0, (U, U): 0, (UB, UB): 0,
    (UBL, UBL): 0, (UBR, UBR): 0, (UF, UF): 0, (UFL, UFL): 0,
    (UFR, UFR): 0, (UL, UL): 0, (UR, UR): 0,

    # DIST 1 PAIRS
    (B, BL): 1, (B, BR): 1, (B, DB): 1, (B, DBL): 1, (B, DBR): 1,
    (B, UB): 1, (B, UBL): 1, (B, UBR): 1, (BL, DBL): 1, (BL, L): 1,
    (BL, UBL): 1, (BR, DBR): 1, (BR, R): 1, (BR, UBR): 1, (D, DB): 1,
    (D, DBL): 1, (D, DBR): 1, (D, DF): 1, (D, DFL): 1, (D, DFR): 1,
    (D, DL): 1, (D, DR): 1, (DB, DBL): 1, (DB, DBR): 1, (DBL, DL): 1,
    (DBL, L): 1, (DBR, DR): 1, (DBR, R): 1, (DF, DFL): 1, (DF, DFR): 1,
    (DF, F): 1, (DFL, DL): 1, (DFL, F): 1, (DFL, FL): 1, (DFL, L): 1,
    (DFR, DR): 1, (DFR, F): 1, (DFR, FR): 1, (DFR, R): 1, (DL, L): 1,
    (DR, R): 1, (F, FL): 1, (F, FR): 1, (F, UF): 1, (F, UFL): 1,
    (F, UFR): 1, (FL, L): 1, (FL, UFL): 1, (FR, R): 1, (FR, UFR): 1,
    (L, UBL): 1, (L, UFL): 1, (L, UL): 1, (R, UBR): 1, (R, UFR): 1,
    (R, UR): 1, (U, UB): 1, (U, UBL): 1, (U, UBR): 1, (U, UF): 1,
    (U, UFL): 1, (U, UFR): 1, (U, UL): 1, (U, UR): 1, (UB, UBL): 1,
    (UB, UBR): 1, (UBL, UL): 1, (UBR, UR): 1, (UF, UFL): 1, (UF, UFR): 1,
    (UFL, UL): 1, (UFR, UR): 1,

    # DIST 2 PAIRS
    (B, D): 2, (B, DL): 2, (B, DR): 2, (B, L): 2, (B, R): 2, (B, U): 2,
    (B, UL): 2, (B, UR): 2, (BL, BR): 2, (BL, D): 2, (BL, DB): 2,
    (BL, DBR): 2, (BL, DFL): 2, (BL, DL): 2, (BL, FL): 2, (BL, U): 2,
    (BL, UB): 2, (BL, UBR): 2, (BL, UFL): 2, (BL, UL): 2, (BR, D): 2,
    (BR, DB): 2, (BR, DBL): 2, (BR, DFR): 2, (BR, DR): 2, (BR, FR): 2,
    (BR, U): 2, (BR, UB): 2, (BR, UBL): 2, (BR, UFR): 2, (BR, UR): 2,
    (D, F): 2, (D, FL): 2, (D, FR): 2, (D, L): 2, (D, R): 2, (DB, DF): 2,
    (DB, DFL): 2, (DB, DFR): 2, (DB, DL): 2, (DB, DR): 2, (DB, L): 2,
    (DB, R): 2, (DB, UB): 2, (DB, UBL): 2, (DB, UBR): 2, (DBL, DBR): 2,
    (DBL, DF): 2, (DBL, DFL): 2, (DBL, DFR): 2, (DBL, DR): 2, (DBL, FL): 2,
    (DBL, UB): 2, (DBL, UBL): 2, (DBL, UBR): 2, (DBL, UFL): 2,
    (DBL, UL): 2, (DBR, DF): 2, (DBR, DFL): 2, (DBR, DFR): 2, (DBR, DL): 2,
    (DBR, FR): 2, (DBR, UB): 2, (DBR, UBL): 2, (DBR, UBR): 2,
    (DBR, UFR): 2, (DBR, UR): 2, (DF, DL): 2, (DF, DR): 2, (DF, FL): 2,
    (DF, FR): 2, (DF, L): 2, (DF, R): 2, (DF, UF): 2, (DF, UFL): 2,
    (DF, UFR): 2, (DFL, DFR): 2, (DFL, DR): 2, (DFL, FR): 2, (DFL, UBL): 2,
    (DFL, UF): 2, (DFL, UFL): 2, (DFL, UFR): 2, (DFL, UL): 2, (DFR, DL): 2,
    (DFR, FL): 2, (DFR, UBR): 2, (DFR, UF): 2, (DFR, UFL): 2,
    (DFR, UFR): 2, (DFR, UR): 2, (DL, DR): 2, (DL, F): 2, (DL, FL): 2,
    (DL, UBL): 2, (DL, UFL): 2, (DL, UL): 2, (DR, F): 2, (DR, FR): 2,
    (DR, UBR): 2, (DR, UFR): 2, (DR, UR): 2, (F, L): 2, (F, R): 2,
    (F, U): 2, (F, UL): 2, (F, UR): 2, (FL, FR): 2, (FL, U): 2,
    (FL, UBL): 2, (FL, UF): 2, (FL, UFR): 2, (FL, UL): 2, (FR, U): 2,
    (FR, UBR): 2, (FR, UF): 2, (FR, UFL): 2, (FR, UR): 2, (L, U): 2,
    (L, UB): 2, (L, UF): 2, (R, U): 2, (R, UB): 2, (R, UF): 2, (UB, UF): 2,
    (UB, UFL): 2, (UB, UFR): 2, (UB, UL): 2, (UB, UR): 2, (UBL, UBR): 2,
    (UBL, UF): 2, (UBL, UFL): 2, (UBL, UFR): 2, (UBL, UR): 2, (UBR, UF): 2,
    (UBR, UFL): 2, (UBR, UFR): 2, (UBR, UL): 2, (UF, UL): 2, (UF, UR): 2,
    (UFL, UFR): 2, (UFL, UR): 2, (UFR, UL): 2, (UL, UR): 2,

    # DIST 3 PAIRS
    (B, DF): 3, (B, DFL): 3, (B, DFR): 3, (B, FL): 3, (B, FR): 3,
    (B, UF): 3, (B, UFL): 3, (B, UFR): 3, (BL, DF): 3, (BL, DFR): 3,
    (BL, DR): 3, (BL, F): 3, (BL, R): 3, (BL, UF): 3, (BL, UFR): 3,
    (BL, UR): 3, (BR, DF): 3, (BR, DFL): 3, (BR, DL): 3, (BR, F): 3,
    (BR, L): 3, (BR, UF): 3, (BR, UFL): 3, (BR, UL): 3, (D, UB): 3,
    (D, UBL): 3, (D, UBR): 3, (D, UF): 3, (D, UFL): 3, (D, UFR): 3,
    (D, UL): 3, (D, UR): 3, (DB, F): 3, (DB, FL): 3, (DB, FR): 3, (DB, U): 3,
    (DB, UFL): 3, (DB, UFR): 3, (DB, UL): 3, (DB, UR): 3, (DBL, F): 3,
    (DBL, FR): 3, (DBL, R): 3, (DBL, U): 3, (DBL, UF): 3, (DBL, UR): 3,
    (DBR, F): 3, (DBR, FL): 3, (DBR, L): 3, (DBR, U): 3, (DBR, UF): 3,
    (DBR, UL): 3, (DF, U): 3, (DF, UBL): 3, (DF, UBR): 3, (DF, UL): 3,
    (DF, UR): 3, (DFL, R): 3, (DFL, U): 3, (DFL, UB): 3, (DFL, UR): 3,
    (DFR, L): 3, (DFR, U): 3, (DFR, UB): 3, (DFR, UL): 3, (DL, FR): 3,
    (DL, R): 3, (DL, U): 3, (DL, UB): 3, (DL, UBR): 3, (DL, UF): 3,
    (DL, UFR): 3, (DR, FL): 3, (DR, L): 3, (DR, U): 3, (DR, UB): 3,
    (DR, UBL): 3, (DR, UF): 3, (DR, UFL): 3, (F, UB): 3, (F, UBL): 3,
    (F, UBR): 3, (FL, R): 3, (FL, UB): 3, (FL, UBR): 3, (FL, UR): 3,
    (FR, L): 3, (FR, UB): 3, (FR, UBL): 3, (FR, UL): 3, (L, UBR): 3,
    (L, UFR): 3, (L, UR): 3, (R, UBL): 3, (R, UFL): 3, (R, UL): 3,

    # DIST 4 PAIRS
    (B, F): 4, (BL, FR): 4, (BR, FL): 4, (D, U): 4, (DB, UF): 4,
    (DBL, UFR): 4, (DBR, UFL): 4, (DF, UB): 4, (DFL, UBR): 4,
    (DFR, UBL): 4, (DL, UR): 4, (DR, UL): 4, (L, R): 4,

    # UNDEFINED DIST PAIRS
    (B, UNKNOWN): np.nan, (BL, UNKNOWN): np.nan, (BR, UNKNOWN): np.nan,
    (D, UNKNOWN): np.nan, (DB, UNKNOWN): np.nan, (DBL, UNKNOWN): np.nan,
    (DBR, UNKNOWN): np.nan, (DF, UNKNOWN): np.nan, (DFL, UNKNOWN): np.nan,
    (DFR, UNKNOWN): np.nan, (DL, UNKNOWN): np.nan, (DR, UNKNOWN): np.nan,
    (F, UNKNOWN): np.nan, (FL, UNKNOWN): np.nan, (FR, UNKNOWN): np.nan,
    (L, UNKNOWN): np.nan, (R, UNKNOWN): np.nan, (U, UNKNOWN): np.nan,
    (UB, UNKNOWN): np.nan, (UBL, UNKNOWN): np.nan, (UBR, UNKNOWN): np.nan,
    (UF, UNKNOWN): np.nan, (UFL, UNKNOWN): np.nan, (UFR, UNKNOWN): np.nan,
    (UL, UNKNOWN): np.nan, (UNKNOWN, B): np.nan, (UNKNOWN, BL): np.nan,
    (UNKNOWN, BR): np.nan, (UNKNOWN, D): np.nan, (UNKNOWN, DB): np.nan,
    (UNKNOWN, DBL): np.nan, (UNKNOWN, DBR): np.nan, (UNKNOWN, DF): np.nan,
    (UNKNOWN, DFL): np.nan, (UNKNOWN, DFR): np.nan, (UNKNOWN, DL): np.nan,
    (UNKNOWN, DR): np.nan, (UNKNOWN, F): np.nan, (UNKNOWN, FL): np.nan,
    (UNKNOWN, FR): np.nan, (UNKNOWN, L): np.nan, (UNKNOWN, R): np.nan,
    (UNKNOWN, U): np.nan, (UNKNOWN, UB): np.nan, (UNKNOWN, UBL): np.nan,
    (UNKNOWN, UBR): np.nan, (UNKNOWN, UF): np.nan, (UNKNOWN, UFL): np.nan,
    (UNKNOWN, UFR): np.nan, (UNKNOWN, UL): np.nan, (UNKNOWN, UR): np.nan,
    (UR, UNKNOWN): np.nan, (UNKNOWN, UNKNOWN): np.nan,
}


# make distance symmetric
for (f1, f2), d in list(VIEW_INT_DIST.items()):
    VIEW_INT_DIST[(f2, f1)] = d


# Make string based version
VIEW_CODE_DIST = {
    (INT_TO_CODE[f1], INT_TO_CODE[f2]): d
    for (f1, f2), d in VIEW_INT_DIST.items()
}


def RhombicuboctahedronDistanceDemo():
    import utool as ut
    def rhombicuboctahedro_faces():
        """ yields names of all 26 rhombicuboctahedron faces"""
        face_axes = [['up', 'down'], ['front', 'back'], ['left', 'right']]
        ordering = {f: p for p, fs in enumerate(face_axes) for f in fs}
        for i in range(1, len(face_axes) + 1):
            for axes in list(ut.combinations(face_axes, i)):
                for combo in ut.product(*axes):
                    sortx = ut.argsort(ut.take(ordering, combo))
                    face = tuple(ut.take(combo, sortx))
                    yield face

    # Each face is a node.
    import networkx as nx
    G = nx.Graph()
    faces = list(rhombicuboctahedro_faces())
    G.add_nodes_from(faces)

    # A fase is connected if they share an edge or a vertex
    # TODO: is there a more general definition?
    face_axes = [['up', 'down'], ['front', 'back'], ['left', 'right']]
    ordering = {f: p for p, fs in enumerate(face_axes) for f in fs}

    # In this case faces might share an edge or vertex if their names intersect
    edges = []
    for face1, face2 in ut.combinations(faces, 2):
        set1 = set(face1)
        set2 = set(face2)
        if len(set1.intersection(set2)) > 0:
            diff1 = set1.difference(set2)
            diff2 = set2.difference(set1)
            sortx1 = ut.argsort(ut.take(ordering, diff1))
            sortx2 = ut.argsort(ut.take(ordering, diff2))
            # If they share a name that is on opposite poles, then they cannot
            # share an edge or vertex.
            if not list(set(sortx1).intersection(set(sortx2))):
                edges.append((face1, face2))
                # print('-----')
                # print('Edge: {} {}'.format(face1, face2))
                # print('diff1 = {!r}'.format(diff1))
                # print('diff2 = {!r}'.format(diff2))
    G.add_edges_from(edges)

    # Build distance lookup table
    lookup = {}
    for face1, face2 in ut.combinations(faces, 2):
        # key = tuple(sorted([''.join(face1), ''.join(face2)]))
        key = tuple(sorted([face1, face2]))
        dist = nx.shortest_path_length(G, face1, face2)
        lookup[key] = dist

    def convert(face):
        if face is None:
            return 'UNKNOWN'
        else:
            return ''.join([p[0].upper() for p in face])

    dist_lookup = {}
    dist_lookup = {
        (convert(k1), convert(k2)): d
        for (k1, k2), d in lookup.items()
    }

    for face in faces:
        dist_lookup[(convert(face), convert(face))] = 0
        dist_lookup[(convert(face), convert(None))] = None
        dist_lookup[(convert(None), convert(face))] = None
    dist_lookup[(convert(None), convert(None))] = None

    # z = {'({}, {})'.format(k1, k2): d for (k1, k2), d in dist_lookup.items()}
    # for i in range(0, 5):
    #     print(ub.repr2({k: v for k, v in z.items() if v == i}, si=True))
    # i = None
    # print(ub.repr2({k: v for k, v in z.items() if v == i}, si=True))

    # z = ut.sort_dict(z, 'vals')
    # print(ub.repr2(z, nl=2, si=True))

    # if False:
    #     from ibeis import constants as const
    #     VIEW = const.VIEW
    #     viewint_dist_lookup = {
    #         (VIEW.CODE_TO_INT[f1], VIEW.CODE_TO_INT[f2]): d
    #         (f1, f2) for (f1, f2), d in viewcode_dist_lookup.items()
    #     }

    # for k, v in viewcode_dist_lookup.items():
    #     if 'up' not in k[0] and 'down' not in k[0]:
    #         if 'up' not in k[1] and 'down' not in k[1]:
    #             print(k, v)

    def visualize_connection_graph():
        # node_to_group = {f: str(len(f)) for f in faces}
        node_to_group = {}
        for f in faces:
            if 'up' in f:
                node_to_group[f] = '0.' + str(len(f))
            elif 'down' in f:
                node_to_group[f] = '1.' + str(len(f))
            else:
                node_to_group[f] = '2.' + str(len(f))
        nx.set_node_attributes(G, name='groupid', values=node_to_group)

        node_to_label = {f: ''.join(ut.take_column(f, 0)).upper() for f in faces}
        nx.set_node_attributes(G, name='label', values=node_to_label)

        import plottool_ibeis as pt
        pt.qt4ensure()
        pt.show_nx(G, prog='neato', groupby='groupid')

    visualize_connection_graph()
