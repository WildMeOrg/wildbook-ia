# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


def rhombicuboctahedron():
    import vtk

    # First, you need to store the vertex locations.

    import numpy as np

    fu = 1  # full unit
    hu = 0.5  # half unit
    d = np.sqrt((fu ** 2) / 2)  # diag
    hh = hu + d  # half height

    # left view faces us

    import utool as ut
    import six
    import itertools

    counter = ut.partial(six.next, itertools.count(0))

    vertex_locations = vtk.vtkPoints()
    vertex_locations.SetNumberOfPoints(24)

    p1, p2, p3 = np.array([(-hu, -hu, hh), (hu, -hu, hh), (hu, hu, hh), (-hu, hu, hh)]).T
    plist = [p1, p2, p3]

    # three of the six main faces
    # perms = list(itertools.permutations((0, 1, 2), 3))
    perms = [(0, 1, 2), (0, 2, 1), (2, 0, 1)]

    vertex_array = []

    # VERTEXES
    # left, up, back
    vplist = ['L', 'U', 'B', 'R', 'D', 'F']
    vpdict = {}
    print('perms = %r' % (perms,))
    for x in range(3):
        vp = vplist[x]
        p = np.vstack(ut.take(plist, perms[x])).T
        counts = [counter() for z in range(4)]
        vpdict[vp] = counts
        vertex_array.extend(p.tolist())
        vertex_locations.SetPoint(counts[0], p[0])
        vertex_locations.SetPoint(counts[1], p[1])
        vertex_locations.SetPoint(counts[2], p[2])
        vertex_locations.SetPoint(counts[3], p[3])

    # three more of the six main faces
    perms = [(0, 1, 2), (0, 2, 1), (2, 0, 1)]
    plist[-1] = -plist[-1]
    # right, down, front
    print('perms = %r' % (perms,))
    for x in range(3):
        p = np.vstack(ut.take(plist, perms[x])).T
        counts = [counter() for z in range(4)]
        vp = vplist[x + 3]
        vpdict[vp] = counts
        vertex_array.extend(p.tolist())
        vertex_locations.SetPoint(counts[0], p[0])
        vertex_locations.SetPoint(counts[1], p[1])
        vertex_locations.SetPoint(counts[2], p[2])
        vertex_locations.SetPoint(counts[3], p[3])

    pd = vtk.vtkPolyData()
    pd.SetPoints(vertex_locations)

    polygon_faces = vtk.vtkCellArray()

    face_dict = {
        'L': [vpdict['L'][0], vpdict['L'][1], vpdict['L'][2], vpdict['L'][3]],
        'D': [vpdict['D'][0], vpdict['D'][1], vpdict['D'][2], vpdict['D'][3]],
        'U': [vpdict['U'][0], vpdict['U'][1], vpdict['U'][2], vpdict['U'][3]],
        'F': [vpdict['F'][0], vpdict['F'][1], vpdict['F'][2], vpdict['F'][3]],
        'R': [vpdict['R'][0], vpdict['R'][1], vpdict['R'][2], vpdict['R'][3]],
        'B': [vpdict['B'][0], vpdict['B'][1], vpdict['B'][2], vpdict['B'][3]],
        'FL': [vpdict['L'][0], vpdict['L'][3], vpdict['F'][2], vpdict['F'][3]],
        'BL': [vpdict['L'][1], vpdict['L'][2], vpdict['B'][2], vpdict['B'][3]],
        'UL': [vpdict['L'][2], vpdict['L'][3], vpdict['U'][3], vpdict['U'][2]],
        'DL': [vpdict['L'][0], vpdict['L'][1], vpdict['D'][2], vpdict['D'][3]],
        'UFL': [vpdict['L'][3], vpdict['F'][2], vpdict['U'][3]],
        'DFL': [vpdict['L'][0], vpdict['F'][3], vpdict['D'][3]],
        'UBL': [vpdict['L'][2], vpdict['B'][2], vpdict['U'][2]],
        'DBL': [vpdict['L'][1], vpdict['B'][3], vpdict['D'][2]],
        'UFR': [vpdict['R'][3], vpdict['F'][1], vpdict['U'][0]],
        'DFR': [vpdict['R'][0], vpdict['F'][0], vpdict['D'][0]],
        'UBR': [vpdict['R'][2], vpdict['B'][1], vpdict['U'][1]],
        'DBR': [vpdict['R'][1], vpdict['B'][0], vpdict['D'][1]],
        'FR': [vpdict['R'][3], vpdict['R'][0], vpdict['F'][0], vpdict['F'][1]],
        'BR': [vpdict['R'][2], vpdict['R'][1], vpdict['B'][0], vpdict['B'][1]],
        'UR': [vpdict['R'][3], vpdict['R'][2], vpdict['U'][1], vpdict['U'][0]],
        'DR': [vpdict['R'][1], vpdict['R'][0], vpdict['D'][0], vpdict['D'][1]],
        'DF': [vpdict['F'][0], vpdict['F'][3], vpdict['D'][3], vpdict['D'][0]],
        'DB': [vpdict['B'][3], vpdict['B'][0], vpdict['D'][1], vpdict['D'][2]],
        'UF': [vpdict['F'][1], vpdict['F'][2], vpdict['U'][3], vpdict['U'][0]],
        'UB': [vpdict['B'][2], vpdict['B'][1], vpdict['U'][1], vpdict['U'][2]],
    }

    for key, vert_ids in face_dict.items():
        # if key != 'L':
        #    continue
        if len(vert_ids) == 4:
            q = vtk.vtkQuad()
        else:
            q = vtk.vtkTriangle()
        for count, idx in enumerate(vert_ids):
            q.GetPointIds().SetId(count, idx)
        polygon_faces.InsertNextCell(q)

    # Next you create a vtkPolyData to store your face and vertex information
    # that
    # represents your polyhedron.
    pd = vtk.vtkPolyData()
    pd.SetPoints(vertex_locations)
    pd.SetPolys(polygon_faces)

    face_stream = vtk.vtkIdList()
    face_stream.InsertNextId(polygon_faces.GetNumberOfCells())
    vertex_list = vtk.vtkIdList()

    polygon_faces.InitTraversal()
    while polygon_faces.GetNextCell(vertex_list) == 1:
        face_stream.InsertNextId(vertex_list.GetNumberOfIds())

        for j in range(vertex_list.GetNumberOfIds()):
            face_stream.InsertNextId(vertex_list.GetId(j))

    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(vertex_locations)
    ug.InsertNextCell(vtk.VTK_POLYHEDRON, face_stream)

    # writer = vtk.vtkUnstructuredGridWriter()
    # writer.SetFileName("rhombicuboctahedron.vtk")
    # # writer.SetInputData(ug)
    # writer.SetInput(ug)
    # writer.Write()

    mapper = vtk.vtkDataSetMapper()
    mapper.SetInput(ug)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    if 1:
        # Read the image data from a file
        import utool as ut

        textureCoords = vtk.vtkFloatArray()
        textureCoords.SetNumberOfComponents(3)
        # coords = ut.take(vertex_array, face_dict['L'])
        # for coord in coords:
        #    textureCoords.InsertNextTuple(tuple(coord))
        textureCoords.InsertNextTuple((0, 0, 0))
        textureCoords.InsertNextTuple((1, 0, 0))
        textureCoords.InsertNextTuple((1, 1, 0))
        textureCoords.InsertNextTuple((0, 1, 0))

        # Create texture object
        fpath = ut.grab_test_imgpath('zebra.png')
        reader = vtk.vtkPNGReader()
        reader.SetFileName(fpath)

        texture = vtk.vtkTexture()
        texture.SetInput(reader.GetOutput())
        texture.RepeatOff()
        texture.InterpolateOff()

        ptdat = pd.GetPointData()
        ptdat.SetTCoords(textureCoords)

        actor.SetTexture(texture)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)

    renw = vtk.vtkRenderWindow()
    renw.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renw)

    ren.ResetCamera()
    renw.Render()
    iren.Start()


def rhombic_dodecahedron():
    # http://www.vtk.org/pipermail/vtkusers/2014-September/085077.html
    import vtk

    # This is a Rhombic Dodecahedron.

    # First, you need to store the vertex locations.
    vertex_locations = vtk.vtkPoints()
    vertex_locations.SetNumberOfPoints(14)
    vertex_locations.SetPoint(0, (-0.816497, -0.816497, 0.00000))
    vertex_locations.SetPoint(1, (-0.816497, 0.000000, -0.57735))
    vertex_locations.SetPoint(2, (-0.816497, 0.000000, 0.57735))
    vertex_locations.SetPoint(3, (-0.816497, 0.816497, 0.00000))
    vertex_locations.SetPoint(4, (0.000000, -0.816497, -0.57735))
    vertex_locations.SetPoint(5, (0.000000, -0.816497, 0.57735))
    vertex_locations.SetPoint(6, (0.000000, 0.000000, -1.15470))
    vertex_locations.SetPoint(7, (0.000000, 0.000000, 1.15470))
    vertex_locations.SetPoint(8, (0.000000, 0.816497, -0.57735))
    vertex_locations.SetPoint(9, (0.000000, 0.816497, 0.57735))
    vertex_locations.SetPoint(10, (0.816497, -0.816497, 0.00000))
    vertex_locations.SetPoint(11, (0.816497, 0.000000, -0.57735))
    vertex_locations.SetPoint(12, (0.816497, 0.000000, 0.57735))
    vertex_locations.SetPoint(13, (0.816497, 0.816497, 0.00000))

    # Next, you describe the polygons that represent the faces using the vertex
    # indices in the vtkPoints that stores the vertex locations. There are a
    # number
    # of ways to do this that you can find in examples on the Wiki.

    polygon_faces = vtk.vtkCellArray()

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 7)
    q.GetPointIds().SetId(1, 12)
    q.GetPointIds().SetId(2, 10)
    q.GetPointIds().SetId(3, 5)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 7)
    q.GetPointIds().SetId(1, 12)
    q.GetPointIds().SetId(2, 13)
    q.GetPointIds().SetId(3, 9)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 7)
    q.GetPointIds().SetId(1, 9)
    q.GetPointIds().SetId(2, 3)
    q.GetPointIds().SetId(3, 2)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 7)
    q.GetPointIds().SetId(1, 2)
    q.GetPointIds().SetId(2, 0)
    q.GetPointIds().SetId(3, 5)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 6)
    q.GetPointIds().SetId(1, 11)
    q.GetPointIds().SetId(2, 10)
    q.GetPointIds().SetId(3, 4)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 6)
    q.GetPointIds().SetId(1, 4)
    q.GetPointIds().SetId(2, 0)
    q.GetPointIds().SetId(3, 1)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 6)
    q.GetPointIds().SetId(1, 1)
    q.GetPointIds().SetId(2, 3)
    q.GetPointIds().SetId(3, 8)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 6)
    q.GetPointIds().SetId(1, 8)
    q.GetPointIds().SetId(2, 13)
    q.GetPointIds().SetId(3, 11)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 10)
    q.GetPointIds().SetId(1, 11)
    q.GetPointIds().SetId(2, 13)
    q.GetPointIds().SetId(3, 12)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 13)
    q.GetPointIds().SetId(1, 8)
    q.GetPointIds().SetId(2, 3)
    q.GetPointIds().SetId(3, 9)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 3)
    q.GetPointIds().SetId(1, 1)
    q.GetPointIds().SetId(2, 0)
    q.GetPointIds().SetId(3, 2)
    polygon_faces.InsertNextCell(q)

    q = vtk.vtkQuad()
    q.GetPointIds().SetId(0, 0)
    q.GetPointIds().SetId(1, 4)
    q.GetPointIds().SetId(2, 10)
    q.GetPointIds().SetId(3, 5)
    polygon_faces.InsertNextCell(q)

    # Next you create a vtkPolyData to store your face and vertex information
    # that
    # represents your polyhedron.
    pd = vtk.vtkPolyData()
    pd.SetPoints(vertex_locations)
    pd.SetPolys(polygon_faces)

    # If you wanted to be able to load in the saved file and select the entire
    # polyhedron, you would need to save it as a vtkUnstructuredGrid, and you
    # would
    # need to put the data into a vtkPolyhedron. This is a bit more involved
    # than
    # the vtkPolyData that I used above. For a more in-depth discussion, see:
    # http://www.vtk.org/Wiki/VTK/Polyhedron_Support

    # Based on the link above, I need to construct a face stream:
    face_stream = vtk.vtkIdList()
    face_stream.InsertNextId(polygon_faces.GetNumberOfCells())
    vertex_list = vtk.vtkIdList()

    polygon_faces.InitTraversal()
    while polygon_faces.GetNextCell(vertex_list) == 1:
        face_stream.InsertNextId(vertex_list.GetNumberOfIds())

        for j in range(vertex_list.GetNumberOfIds()):
            face_stream.InsertNextId(vertex_list.GetId(j))

    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(vertex_locations)
    ug.InsertNextCell(vtk.VTK_POLYHEDRON, face_stream)

    # --------------#
    # output stuff #
    # --------------#

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName('rhombic_dodecahedron.vtk')
    # writer.SetInputData(ug)
    writer.SetInput(ug)
    writer.Write()

    # ---------------------#
    # visualization stuff #
    # ---------------------#
    # mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(pd)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInput(ug)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    ren = vtk.vtkRenderer()
    ren.AddActor(actor)

    renw = vtk.vtkRenderWindow()
    renw.AddRenderer(ren)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renw)

    ren.ResetCamera()
    renw.Render()
    iren.Start()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.plottool.test_vtk_poly
        python -m wbia.plottool.test_vtk_poly --allexamples
        python plottool/test_vtk_poly.py
    """
    rhombicuboctahedron()
