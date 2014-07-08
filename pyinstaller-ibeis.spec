# -*- mode: python -*-
a = Analysis(['main.py'],
             pathex=['/home/zack/code/ibeis'],
             hiddenimports=['sklearn.utils.sparsetools._graph_validation', 'sklearn.utils.sparsetools._graph_tools',  'scipy.special._ufuncs_cxx',
                            'sklearn.utils.lgamma', 'sklearn.utils.weight_vector', 'sklearn.neighbors.typedefs', 'mpl_toolkits.axes_grid1'],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
a.binaries.append(('libhesaff.so', '../hesaff/build/libhesaff.so', 'BINARY'))
a.binaries.append(('libpyrf.so', '../pyrf/build/libpyrf.so', 'BINARY'))
a.binaries.append(('libflann.so', '../flann/build/lib/libflann.so', 'BINARY'))
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='main',
          debug=False,
          strip=None,
          upx=True,
          console=True )
