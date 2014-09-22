%timeit np.meshgrid(qfxs, dfxs, indexing='ij')
x, y = np.meshgrid(qfxs, dfxs, indexing='ij')
np.array(zip(x.flat, y.flat))
x, y =

fm_ = np.array(zip(*[x.flat for x in np.meshgrid(qfxs, dfxs, indexing='ij')]), dtype=np.int32)
fm_ = np.fromiter(zip(*[x.flat for x in np.meshgrid(qfxs, dfxs, indexing='ij')]))
%timeit np.fromiter(chain(*[x.flat for x in np.meshgrid(qfxs, dfxs, indexing='ij')]), np.int32)

%timeit np.array(zip(*[x.flat for x in np.meshgrid(qfxs, dfxs, indexing='ij')]), dtype=np.int32)
%timeit fm_ = np.vstack(np.dstack(np.meshgrid(qfxs, dfxs, indexing='ij')))
%timeit fm_ = np.dstack(np.meshgrid(qfxs, dfxs, indexing='ij', copy=False)).reshape(qfxs.size * dfxs.size, 2)
%timeit fm_ = np.concatenate(np.meshgrid(qfxs, dfxs, indexing='ij'), axis=1).T
.reshape(qfxs.size * dfxs.size, 2)
#fm_ = np.vstack(np.dstack(np.meshgrid(qfxs, dfxs, indexing='ij')))
