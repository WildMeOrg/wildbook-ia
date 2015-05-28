import numpy as np


def make_projetion_matrix(vec_dim):
    sig_dim = 64   # binary signature dimension
    # Initialize random matrix
    shape = (vec_dim, vec_dim)
    randstate = np.random.RandomState(0)
    rand_mat = randstate.normal(10, size=shape)
    # Q is orthogonal
    # R is upper triangular
    Q, R = np.linalg.qr(rand_mat)
    # alias Q to be the projection matrix P
    P = Q[0:sig_dim]
    return P


def get_vecs_hamming_encoding(vecs):
    """
    Args:
        vecs (ndarray): descriptors assigned to a single word
        P (ndarray): random orthognoal projection matrix for that word


    Exmaple:
        >>> np.random.seed(0)
        >>> vecs = np.random.rand(10, 128)
    """
    # Embeding Step to compute binary signature:
    vec_dim = vecs.shape[1]  # raw vector dimension
    P = make_projetion_matrix(vec_dim)
    z = P.dot(vecs.T)
    thresh = np.median(z, axis=0)
    # find median values
    hamming_sig_bools = (z > thresh[None, :]).T

    # mapping to binary encoding
    sig_dim = hamming_sig_bools.shape[1]
    assert sig_dim <= 64, 'only up to 64 dimensions implemented'
    basis = np.power(np.uint64(2), np.arange(sig_dim).astype(np.uint64))
    haming_codes = [basis[bools].sum() for bools in hamming_sig_bools]
    return haming_codes
