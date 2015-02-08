
SIMPLE_MERGE = True
if SIMPLE_MERGE:
else:
    # Find cases where vsone and prior disagree
    en_flags1 = np.in1d(fm_vsone.T[0], fm_prior.T[0])
    en_flags2 = np.in1d(fm_prior.T[0], fm_vsone.T[0])
    ne_flags1 = np.in1d(fm_vsone.T[1], fm_prior.T[1])
    ne_flags2 = np.in1d(fm_prior.T[1], fm_vsone.T[1])
    print(fm_vsone.compress(en_flags1, axis=0))
    print(fm_prior.compress(en_flags2, axis=0))
    print(fm_vsone.compress(ne_flags1, axis=0))
    print(fm_prior.compress(ne_flags2, axis=0))

    # Cases where matches are mutually exclusive
    # (the other method did not find a match or match to these indicies)
    mutex_flags1 = np.logical_and(~en_flags1, ~ne_flags1)
    mutex_flags2 = np.logical_and(~en_flags2, ~ne_flags2)
    fm_vsone_mutex = fm_vsone.compress(mutex_flags1, axis=0)
    fm_prior_mutex = fm_prior.compress(mutex_flags2, axis=0)
    print(fm_vsone_mutex)
    print(fm_prior_mutex)
    fm_both = np.vstack([ fm_both, fm_vsone_mutex, fm_prior_mutex ])
    print(fm_both)
