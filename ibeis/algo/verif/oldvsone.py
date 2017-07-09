# @profile
# def edge_hashids(samples):
#     qvuuids = samples.annots1.visual_uuids
#     dvuuids = samples.annots2.visual_uuids
#     # edge_uuids = [ut.combine_uuids(uuids)
#     #                for uuids in zip(qvuuids, dvuuids)]
#     edge_hashids = [make_edge_hashid(uuid1, uuid2) for uuid1, uuid2 in zip(qvuuids, dvuuids)]
#     # edge_uuids = [combine_2uuids(uuid1, uuid2)
#     #                for uuid1, uuid2 in zip(qvuuids, dvuuids)]
#     return edge_hashids

# @profile
# def edge_hashid(samples):
#     edge_hashids = samples.edge_hashids()
#     edge_hashid = ut.hashstr_arr27(edge_hashids, 'edges', hashlen=32,
#                                    pathsafe=True)
#     return edge_hashid

# @profile
# def make_edge_hashid(uuid1, uuid2):
#     """
#     Slightly faster than using ut.combine_uuids, because we condense and don't
#     bother casting back to UUIDS
#     """
#     sep_str = '-'
#     sep_byte = six.b(sep_str)
#     pref = six.b('{}2'.format(sep_str))
#     combined_bytes = pref + sep_byte.join([uuid1.bytes, uuid2.bytes])
#     bytes_sha1 = hashlib.sha1(combined_bytes)
#     # Digest them into a hash
#     hashbytes_20 = bytes_sha1.digest()
#     hashbytes_16 = hashbytes_20[0:16]
#     # uuid_ = uuid.UUID(bytes=hashbytes_16)
#     return hashbytes_16

