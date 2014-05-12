if __name__ == '__main__':
    import utool
    unflat_list = [(1, 2), (), (3, 4, 5), (6,), (7,)]
    flat_list, reverse_list = utool.invertable_flatten(unflat_list)
    unflat_list2 = utool.unflatten(flat_list, reverse_list)
    assert unflat_list2 == unflat_list and id(unflat_list) != id(unflat_list2)
