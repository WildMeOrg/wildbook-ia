import random
import sys


def loss(x, y):
    return min(abs(x - y), 1 - abs(x - y))

for i in range(10 ** 6):
    x = random.uniform(0.0, 1.0)
    y = random.uniform(0.0, 1.0)
    l = loss(x, y)

    try:
        if x < y:
            inner_dist = y - x
            outer_dist = 1.0 - y + x
            if inner_dist < outer_dist:
                assert l == inner_dist
            else:
                assert l == outer_dist
        elif y < x:
            inner_dist = x - y
            outer_dist = 1.0 - x + y
            if inner_dist < outer_dist:
                assert l == inner_dist
            else:
                assert l == outer_dist
        else:
            inner_dist = x - y
            outer_dist = 1.0 - x + y
            assert l == inner_dist
            assert inner_dist == 0.0 and outer_dist == 1.0
    except AssertionError:
        print(x, y, l, inner_dist, outer_dist)
        t, v, tb = sys.exc_info()
        raise t, v, tb
