from __future__ import absolute_import, division, print_function
import weakref


__DUMMY_WEAKREFS__ = []


class Dummy(object):
    def __init__(self):
        self.myvar = 42

    def notify(self):
        print(self.myvar)

    def register_dummy(self):
        global __DUMMY_WEAKREFS__
        self_weakref = weakref.ref(self)
        __DUMMY_WEAKREFS__.append(self_weakref)

    def unregister_dummy(self):
        global __DUMMY_WEAKREFS__
        self_weakref = weakref.ref(self)
        __DUMMY_WEAKREFS__.remove(self_weakref)


if __name__ == '__main__':
    self = Dummy()
    self.register_dummy()
    assert len(__DUMMY_WEAKREFS__) == 1
    self.unregister_dummy()
    assert len(__DUMMY_WEAKREFS__) == 0
