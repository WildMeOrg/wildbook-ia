#Seems like this wont work

## -*- coding: utf-8 -*-
#from __future__ import absolute_import, division, print_function, unicode_literals
#import utool as ut
#import luigi
#(print, rrr, profile) = ut.inject2(__name__, '[luigi_backend]')


#def execute_luigi_task(class_):
#    classname_long = ut.get_classname(HelloWorldTask, local=True)
#    luigi_namespace = getattr(class_, 'task_namespace', None)
#    task_name_parts = []
#    if luigi_namespace is not None:
#        task_name_parts += [luigi_namespace]
#    task_name_parts += [classname_long]
#    luigi_taskname = '.'.join(task_name_parts)
#    luigi.run([luigi_taskname, '--workers', '1', '--local-scheduler'])


#class HelloWorldTask(luigi.Task):
#    r"""
#    CommandLine:
#        python -m ibeis.luigi_backend --exec-ibeis.luigi_backend.HelloWorldTask --show

#    Example:
#        >>> # DISABLE_DOCTEST
#        >>> from ibeis.luigi_backend import *  # NOQA
#        >>> class_ = HelloWorldTask
#        >>> execute_luigi_task(class_)
#    """
#    task_namespace = 'dummy_namespace'

#    def run(self):
#        print("{task} says: Hello world!".format(task=self.__class__.__name__))


#class FeatureTask(luigi.Task):
#    param = luigi.Parameter(default=42)

#    def run(self):
#        print('preproc chip')


#class KeypointTask(luigi.Task):
#    param = luigi.Parameter(default=42)

#    def run(self):
#        print("{task} says: Hello world!".format(task=self.__class__.__name__))


#if __name__ == '__main__':
#    r"""
#    CommandLine:
#        python -m ibeis.luigi_backend
#        python -m ibeis.luigi_backend --allexamples
#    """
#    import multiprocessing
#    multiprocessing.freeze_support()  # for win32
#    import utool as ut  # NOQA
#    ut.doctest_funcs()
