# -*- coding: utf-8 -*-
def pytest_addoption(parser):
    # This needs to be in the project root not in wbia/conftest.py otherwise
    # e.g. "pytest --gui" doesn't work.
    parser.addoption('--fixme', action='store_true')
    parser.addoption('--gui', action='store_true')
    parser.addoption('--show', action='store_true')
    parser.addoption('--tomcat', action='store_true')
    parser.addoption('--web', action='store_true')
    parser.addoption('--weird', action='store_true')
