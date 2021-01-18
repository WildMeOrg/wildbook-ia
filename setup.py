#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from os.path import exists
from collections import OrderedDict

from setuptools import find_packages
from skbuild import setup


def native_mb_python_tag(plat_impl=None, version_info=None):
    """
    Example:
        >>> print(native_mb_python_tag())
        >>> print(native_mb_python_tag('PyPy', (2, 7)))
        >>> print(native_mb_python_tag('CPython', (3, 8)))
    """
    if plat_impl is None:
        import platform

        plat_impl = platform.python_implementation()

    if version_info is None:
        import sys

        version_info = sys.version_info

    major, minor = version_info[0:2]
    ver = '{}{}'.format(major, minor)

    if plat_impl == 'CPython':
        # TODO: get if cp27m or cp27mu
        impl = 'cp'
        if ver == '27':
            IS_27_BUILT_WITH_UNICODE = True  # how to determine this?
            if IS_27_BUILT_WITH_UNICODE:
                abi = 'mu'
            else:
                abi = 'm'
        else:
            if ver == '38':
                # no abi in 38?
                abi = ''
            else:
                abi = 'm'
        mb_tag = '{impl}{ver}-{impl}{ver}{abi}'.format(**locals())
    elif plat_impl == 'PyPy':
        abi = ''
        impl = 'pypy'
        ver = '{}{}'.format(major, minor)
        mb_tag = '{impl}-{ver}'.format(**locals())
    else:
        raise NotImplementedError(plat_impl)
    return mb_tag


def parse_version(fpath='wbia/__init__.py'):
    """
    Statically parse the version number from a python file


    """
    import ast

    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)

    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s

    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


def parse_long_description(fpath='README.rst'):
    """
    Reads README text, but doesn't break if README does not exist.
    """
    if exists(fpath):
        with open(fpath, 'r') as file:
            return file.read()
    return ''


def parse_requirements(fname='requirements.txt', with_version=True):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=True): if true include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
        python -c "import setup; print(chr(10).join(setup.parse_requirements(with_version=True)))"
    """
    from os.path import exists
    import re

    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


NAME = 'wildbook-ia'


MB_PYTHON_TAG = native_mb_python_tag()  # NOQA

AUTHORS = [
    'Jason Parham',
    'Dr. Jon Crall',
    'Dr. Charles Stewart',
    'Drew Blount',
    'Ben Scheiner',
    'Wild Me Developers',
    'Karen Chan',
    'Michael Mulich',
    'Hendrik Weideman',
    'A. Batbouta',
    'A. Beard',
    'Z. Jablons',
    'D. Lowe',
    'Z. Rutfield',
    'K. Southerland',
    'A. Weinstock',
    'J. Wrona',
]
AUTHOR_EMAIL = 'dev@wildme.org'
URL = 'https://github.com/WildMeOrg/wildbook-ia'
LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Wildbook IA (WBIA) - Machine learning service for the WildBook project'
KEYWORDS = [
    'wildbook',
    'wildme',
    'ibeis',
    'ecological',
    'wildlife',
    'conservation',
    'machine learning',
    'ai',
    'hotspotter',
    'detection',
    'classification',
    'animal ID',
    're-id',
    're-identification',
    'flukebook',
]


KWARGS = OrderedDict(
    name=NAME,
    author=', '.join(AUTHORS),
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=parse_long_description('README.rst'),
    long_description_content_type='text/x-rst',
    keywords=', '.join(KEYWORDS),
    url=URL,
    license=LICENSE,
    install_requires=parse_requirements('requirements/runtime.txt'),
    extras_require={
        'all': parse_requirements('requirements.txt'),
        'tests': parse_requirements('requirements/tests.txt'),
        'build': parse_requirements('requirements/build.txt'),
        'runtime': parse_requirements('requirements/runtime.txt'),
        'optional': parse_requirements('requirements/optional.txt'),
        'postgres': parse_requirements('requirements/postgres.txt'),
    },
    # --- VERSION ---
    # The following settings retreive the version from git.
    # See https://github.com/pypa/setuptools_scm/ for more information
    setup_requires=['setuptools_scm'],
    use_scm_version={
        'write_to': 'wbia/_version.py',
        'write_to_template': '__version__ = "{version}"',
        'tag_regex': '^(?P<prefix>v)?(?P<version>[^\\+]+)(?P<suffix>.*)?$',
        'local_scheme': 'dirty-tag',
    },
    packages=find_packages(),
    package_dir={'wbia': 'wbia'},
    python_requires='>=3.6, <4',
    include_package_data=True,
    # List of classifiers available at:
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA :: 11.0',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Unix',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/WildMeOrg/wildbook-ia/issues',
        'Funding': 'https://www.wildme.org/donate/',
        'Say Thanks!': 'https://community.wildbook.org',
        'Source': URL,
    },
    entry_points="""\
    [console_scripts]
    wbia-init-testdbs = wbia.cli.testdbs:main
    wbia-convert-hsdb = wbia.cli.convert_hsdb:main
    wbia-migrate-sqlite-to-postgres = wbia.cli.migrate_sqlite_to_postgres:main
    wbia-compare-databases = wbia.cli.compare_databases:main
    """,
)

if __name__ == '__main__':
    """
    python -c "import wbia; print(wbia.__file__)"
    """
    setup(**KWARGS)
