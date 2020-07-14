#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from os.path import dirname, join, exists
import sys


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        pandoc --from=markdown --to=rst --output=README.rst README.md
        python -c "import setup; print(setup.parse_description())"
    """
    readme_fpath = join(dirname(__file__), 'README.rst')
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
        with open(readme_fpath, 'r') as f:
            text = f.read()
        return text
    return ''


def parse_requirements(fname='requirements.txt', with_version=False):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if true include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
        python -c "import setup; print(chr(10).join(setup.parse_requirements(with_version=True)))"
    """
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


# @setman.register_command
def autogen_explicit_imports():
    """
    Excpliticly generated injectable code in order to aid auto complete
    programs like jedi as well as allow for a more transparent stack trace.

    python -m wbia dev_autogen_explicit_injects
    """
    import wbia  # NOQA
    from wbia.control import controller_inject

    controller_inject.dev_autogen_explicit_injects()


NAME = 'wildbook-ia'


if __name__ == '__main__':
    extras_require = {
        'all': parse_requirements('requirements.txt'),
        'build': parse_requirements('requirements/build.txt'),
        'tests': parse_requirements('requirements/tests.txt'),
        'optional': parse_requirements('requirements/optional.txt'),
    }
    install_requires = parse_requirements('requirements/runtime.txt')
    print('install_requires = {!r}'.format(install_requires))

    from setuptools import setup, find_packages

    kwargs = dict(
        name=NAME,
        description='Image Based Ecological Information System',
        long_description=parse_description(),
        long_description_content_type='text/x-rst',
        author='Jon Crall, Jason Parham',
        author_email='dev@wildme.org',
        # The following settings retreive the version from git.
        # See https://github.com/pypa/setuptools_scm/ for more information
        setup_requires=['setuptools_scm'],
        use_scm_version={
            'write_to': 'wbia/_version.py',
            'write_to_template': '__version__ = "{version}"',
            'tag_regex': '^(?P<prefix>v)?(?P<version>[^\\+]+)(?P<suffix>.*)?$',
            'local_scheme': 'dirty-tag',
        },
        install_requires=install_requires,
        extras_require=extras_require,
        entry_points={
            'console_scripts': [
                # Register specific python functions as command line scripts
                'wbia=wbia.__main__:run_wbia',
            ],
        },
        # cython_files=CYTHON_FILES,
        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 4 - Beta',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 3',
        ],
        packages=find_packages('.'),
    )
    setup(**kwargs)
