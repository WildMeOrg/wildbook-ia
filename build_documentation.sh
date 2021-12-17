#!/bin/bash
# python -m utool.util_setup --exec-autogen_sphinx_apidoc  --nomake

make -C _docs html
rm -rf docs/
mkdir -p docs/
cp -r _docs/_build/html/* docs/
touch docs/.nojekyll
