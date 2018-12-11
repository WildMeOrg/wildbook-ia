#!/bin/bash
# rm -rf _doc
# rm -rf _page
# python -m utool.util_setup --exec-autogen_sphinx_apidoc  --nomake

make -C _doc json
rm -rf _page/
mkdir -p _page/
cp -r _doc/_build/html/* _page
touch _page/.nojekyll
