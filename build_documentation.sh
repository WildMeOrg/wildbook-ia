#!/bin/bash
# rm -rf _doc
# rm -rf _page
# python -m utool.util_setup --exec-autogen_sphinx_apidoc  --nomake

make -C _doc html
mkdir _page
rm -rf _page/*
cp -r _doc/_build/html/* _page
touch _page/.nojekyll
