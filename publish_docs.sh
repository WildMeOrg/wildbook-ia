#!/bin/bash
autogen_sphinx_docs.py
cp -r _doc/_build/html/* _page
#git add _page/.nojekyll
git add _page/*
#git add _page
git commit -m "updated docs"
git subtree push --prefix _page origin gh-pages
