#!/bin/bash
# rm -rf _doc
# rm -rf _page
autogen_sphinx_docs.py
mkdir _page
cp -r _doc/_build/html/* _page
# show page
#python -c "import utool as ut; ut.startfile('_doc/_build/html/index.html')"
touch _page/.nojekyll
git add _page/.nojekyll
git add _page/*
#git add _page
git commit -m "updated docs"
#git subtree add --prefix _page origin gh-pages 
#git subtree pull --prefix _page origin gh-pages
git subtree push --prefix _page origin gh-pages

#git push origin --delete gh-pages
