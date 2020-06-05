#!/bin/bash
# rm -rf _doc
# rm -rf _page
#autogen_sphinx_docs.py
#python -m utool.util_setup --exec-autogen_sphinx_apidoc --dry
#sphinx-apidoc --force --full --maxdepth="8" --doc-author="Jon Crall" --doc-version="1.4.4" --doc-release="1.4.4" --output-dir="_doc" --private wbia
# manual edits
#manual edits of conf.py
#make html
#python -m utool.util_setup --exec-autogen_sphinx_apidoc 
python -m utool.util_setup --exec-autogen_sphinx_apidoc  --nomake
make -C _doc html
mkdir _page
rm -rf _page/*
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

# Force a subtree push
# References: http://stevenclontz.com/blog/2014/05/08/git-subtree-push-for-deployment/
# Command does not work on windows
#git push origin `git subtree split --prefix _page next`:gh-pages --force
#git push origin `git subtree split --prefix _page {BRANCH}`:gh-pages --force


#git push origin --delete gh-pages
