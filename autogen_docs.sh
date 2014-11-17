mkdir _doc
#autogen_sphinx_docs.py

sphinx-apidoc --full --maxdepth="8" --doc-author="Jon Crall" --doc-version="1.0.0" --doc-release="1.0.0" --output-dir="_doc" ibeis
cd _doc 
make html
