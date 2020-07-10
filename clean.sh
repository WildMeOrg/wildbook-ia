#!/bin/bash

rm -rf __pycache__
rm -rf _skbuild
rm -rf dist
rm -rf build
rm -rf wbia.egg-info

rm -rf ibeis
rm -rf ibeis.egg-info

rm -rf _docs/_build/
rm -rf _page/

rm -rf mb_work
rm -rf wheelhouse

rm -rf timings.txt

CLEAN_PYTHON='find . -iname __pycache__ -delete && find . -iname *.pyc -delete && find . -iname *.pyo -delete'
bash -c "$CLEAN_PYTHON"
