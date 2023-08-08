#!/bin/bash

./clean.sh
python setup.py clean

pip install --no-cache-dir -r requirements.txt
# pip install -r requirements.txt --use-deprecated=legacy-resolver

pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install https://github.com/Lasagne/Lasagne/archive/master.zip

python setup.py develop

pip install --no-cache-dir -e .
