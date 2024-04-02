#!/bin/bash

./clean.sh
python setup.py clean

pip install --no-cache-dir -r requirements.txt
# pip install -r requirements.txt --use-deprecated=legacy-resolver

python setup.py develop

pip install --no-cache-dir -e .
