#!/bin/bash
set -xe
python dev/reset_dbs.py
xdoctest wbia --style=google all
