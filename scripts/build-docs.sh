#!/usr/bin/env bash

set -ex

# Use the bash script location (at $0) to determine the project root.
PROJECT_ROOT=$(python3 -c "from pathlib import Path; print(Path('$0').parent.parent.resolve())")

cd $PROJECT_ROOT
pip install sphinx
cd docs
make html
