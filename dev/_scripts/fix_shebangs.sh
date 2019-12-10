#!/bin/bash

# recrusive sed command for ibeis modificiations
export IBEIS_DIR='~/code/ibeis'
# This should work generally to get the ibeis dir
export IBEIS_DIR=$(python2.7 -c "import os, ibeis; print(os.path.dirname(os.path.dirname(os.path.realpath(ibeis.__file__))))")


find $IBEIS_DIR -name "*.py" -type f -exec sed -n 's/#!\/usr\/bin\/env python *$/#!\/usr\/bin\/env python2.7/g' {} +



echo -e "Bla\nBla\nImportant1: One \nBla\nImportant2: Two\nBla\nBla" | \
   sed -n 's/^Important1: *\([^ ]*\) */\1/p'


export RECL="find $IBEIS_DIR -name "*.py" -type f -exec"
export RECR=" {} +"

export PATTERN='#!\/usr\/bin\/env python *$'
export REPL='#!/usr/bin/env python2.7'

# I like my sed better (sedr = sed recrusive)
cd $IBEIS_DIR
rob sedr "$PATTERN" "$REPL" True
cd ~/code/hesaff
rob sedr "$PATTERN" "$REPL" True


rob sp "$PATTERN" "$REPL" False
