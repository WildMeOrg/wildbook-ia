#!/bin/bash
#./dev.py -t smk2 --db Oxford --allgt --index 0:55 --exclude-query
./dev.py --db Oxford -t smk --qindex 0:55 --dindex 55:1000000 --all
./dev.py --db Oxford -t oxford --qindex 0:55 --dindex 55:1000000 --all --print-all
