#export TESTDB="GZ"
#export TESTDB="testdb1"
#python dev.py -t gv_scores --allgt --db $TESTDB
#python dev.py -t gv_test --allgt --db $TESTDB

python dev.py -t smk5 --allgt --db PZ_Master0 --noqcache
python dev.py -t smk5 --allgt --db GZ_ALL --noqcache

