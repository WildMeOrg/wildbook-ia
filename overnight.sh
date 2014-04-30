#export TESTDB="GZ"
export TESTDB="testdb1"
python dev.py -t gv_scores --allgt --db $TESTDB
python dev.py -t gv_test --allgt --db $TESTDB
