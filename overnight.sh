#export TESTDB="GZ"
#export TESTDB="testdb1"
#python dev.py -t gv_scores --allgt --db $TESTDB
#python dev.py -t gv_test --allgt --db $TESTDB

#export IBSFLAGS='--noqcache --screen'
#export IBSFLAGS='--noqcache --screen --delete-query-cache'
export IBSFLAGS=''


#export IBSFLAGS=''
#python dev.py -t smk5 --allgt --db PZ_Master0 $IBSFLAGS
#python dev.py -t smk5 --allgt --db GZ_ALL $IBSFLAGS
#python dev.py -t smk_64k --allgt --db PZ_Mothers $IBSFLAGS
#python dev.py -t smk_128k --allgt --db PZ_Mothers $IBSFLAGS
#python dev.py -t smk5 --allgt --db PZ_Mothers $IBSFLAGS

#python dev.py -t smk7_overnight --allgt --db GZ_ALL $IBSFLAGS
#python dev.py -t smk7_overnight --allgt --db PZ_Master0 $IBSFLAGS --index 0:200
#python dev.py -t smk7_overnight --allgt --db PZ_Mothers $IBSFLAGS

python dev.py -t featparams_big --db PZ_Mothers --allgt
python dev.py -t featparams_big --db GZ_ALL --allgt
python dev.py -t smk2 --db PZ_Master0 --allgt


export IBSFLAGS=''

#python dev.py -t smk6_overnight --allgt --db GZ_ALL $IBSFLAGS
#python dev.py -t smk6_overnight --allgt --db PZ_Master0 $IBSFLAGS 
#python dev.py -t smk6_overnight --allgt --db PZ_Mothers $IBSFLAGS

#python dev.py -t smk6_overnight --allgt --db GZ_ALL $IBSFLAGS
#python dev.py -t smk6_overnight --allgt --db PZ_Mothers $IBSFLAGS
#python dev.py -t smk6_overnight --allgt --db PZ_Master0 $IBSFLAGS
