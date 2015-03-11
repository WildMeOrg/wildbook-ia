# We are different than lnbnn still, but that is because vsonerr strips out
# dupvote and fgweight Turn those off in custom. 

python dev.py --allgt -t custom custom:rrvsone_on=True,prior_coeff=1,unconstrained_coeff=0.0,fs_lnbnn_min=0,fs_lnbnn_max=1,nAnnotPerName=200,nNameShortlistVsone=200 --print-confusion-stats --noqcache
# Doing this gets close
python dev.py --allgt -t custom:fg_weight=0.0,dupvote_weight=0.0 custom:rrvsone_on=True,prior_coeff=1,unconstrained_coeff=0.0,fs_lnbnn_min=0,fs_lnbnn_max=1,nAnnotPerName=200,nNameShortlistVsone=200 --print-confusion-stats 
# Remmbering to turn them on in both instance gets even closer BUT STILL NOT THERE!
python dev.py --allgt -t custom:fg_weight=0.0,dupvote_weight=0.0 custom:rrvsone_on=True,prior_coeff=1,unconstrained_coeff=0.0,fs_lnbnn_min=0,fs_lnbnn_max=1,nAnnotPerName=200,nNameShortlistVsone=200,fg_weight=0.0,dupvote_weight=0.0 --print-confusion-stats  --index 40:60

# Single bad case 
python dev.py --allgt -t custom:fg_weight=0.0,dupvote_weight=0.0 custom:rrvsone_on=True,prior_coeff=1,unconstrained_coeff=0.0,fs_lnbnn_min=0,fs_lnbnn_max=1,nAnnotPerName=200,nNameShortlistVsone=200,fg_weight=0.0,dupvote_weight=0.0 --print-confusion-stats  --index 53:54  --print-gtscore --noqcache --db PZ_Master0

# Ok, got there
# these are the same configurations for vsone reranking and vsmany
# notice that name scoring and feature scoring are turned off. 
# also everything is reranked
python dev.py --allgt -t custom:fg_weight=0.0,dupvote_weight=0.0 custom:rrvsone_on=True,prior_coeff=1,unconstrained_coeff=0.0,fs_lnbnn_min=0,fs_lnbnn_max=1,nAnnotPerName=200,nNameShortlistVsone=200,fg_weight=0.0,dupvote_weight=0.0 --print-confusion-stats --print-gtscore --noqcache
