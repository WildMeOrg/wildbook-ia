python dev.py -t custom:scale_max=50                             --db PZ_Master0 --allgt
python dev.py -t custom:affine_invariance=False,scale_max=50     --db PZ_Master0 --allgt
python dev.py -t custom:affine_invariance=False                  --db PZ_Master0 --allgt
python dev.py -t custom                                          --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True              --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,scale_max=50 --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,affine_invariance=False --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,affine_invariance=False,scale_max=50 --db PZ_Master0 --allgt


python dev.py -t custom:scale_max=150                             --db PZ_Master0 --allgt
python dev.py -t custom:affine_invariance=False,scale_max=150     --db PZ_Master0 --allgt
python dev.py -t custom:affine_invariance=False                  --db PZ_Master0 --allgt
python dev.py -t custom                                          --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True              --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,scale_max=150 --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,affine_invariance=False --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,affine_invariance=False,scale_max=150 --db PZ_Master0 --allgt


python dev.py -t custom:rotation_invariance=True              --db PZ_Master0 --allgt --noqcache
python dev.py -t custom:rotation_invariance=True,scale_max=150 --db PZ_Master0 --allgt --noqcache
python dev.py -t custom:rotation_invariance=True,affine_invariance=False --db PZ_Master0 --allgt --noqcache
python dev.py -t custom:rotation_invariance=True,affine_invariance=False,scale_max=150 --db PZ_Master0 --allgt --noqcache


python dev.py -t custom:scale_max=200                            --db PZ_Master0 --allgt
python dev.py -t custom:affine_invariance=False,scale_max=200    --db PZ_Master0 --allgt
python dev.py -t custom:affine_invariance=False                  --db PZ_Master0 --allgt
python dev.py -t custom                                          --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True              --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,scale_max=200 --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,affine_invariance=False --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,affine_invariance=False,scale_max=200 --db PZ_Master0 --allgt


python dev.py --db PZ_Master0 --allgt -t custom:scale_max=50 custom:affine_invariance=False,scale_max=50  custom:affine_invariance=False custom custom:augment_queryside_hack=True  custom:augment_queryside_hack=True,scale_max=50 custom:augment_queryside_hack=True,affine_invariance=False

python dev.py --db PZ_Master0 --allgt -t custom:scale_max=100 custom:affine_invariance=False,scale_max=100  custom:affine_invariance=False custom custom:augment_queryside_hack=True  custom:augment_queryside_hack=True,scale_max=100 custom:augment_queryside_hack=True,affine_invariance=False

python dev.py --db PZ_Master0 --allgt -t custom:scale_max=150 custom:affine_invariance=False,scale_max=150  custom:affine_invariance=False custom custom:augment_queryside_hack=True  custom:augment_queryside_hack=True,scale_max=150 custom:augment_queryside_hack=True,affine_invariance=False

python dev.py --db PZ_Master0 --allgt -t custom:scale_max=200 custom:affine_invariance=False,scale_max=200  custom:affine_invariance=False custom custom:augment_queryside_hack=True  custom:augment_queryside_hack=True,scale_max=200 custom:augment_queryside_hack=True,affine_invariance=False


python dev.py --db PZ_Master0 --allgt -t custom:scale_max=200 custom:affine_invariance=False,scale_max=200  custom:affine_invariance=False custom custom:augment_queryside_hack=True  custom:augment_queryside_hack=True,scale_max=200 custom:augment_queryside_hack=True,affine_invariance=False custom:scale_max=50 custom:scale_max=150 custom:affine_invariance=False,scale_max=150   custom:augment_queryside_hack=True,scale_max=150 custom:augment_queryside_hack=True,scale_max=50 custom:augment_queryside_hack=True,affine_invariance=False,scale_max=200 custom:augment_queryside_hack=True,affine_invariance=False,scale_max=150 custom:augment_queryside_hack=True,affine_invariance=False,scale_max=100 custom:rotation_invariance=True custom:rotation_invariance=True,scale_max=150 custom:rotation_invariance=True,affine_invariance=False custom:rotation_invariance=True,affine_invariance=False,scale_max=150

python dev.py --db PZ_MTEST --allgt -t pyrscale


python dev.py --db Elephants_drop1 --allgt -t custom:affine_invariance=False
