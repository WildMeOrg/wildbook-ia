python dev.py -t custom:scale_max=30                             --db PZ_Master0 --allgt
python dev.py -t custom:affine_invariance=False,scale_max=30     --db PZ_Master0 --allgt
python dev.py -t custom:affine_invariance=False                  --db PZ_Master0 --allgt
python dev.py -t custom                                          --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True              --db PZ_Master0 --allgt
python dev.py -t custom:augment_queryside_hack=True,scale_max=30 --db PZ_Master0 --allgt


python dev.py --db PZ_Master0 --allgt -t custom:scale_max=30 custom:affine_invariance=False,scale_max=30  custom:affine_invariance=False custom custom:augment_queryside_hack=True  custom:augment_queryside_hack=True,scale_max=30
