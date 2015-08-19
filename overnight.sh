#!/bin/sh
python -m ibeis.scripts.gen_cand_expts --exec-generate_all


# Make sure test are running try them again if they fail
rm marker*.txt
./overnight_experiments.sh

echo 'marker1' >> marker1.txt
./experiment_preload.sh

echo 'marker2' >> marker2.txt
./overnight_experiments.sh

python -m ibeis.scripts.gen_cand_expts --exec-generate_all --full

echo 'marker3' >> marker3.txt
./overnight_experiments.sh

echo 'marker4' >> marker4.txt
./overnight_experiments.sh

# Do lengthier experiments if time permits

#python -m ibeis.scripts.gen_cand_expts --exec-baseline_experiments --full
#python -m ibeis.scripts.gen_cand_expts --exec-invariance_experiments --full

#echo 'marker3' >> marker3.txt
#./experiment_baseline.sh

#echo 'marker4' >> marker4.txt
#./experiment_invar.sh

echo 'markerFINISH' >> markerFINISH.txt
