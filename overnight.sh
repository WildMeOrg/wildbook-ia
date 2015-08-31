
python -m ibeis.dev -e draw_rank_surface -t candidacy_k -a varysize_gz --db GZ_ALL --hargv=expt 

python -m ibeis.dev -e draw_rank_surface -t candidacy_k -a varysize_girm --db NNP_MasterGIRM_core --hargv=expt 


python -m ibeis.dev -e draw_rank_cdf -t candidacy_namescore -a varypername_pzm --db GZ_ALL --hargv=expt


python -m ibeis.dev -e draw_rank_cdf -t candidacy_namescore -a varypername_pzm --db NNP_MasterGIRM_core --hargv=expt

python -m ibeis.dev -e draw_rank_cdf -t candidacy_namescore -a varypername_pzm --db PZ_Master1 --hargv=expt

python -m ibeis.dev -e draw_rank_surface -t candidacy_k -a varysize_pzm --db PZ_Master1 --hargv=expt
