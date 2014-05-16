export VTOOL_DIR=$(python -c "import os, vtool; print(os.path.dirname(vtool.__file__))")

python $VTOOL_DIR/tests/test_draw_keypoint.py --noshow 

python $VTOOL_DIR/tests/test_spatial_verification.py --noshow 

python $VTOOL_DIR/tests/test_exhaustive_ori_extract.py --noshow 

python $VTOOL_DIR/tests/test_vtool.py 
