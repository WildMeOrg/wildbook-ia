export HESAFF_DIR=$(python -c "import os, pyhesaff; print(os.path.dirname(pyhesaff.__file__))")

python $HESAFF_DIR/tests/test_adaptive_scale.py
python $HESAFF_DIR/tests/test_draw_keypoint.py
python $HESAFF_DIR/tests/test_ellipse.py
python $HESAFF_DIR/tests/test_exhaustive_ori_extract.py
python $HESAFF_DIR/tests/test_patch_orientation.py
python $HESAFF_DIR/tests/test_pyhesaff.py
python $HESAFF_DIR/tests/test_pyhesaff_simple_iterative.py
python $HESAFF_DIR/tests/test_pyhesaff_simple_parallel.py
