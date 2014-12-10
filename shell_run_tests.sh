#!/bin/bash
# Runs all tests
# Win32 path hacks
export CWD=$(pwd)
export PYMAJOR="$(python -c "import sys; print(sys.version_info[0])")"

# <CORRECT_PYTHON>
# GET CORRECT PYTHON ON ALL PLATFORMS
export SYSNAME="$(expr substr $(uname -s) 1 10)"
if [ "$SYSNAME" = "MINGW32_NT" ]; then
    export PYEXE=python
else
    if [ "$PYMAJOR" = "3" ]; then
        # virtual env?
        export PYEXE=python
    else
        export PYEXE=python2.7
    fi
fi
# </CORRECT_PYTHON>

PRINT_DELIMETER()
{
    printf "\n#\n#\n#>>>>>>>>>>> next_test\n\n"
}

export TEST_ARGV="--quiet --noshow $@"

export VTOOL_DIR=$($PYEXE -c "import os, vtool;print(str(os.path.dirname(os.path.dirname(vtool.__file__))))")
export HESAFF_DIR=$($PYEXE -c "import os, pyhesaff;print(str(os.path.dirname(os.path.dirname(pyhesaff.__file__))))")

# Default tests to run
set_test_flags()
{
    export DEFAULT=$1
    export VTOOL_TEST=$DEFAULT
    export GUI_TEST=$DEFAULT
    export IBEIS_TEST=$DEFAULT
    export SQL_TEST=$DEFAULT
    export VIEW_TEST=$DEFAULT
    export MISC_TEST=$DEFAULT
    export OTHER_TEST=$DEFAULT
    export HESAFF_TEST=$DEFAULT
    export DOC_TEST=$DEFAULT
}
set_test_flags OFF
export DOC_TEST=ON

# Parse for bash commandline args
for i in "$@"
do
case $i in --testall)
    set_test_flags ON
    ;;
esac
case $i in --notestvtool)
    export VTOOL_TEST=OFF
    ;;
esac
case $i in --testvtool)
    export VTOOL_TEST=ON
    ;;
esac
case $i in --notestgui)
    export GUI_TEST=OFF
    ;;
esac
case $i in --testgui)
    export GUI_TEST=ON
    ;;
esac
case $i in --notestibeis)
    export IBEIS_TEST=OFF
    ;;
esac
case $i in --testibeis)
    export IBEIS_TEST=ON
    ;;
esac
case $i in --notestsql)
    export SQL_TEST=OFF
    ;;
esac
case $i in --testsql)
    export SQL_TEST=ON
    ;;
esac
case $i in --notestview)
    export VIEW_TEST=OFF
    ;;
esac
case $i in --testview)
    export VIEW_TEST=ON
    ;;
esac
case $i in --notestmisc)
    export MISC_TEST=OFF
    ;;
esac
case $i in --testmisc)
    export MISC_TEST=ON
    ;;
esac
case $i in --notestother)
    export OTHER_TEST=OFF
    ;;
esac
case $i in --testother)
    export OTHER_TEST=ON
    ;;
esac
case $i in --notesthesaff)
    export HESAFF_TEST=OFF
    ;;
esac
case $i in --testhesaff)
    export HESAFF_TEST=ON
    ;;
esac
case $i in --notestdoc)
    export DOC_TEST=OFF
    ;;
esac
case $i in --testdoc)
    export DOC_TEST=ON
    ;;
esac
done

BEGIN_TESTS()
{
cat <<EOF
  ______ _     _ __   _      _______ _______ _______ _______ _______
 |_____/ |     | | \  |         |    |______ |______    |    |______
 |    \_ |_____| |  \_|         |    |______ ______|    |    ______|
                                                                    

EOF
    echo "BEGIN: TEST_ARGV=$TEST_ARGV"
    PRINT_DELIMETER
    num_passed=0
    num_ran=0
    export FAILED_TESTS=''
}

RUN_TEST()
{
    echo "RUN_TEST: $@"
    export TEST="$PYEXE $@ $TEST_ARGV"
    $TEST
    export RETURN_CODE=$?
    echo "RETURN_CODE=$RETURN_CODE"
    PRINT_DELIMETER
    num_ran=$(($num_ran + 1))
    if [ "$RETURN_CODE" == "0" ] ; then
        num_passed=$(($num_passed + 1))
    fi
    if [ "$RETURN_CODE" != "0" ] ; then
        export FAILED_TESTS="$FAILED_TESTS\n$TEST"
    fi
}

END_TESTS()
{
    echo "RUN_TESTS: DONE"
    if [ "$FAILED_TESTS" != "" ] ; then
        echo "-----"
        printf "Failed Tests:"
        printf "$FAILED_TESTS\n"
        printf "$FAILED_TESTS\n" >> failed_shelltests.txt
        echo "-----"
    fi
    echo "$num_passed / $num_ran tests passed"
}

#---------------------------------------------
# START TESTS
BEGIN_TESTS

# Quick Tests (always run)
RUN_TEST ibeis/tests/assert_modules.py

#---------------------------------------------
#VTOOL TESTS
if [ "$VTOOL_TEST" = "ON" ] ; then
cat <<EOF
    _  _ ___ ____ ____ _       ___ ____ ____ ___ ____ 
    |  |  |  |  | |  | |        |  |___ [__   |  [__  
     \/   |  |__| |__| |___     |  |___ ___]  |  ___]
EOF
    RUN_TEST $VTOOL_DIR/vtool/tests/test_draw_keypoint.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_spatial_verification.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_exhaustive_ori_extract.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_vtool.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_akmeans.py 
    RUN_TEST $VTOOL_DIR/vtool/tests/test_pyflann.py 
fi
#---------------------------------------------
#GUI TESTS
if [ "$GUI_TEST" = "ON" ] ; then
cat <<EOF
    ____ _  _ _    ___ ____ ____ ___ ____ 
    | __ |  | |     |  |___ [__   |  [__  
    |__] |__| |     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_gui_selection.py 
    RUN_TEST ibeis/tests/test_gui_open_database.py 
    RUN_TEST ibeis/tests/test_gui_add_annotation.py 
    RUN_TEST ibeis/tests/test_gui_all.py 
    RUN_TEST ibeis/tests/test_gui_import_images.py 
fi
#---------------------------------------------
#IBEIS TESTS
if [ "$IBEIS_TEST" = "ON" ] ; then
cat <<EOF
    _ ___  ____ _ ____    ___ ____ ____ ___ ____ 
    | |__] |___ | [__      |  |___ [__   |  [__  
    | |__] |___ | ___]     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_ibs_detect.py 
    RUN_TEST ibeis/tests/test_ibs_query_components.py 
    RUN_TEST ibeis/tests/test_ibs.py 
    RUN_TEST ibeis/tests/test_ibs_query.py 
    RUN_TEST ibeis/tests/test_ibs_encounters.py 
    RUN_TEST ibeis/tests/test_ibs_control.py 
    RUN_TEST ibeis/tests/test_ibs_feat_compute.py 
    RUN_TEST ibeis/tests/test_ibs_uuid.py 
    RUN_TEST ibeis/tests/test_ibs_localize_images.py 
    RUN_TEST ibeis/tests/test_ibs_add_images.py 
    RUN_TEST ibeis/tests/test_ibs_chip_compute.py 
    RUN_TEST ibeis/tests/test_ibs_info.py 
    RUN_TEST ibeis/tests/test_ibs_add_name.py 
    RUN_TEST ibeis/tests/test_ibs_getters.py 
    RUN_TEST ibeis/tests/test_ibs_convert_bbox_poly.py 
    RUN_TEST ibeis/tests/test_ibs_detectimg_compute.py 
    RUN_TEST ibeis/tests/test_delete_names.py 
    RUN_TEST ibeis/tests/test_delete_features.py 
    RUN_TEST ibeis/tests/test_delete_enc.py 
    RUN_TEST ibeis/tests/test_delete_image.py 
    RUN_TEST ibeis/tests/test_delete_annotation.py 
    RUN_TEST ibeis/tests/test_delete_image_thumbtups.py 
    RUN_TEST ibeis/tests/test_delete_annotation_chips.py 
    RUN_TEST ibeis/tests/test_delete_annotation_all.py 
    RUN_TEST ibeis/tests/test_delete_chips.py 
fi
#---------------------------------------------
#SQL TESTS
if [ "$SQL_TEST" = "ON" ] ; then
cat <<EOF
    ____ ____ _       ___ ____ ____ ___ ____ 
    [__  |  | |        |  |___ [__   |  [__  
    ___] |_\| |___     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_sql_numpy.py 
    RUN_TEST ibeis/tests/test_sql_modify.py 
    RUN_TEST ibeis/tests/test_sql_revert.py 
    RUN_TEST ibeis/tests/test_sql_names.py 
    RUN_TEST ibeis/tests/test_sql_ids.py 
    RUN_TEST ibeis/tests/test_sql_control.py 
fi
#---------------------------------------------
#VIEW TESTS
if [ "$VIEW_TEST" = "ON" ] ; then
cat <<EOF
    _  _ _ ____ _ _ _    ___ ____ ____ ___ ____ 
    |  | | |___ | | |     |  |___ [__   |  [__  
     \/  | |___ |_|_|     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_view_viz.py 
    RUN_TEST ibeis/tests/test_view_interact.py 
fi
#---------------------------------------------
#MISC TESTS
if [ "$MISC_TEST" = "ON" ] ; then
cat <<EOF
    _  _ _ ____ ____    ___ ____ ____ ___ ____ 
    |\/| | [__  |        |  |___ [__   |  [__  
    |  | | ___] |___     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/test_utool_parallel.py 
    RUN_TEST ibeis/tests/test_pil_hash.py 
fi
#---------------------------------------------
#OTHER TESTS
if [ "$OTHER_TEST" = "ON" ] ; then
cat <<EOF
    ____ ___ _  _ ____ ____    ___ ____ ____ ___ ____ 
    |  |  |  |__| |___ |__/     |  |___ [__   |  [__  
    |__|  |  |  | |___ |  \     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/tests/assert_modules.py 
    RUN_TEST ibeis/tests/reset_testdbs.py 
    RUN_TEST ibeis/tests/test_weakref.py 
    RUN_TEST ibeis/tests/test_qt_tree.py 
    RUN_TEST ibeis/tests/test_new_schema.py 
    RUN_TEST ibeis/tests/test_uuid_consistency.py 
fi
#---------------------------------------------
#HESAFF TESTS
if [ "$HESAFF_TEST" = "ON" ] ; then
cat <<EOF
    _  _ ____ ____ ____ ____ ____    ___ ____ ____ ___ ____ 
    |__| |___ [__  |__| |___ |___     |  |___ [__   |  [__  
    |  | |___ ___] |  | |    |        |  |___ ___]  |  ___]
EOF
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_draw_keypoint.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_pyhesaff_simple_parallel.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_cpp_rotation_invariance.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_adaptive_scale.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_ellipse.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_exhaustive_ori_extract.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_pyhesaff_simple_iterative.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_patch_orientation.py 
    RUN_TEST $HESAFF_DIR/pyhesaff/tests/test_pyhesaff.py 
fi
#---------------------------------------------
#DOC TESTS
if [ "$DOC_TEST" = "ON" ] ; then
cat <<EOF
    ___  ____ ____    ___ ____ ____ ___ ____ 
    |  \ |  | |        |  |___ [__   |  [__  
    |__/ |__| |___     |  |___ ___]  |  ___]
EOF
    RUN_TEST ibeis/ibsfuncs.py --test-check_annot_consistency:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-get_annot_groundfalse_sample:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-get_annot_groundtruth_sample:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-get_annot_is_hard:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-get_upsize_data:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-invertable_flatten2:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-postinject_func:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-print_annotation_table:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-unflat_map:0 --sysexitonfail
    RUN_TEST ibeis/ibsfuncs.py --test-unflatten2:0 --sysexitonfail
    RUN_TEST ibeis/viz/viz_sver.py --test-show_sver:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-DetectionConfig:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-FeatureConfig:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-FeatureWeightConfig:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-FilterConfig:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-QueryConfig:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-SMKConfig:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-SMKConfig:1 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-VocabAssignConfig:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-VocabTrainConfig:0 --sysexitonfail
    RUN_TEST ibeis/model/Config.py --test-parse_config_items:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/name_scoring.py --test-get_one_score_per_name:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/neighbor_index.py --test-NeighborIndex:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/neighbor_index.py --test-get_nn_aids:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/neighbor_index.py --test-request_ibeis_nnindexer:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/neighbor_index.py --test-test_nnindexer:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/hots_query_result.py --test-get_nscoretup:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/nn_weights.py --test-apply_normweight:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/nn_weights.py --test-cos_match_weighter:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/nn_weights.py --test-dupvote_match_weighter:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/nn_weights.py --test-fg_match_weighter:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/nn_weights.py --test-lnbnn_fn:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/nn_weights.py --test-nn_normalized_weight:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/nn_weights.py --test-test_all_normalized_weights:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-_threshold_and_scale_weights:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-build_chipmatches:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-chipmatch_to_resdict:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-filter_neighbors:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-get_pipeline_testdata:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-nearest_neighbors:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-precompute_topx2_dlen_sqrd:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-request_ibeis_query_L0:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-request_ibeis_query_L0:1 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-spatial_verification:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/pipeline.py --test-weight_neighbors:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/score_normalization.py --test-cached_ibeis_score_normalizer:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/score_normalization.py --test-find_score_maxclip:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/score_normalization.py --test-learn_score_normalization:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/query_request.py --test-new_ibeis_query_request:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/query_request.py --test-new_ibeis_query_request:1 --sysexitonfail
    RUN_TEST ibeis/model/hots/query_request.py --test-shallowcopy:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/query_request.py --test-test_cfg_deepcopy:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-MultiNeighborIndex:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-get_nIndexed_list:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-get_nn_aids:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-get_nn_featxs:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-get_nn_fgws:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-get_offsets:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-knn:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-knn2:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-multi_knn:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-num_indexed_annots:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-num_indexed_vecs:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/multi_index.py --test-request_ibeis_mindexer:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/voting_rules2.py --test-score_chipmatch_csum:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/voting_rules2.py --test-score_chipmatch_nsum:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/match_chips4.py --test-execute_query_and_save_L1:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/match_chips4.py --test-submit_query_request:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/automated_matcher.py --test-setup_incremental_test:0 --sysexitonfail
    RUN_TEST ibeis/model/hots/automated_matcher.py --test-setup_incremental_test:1 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_annot.py --test-make_annot_semantic_uuid:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_annot.py --test-make_annot_visual_uuid:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_detectimg.py --test-compute_and_write_detectimg:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_detectimg.py --test-compute_and_write_detectimg_lazy:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_detectimg.py --test-get_image_detectimg_fpath_list:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_featweight.py --test-compute_fgweights:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_featweight.py --test-gen_featweight_worker:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_featweight.py --test-generate_featweight_properties:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_image.py --test-add_images_params_gen:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_feat.py --test-generate_feat_properties:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_chip.py --test-compute_and_write_chips:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_chip.py --test-compute_or_read_chip_images:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_chip.py --test-compute_or_read_chip_images:1 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_chip.py --test-get_annot_cfpath_list:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_chip.py --test-get_chip_fname_fmt:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_chip.py --test-on_delete:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_probchip.py --test-get_annot_probchip_fpath_list:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_probchip.py --test-get_probchip_fname_fmt:0 --sysexitonfail
    RUN_TEST ibeis/model/preproc/preproc_probchip.py --test-group_aids_by_featweight_species:0 --sysexitonfail
    RUN_TEST ibeis/model/detect/randomforest.py --test-_get_detector:0 --sysexitonfail
    RUN_TEST ibeis/model/detect/randomforest.py --test-_get_detector:1 --sysexitonfail
    RUN_TEST ibeis/model/detect/grabmodels.py --test-ensure_models:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-add_annots:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-add_annots:1 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_chip_thumbtup:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_gids:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_groundtruth:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_groundtruth:1 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_groundtruth:2 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_has_groundtruth:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_image_uuids:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_images:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_name_rowids:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_names:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_num_feats:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_num_groundtruth:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_num_groundtruth:1 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_semantic_uuid_info:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_semantic_uuids:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_species:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_species:1 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_thetas:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_viewpoints:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_visual_uuid_info:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_annot_funcs.py --test-get_annot_visual_uuids:0 --sysexitonfail
    RUN_TEST ibeis/control/controller_inject.py --test-find_unregistered_methods:0 --sysexitonfail
    RUN_TEST ibeis/control/IBEISControl.py --test-_query_chips4:0 --sysexitonfail
    RUN_TEST ibeis/control/IBEISControl.py --test-get_scorenorm_cachedir:0 --sysexitonfail
    RUN_TEST ibeis/control/IBEISControl.py --test-get_species_scorenorm_cachedir:0 --sysexitonfail
    RUN_TEST ibeis/control/IBEISControl.py --test-request_IBEISController:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-_get_all_featweight_rowids:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-add_annot_featweights:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-add_feat_featweights:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-delete_annot_featweight:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-delete_featweight:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-get_annot_featweight_rowids:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-get_feat_featweight_rowids:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-get_featweight_config_rowid:0 --sysexitonfail
    RUN_TEST ibeis/control/_autogen_featweight_funcs.py --test-get_featweight_fgweights:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_invalid_nids:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_name_aids:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_name_exemplar_aids:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_name_gids:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_name_num_annotations:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_name_rowids_from_text:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_name_rowids_from_text:1 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_name_texts:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_num_names:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_species_rowids_from_text:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_species_rowids_from_text:1 --sysexitonfail
    RUN_TEST ibeis/control/manual_name_species_funcs.py --test-get_species_texts:0 --sysexitonfail
    RUN_TEST ibeis/control/DB_SCHEMA.py --test-test_dbschema:0 --sysexitonfail
    RUN_TEST ibeis/control/manual_image_funcs.py --test-get_encounter_nids:0 --sysexitonfail
    RUN_TEST ibeis/control/DBCACHE_SCHEMA.py --test-test_dbcache_schema:0 --sysexitonfail
    RUN_TEST ibeis/dev/experiment_harness.py --test-get_qx2_bestrank:0 --sysexitonfail
    RUN_TEST ibeis/dev/experiment_harness.py --test-get_qx2_bestrank:1 --sysexitonfail
    RUN_TEST ibeis/dbio/export_subset.py --test-merge_databases:0 --sysexitonfail
fi

#---------------------------------------------
# END TESTING
END_TESTS