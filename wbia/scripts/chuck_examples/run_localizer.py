# START BY SETTING GIDs

# gids = ibs.get_valid_gids()
# gids = gids3

''' *****************************************
*
*  LOCALIZE:
*  
*  Produces bounding boxes and initial species labels.
*  These ids are coarse -- for example plains and Grevy's
*  zebras are combined into the generic "zebra" lbel
* 
'''

# Set the localizer configuration
config = {
    'algo'           : 'lightnet',
    'config_filepath': 'ggr2',
    'weight_filepath': 'ggr2',
    'sensitivity'    : 0.4,
    'nms'            : True,
    'nms_thresh'     : 0.4,
}

# Use the dependency caching mechanism to run the localizer.
results = ibs.depc_image.get_property(
    'localizations',
    gids,
    None,
    config=config
)

# Commit the localization results to the database.
# At this point you should be able to see them in
# the web interface as a detection.
aids_list = ibs.commit_localization_results(
    gids,
    results
)

# Convert from a list of lists of aids to just
# a list
aids = ut.flatten(aids_list)
