'''
Need a list of aids that have had (bounding box) localization
run first.
'''
# aids = ibs.get_valid_aids()   # gets all of them, if you want..


''' *****************************************
*
*  SPECIES LABEL AND VIEWPOINT:
*  
*  Produces bounding boxes and initial species labels.
*  These ids are coarse -- for example plains and Grevy's
*  zebras are combined into the generic "zebra" label
'''

# Filter the localizer annotation results to get only 
# "zebra" class
species = ibs.get_annot_species(aids)
flags = ['zebra' in value for value in species]
aids = ut.compress(aids, flags)

# Configure the labeler
config = {
    'labeler_algo': 'densenet',
    'labeler_weight_filepath': 'zebra_v1',
}

# Use the dependency cache to run the labeler or to
# access the results.
results = ibs.depc_annot.get_property(
    'labeler', 
    aids, 
    None, 
    config=config
)

# Retrieve the species and viewpoint labels from the 
# results.  
species = ut.take_column(results, 1)
viewpoints = ut.take_column(results, 2)

# Record the species and viewpoint labels in the database.
# Until this point they weren't recorded.
ibs.set_annot_species(aids, species)
ibs.set_annot_viewpoints(aids, viewpoints)

# Output some statistics.
print(ut.repr3(ut.dict_hist(species)))
print(ut.repr3(ut.dict_hist(viewpoints)))

''' 
At this point you can stop and review annotations in the 
web interface. Just be sure the imageset id is chosen appropriately.
To do this, select View and then for the Imageset that you 
decide upon, click an option on the right such as Detection or
Annotation
'''
