'''
Need a list of aids that have had the localizer and 
labeler run on them.
'''

''' *****************************************
*
*  CENSUS ANNOATION (CA) LABELING
*
*  Determine for the current list of annotations which ones are 
*  census annotations. In this case, "census" is equivalent to
*  "seeing a clear shot of the right hip and shoulder".  More generally,
*
'''

aids = [
    aid
    for aid, sp, vpt in zip(aids, species, viewpoints)
    if sp == 'zebra_grevys' and vpt in ['right', 'backright', 'frontright']
]

# Configure and run the CA classify. Again, this is triggered
# with the get_property function call. The first call to get_property
# triggers the actual call (if it hasn't already been run) and returns
# the classification predictions ("positive" or "negative"). The second
# call returns immediately with the confidences in the predictions.
config = {
    'classifier_algo': 'densenet', 
    'classifier_weight_filepath': 'canonical_zebra_grevys_v4'
}
predictions = ibs.depc_annot.get_property(
    'classifier', 
    aids, 
    'class', 
    config=config
)
confidences = ibs.depc_annot.get_property(
    'classifier', 
    aids, 
    'score', 
    config=config
)

# The confidence is the confidence in the decision, and therefore always
# above 0.5. In order set a lower threshold for the CA decision we need
# to make the confidence a "confidence in positive" measure.  Therefore
# for negative predictions, we replace the confidence by subtracting it
# from 1
confidences = [
    confidence if prediction == 'positive' else 1.0 - confidence
    for prediction, confidence in zip(predictions, confidences)
]

#  Keep only the annotations as CA if the confidence is above
#  a threshold. The value here was tuned separately.
flags = [confidence >= 0.31 for confidence in confidences]
ca_aids = ut.compress(aids, flags)
confs = ut.compress(confidences, flags)


'''' ***************************************************
*
*  SAVING THE CA DESIGNATION:  The term 'canonical' was
*  used before 'census' and so we use this to record the
*  CA status. The default is False.
'''
ibs.set_annot_canonical(ca_aids, [True] * len(ca_aids))


''' ***************************************************
* 
*  OTHER "DETECTION" OPERATIONS ON ANNOTATIONS
*
*  Many other operation may be applied to refine and filter
*  annotations.
*
*  1. Bounding box regressions to form census annotation regions
*  2. Removing annotations with extreme bounding box aspect ratios
*  3. Removing annotations that are too small
*  4. Removing annotations that have too low contrast.
*
*  Most of these would be unnecessary with larger volumes
*  of training data, but we don't have that luxury.
*
*  Code will be in a separate file.
'''





