# TODO

* Migrate vsmany / vsone algorithm to dependency cache
* Incorporate image level algorithms into the dependency cache.
* Encounter/occurrence annotation configuration
* vsmany should accpet multiple annotations as input
* Cross-validataion in annot configurations. 



* Migrate all hotspotter features to dependency cache
   - Replace manual chip functions with calls to the depcache

   REQUIRES:
   - depcache needs better deleter support

   Need to integrate depcache with 
   * manual_annot_funcs.delete_annots
   * preproc_annot.on_delete
   * preproc_chip.on_delete
   * ibs.delete_annot_chips(aid_list)
   * ibs.delete_chips
   * ibs.delete_image_thumbs
   * ibs.delete_annot_chip_thumbs(aid_list)
   * ibs.delete_features(fid_list, config2_=config2_)

   ALSO:
   * need to specify properties of annots. 
   When annots are changed depcache needs to be updated. 
