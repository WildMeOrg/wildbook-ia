# TODO

* Migrate vsmany / vsone algorithm to dependency cache
* Incorporate image level algorithms into the dependency cache.
* Encounter/occurrence annotation configuration
* vsmany should accept multiple annotations as input
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


* Turtle viewpoint
 - ignore initial theta orientatino normalization
 - Euler angles ordered by roll pitch yaw.
 - Can add some error pertibation whenever pitch (middle gimble) approaches 90
   degrees to account for gimble lock.

* Look into GIF image failures (and add tests)

* Fix database to store invalid timestamps and GPS coords as NaN instead of -1
* Fix database to store the UNKNOWN name as NULL instead of 0.
* Fix database to use FOREIGN KEYS
* Fix database to CREATE INDEX on the appropriate tables at build time instead of at call time.


* Change annotation verts / bbox / theta such:
(1) verts are primarily used instead of bboxes
(2) that vert positions are not modified if theta is modified
(3) theta indicates the way to rotate the chip after the image bounded by the
    verts is extracted.
(4) the bbox property gets a convex bbox around the verts. (this is shown in the thumbnails,
but the verts are shown when editing)

* Add notion of annotation parts

* Better editing of annotation tags
* Add tags for images as well.


* Replace print statements in classes with dedicated loggers
This will allow debug info to be written to file and not stdout and separate 
the streams of different print statements.


* Delay the execution of costly imports such as theano and matplotlib to allow for 
a faster startup time of the program. Only incur the cost of initialization if these 
modules are used. The same goes for dtool databases and database backups.
