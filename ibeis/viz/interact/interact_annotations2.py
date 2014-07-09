from __future__ import absolute_import, division, print_function
from plottool import interact_annotations
from plottool import draw_func2 as df2
from itertools import izip


class ANNOTATION_Interaction2(object):
    def __init__(self, ibs, gid, next_callback=None, prev_callback=None, rows_updated_callback=lambda: None):
        self.ibs = ibs
        self.gid = gid
        self.rows_updated_callback = rows_updated_callback
        img = ibs.get_images(self.gid)
        self.aid_list = ibs.get_image_aids(self.gid)
        bbox_list = ibs.get_annot_bboxes(self.aid_list)
        theta_list = ibs.get_annot_thetas(self.aid_list)
        species_list = ibs.get_annot_species(self.aid_list)
        self.interact_ANNOTATIONS = interact_annotations.ANNOTATIONInteraction(
            img,
            bbox_list=bbox_list,
            theta_list=theta_list,
            species_list=species_list,
            callback=self.callback,
            default_species=self.ibs.cfg.detect_cfg.species,
            next_callback=next_callback,
            prev_callback=prev_callback,
            fnum=12
        )


        df2.update()

    def callback(self, deleted_list, changed_list, new_list):
        rows_updated = False
        if len(deleted_list) > 0:
            print(deleted_list)
            rows_updated = True
            deleted = [self.aid_list[del_index] for del_index in deleted_list]
            self.ibs.delete_annots(deleted)
        if len(changed_list) > 0:
            changed_aid = [self.aid_list[changed[0]] for changed, theta in changed_list]
            changed_bbox = [changed[1] for (changed, theta) in changed_list]
            self.ibs.set_annot_bboxes(changed_aid, changed_bbox)
        if len(new_list) > 0:
            rows_updated = True
            bbox_list, theta_list, species_list = izip(*[((x, y, w, h), t, s) for (x, y, w, h, t, s) in new_list])
            #print("species_list in annotation_interaction2: %r" % list(species_list))
            self.ibs.add_annots([self.gid] * len(new_list), bbox_list, theta_list=theta_list, species_list=species_list)
        if rows_updated:
            self.rows_updated_callback()

if __name__ == '__main__':
    import ibeis
    main_locals = ibeis.main(gui=False)
    ibs = main_locals['ibs']
    gid_list = ibs.get_valid_gids()
    gid = gid_list[len(gid_list) - 1]
    annotation = ANNOTATION_Interaction2(ibs, gid)
    exec(df2.present())
