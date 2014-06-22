from __future__ import absolute_import, division, print_function
from plottool import interact_rois
from plottool import draw_func2 as df2
import os


class ROI_Interaction2:
    def __init__(self, ibs, gid):
        self.ibs = ibs
        self.gid = gid
        img = ibs.get_images(self.gid)
        self.rid_list = ibs.get_image_rids(self.gid)
        bbox_list = ibs.get_roi_bboxes(self.rid_list)
        theta_list = ibs.get_roi_thetas(self.rid_list)
        self.interact_ROIS = interact_rois.ROIInteraction(img, callback=self.callback,bbox_list=bbox_list, theta_list=theta_list)
        #exec(df2.present())
        df2.update()

    def callback(self, deleted_list, changed_list, new_list):
        #print('Deleted BBoxes')
        if len(deleted_list) > 0:
            deleted = [self.rid_list[del_index] for del_index in deleted_list]
            #print(deleted)
            self.ibs.delete_rois(deleted)
        #print('Changed BBoxes')
        if len(changed_list) > 0:
            changed_rid = [self.rid_list[changed[0]] for changed in changed_list]
            changed_bbox = [changed[1] for changed in changed_list]
            #print(changed_rid)
            #print(changed_bbox)
            self.ibs.set_roi_bboxes(changed_rid, changed_bbox)
        #print('New BBoxes')
        if len(new_list) > 0:
            #print(new_list)
            self.ibs.add_rois([self.gid] * len(new_list), new_list)
        thumb_path, image_path, bbox_list = self.ibs.get_image_thumbtup(self.gid)
        print('Current BBoxes')
        print(bbox_list)
        if os.path.exists(thumb_path):
            os.remove(thumb_path)  # Force refresh

if __name__ == '__main__':
    import ibeis
    main_locals = ibeis.main(gui=False)
    ibs = main_locals['ibs']
    gid_list = ibs.get_valid_gids()
    gid = gid_list[len(gid_list) - 1]
    self = ROI_Interaction2(ibs, gid)
