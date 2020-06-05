# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.plottool import interact_annotations
import wbia.plottool as pt  # NOQA
import utool as ut

print, rrr, profile = ut.inject2(__name__)


# DESTROY_OLD_WINDOW = True
DESTROY_OLD_WINDOW = False


def ishow_image2(ibs, gid, fnum=None, dodraw=True):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        gid (int):
        dodraw (bool):

    CommandLine:
        python -m wbia.viz.interact.interact_annotations2 --test-ishow_image2 --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.viz.interact.interact_annotations2 import *  # NOQA
        >>> import wbia
        >>> # build test data
        >>> ibs = wbia.opendb('testdb1')
        >>> gid = 2
        >>> dodraw = True
        >>> # execute function
        >>> self = ishow_image2(ibs, gid, dodraw)
        >>> # verify results
        >>> result = str(self)
        >>> print(result)
        >>> pt.show_if_requested()
    """
    self = ANNOTATION_Interaction2(ibs, gid, fnum=fnum, dodraw=dodraw)
    return self


class ANNOTATION_Interaction2(object):
    def __init__(
        self,
        ibs,
        gid,
        next_callback=None,
        prev_callback=None,
        rows_updated_callback=None,
        reset_window=True,
        dodraw=True,
        fnum=None,
    ):
        """
        TODO: rename to interact image annotations?
        """
        self.ibs = ibs
        self.gid = gid
        self.rows_updated_callback = rows_updated_callback
        img = ibs.get_images(self.gid)
        self.aid_list = ibs.get_image_aids(self.gid)
        bbox_list = ibs.get_annot_bboxes(self.aid_list)
        # verts_list    = ibs.get_annot_verts(self.aid_list)  # TODO
        theta_list = ibs.get_annot_thetas(self.aid_list)
        species_list = ibs.get_annot_species_texts(self.aid_list)
        # valid_species = ibs.get_all_species_texts()
        valid_species = [tup[1] for tup in ibs.get_working_species()]
        metadata_list = [ibs.get_annot_lazy_dict(aid) for aid in self.aid_list]
        for metadata in metadata_list:
            # eager eval on name
            metadata['name']
        if True:
            interact_annotations.rrr()
        self.interact_ANNOTATIONS = interact_annotations.AnnotationInteraction(
            img,
            bbox_list=bbox_list,
            theta_list=theta_list,
            species_list=species_list,
            metadata_list=metadata_list,
            commit_callback=self.commit_callback,
            # TODO: get default species in a better way
            default_species=self.ibs.cfg.detect_cfg.species_text,
            next_callback=next_callback,
            prev_callback=prev_callback,
            fnum=fnum,
            valid_species=valid_species,
            # figure_to_use=None if reset_window else self.interact_ANNOTATIONS.fig,
        )
        if dodraw:
            self.interact_ANNOTATIONS.start()
            # pt.update()

    def commit_callback(
        self,
        unchanged_indices,
        deleted_indices,
        changed_indices,
        changed_annottups,
        new_annottups,
    ):
        """
        TODO: Rename to commit_callback
        Callback from interact_annotations to ibs for when data is modified
        """
        print('[interact_annot2] enter commit_callback')
        print(
            '[interact_annot2] nUnchanged=%d, nDelete=%d, nChanged=%d, nNew=%d'
            % (
                len(unchanged_indices),
                len(deleted_indices),
                len(changed_indices),
                len(new_annottups),
            )
        )
        rows_updated = False
        # Delete annotations
        if len(deleted_indices) > 0:
            rows_updated = True
            deleted_aids = [self.aid_list[del_index] for del_index in deleted_indices]
            print('[interact_annot2] deleted_indexes: %r' % (deleted_indices,))
            print('[interact_annot2] deleted_aids: %r' % (deleted_aids,))
            self.ibs.delete_annots(deleted_aids)
        # Set/Change annotations
        if len(changed_annottups) > 0:
            changed_aid = [self.aid_list[index] for index in changed_indices]
            bbox_list1 = [bbox for (bbox, t, s) in changed_annottups]
            theta_list1 = [t for (bbox, t, s) in changed_annottups]
            species_list1 = [s for (bbox, t, s) in changed_annottups]
            print('[interact_annot2] changed_indexes: %r' % (changed_indices,))
            print('[interact_annot2] changed_aid: %r' % (changed_aid,))
            self.ibs.set_annot_species(changed_aid, species_list1)
            self.ibs.set_annot_thetas(changed_aid, theta_list1, delete_thumbs=False)
            self.ibs.set_annot_bboxes(changed_aid, bbox_list1, delete_thumbs=True)
        # Add annotations
        if len(new_annottups) > 0:
            # New list returns a list of tuples [(x, y, w, h, theta, species) ...]
            rows_updated = True
            bbox_list2 = [bbox for (bbox, t, s) in new_annottups]
            theta_list2 = [t for (bbox, t, s) in new_annottups]
            species_list2 = [s for (bbox, t, s) in new_annottups]
            gid_list = [self.gid] * len(new_annottups)
            new_aids = self.ibs.add_annots(
                gid_list,
                bbox_list=bbox_list2,
                theta_list=theta_list2,
                species_list=species_list2,
            )
            print('[interact_annot2] new_indexes: %r' % (new_annottups,))
            print('[interact_annot2] new_aids: %r' % (new_aids,))

        print('[interact_annot2] about to exit callback')
        if rows_updated and self.rows_updated_callback is not None:
            self.rows_updated_callback()
        print('[interact_annot2] exit callback')

    def update_image_and_callbacks(self, gid, nextcb, prevcb, do_save=True):
        if do_save:
            # save the current changes when pressing next or previous
            self.interact_ANNOTATIONS.save_and_exit(None, do_close=False)
        if DESTROY_OLD_WINDOW:
            ANNOTATION_Interaction2.__init__(
                self,
                self.ibs,
                gid,
                next_callback=nextcb,
                prev_callback=prevcb,
                rows_updated_callback=self.rows_updated_callback,
                reset_window=False,
            )
        else:
            if True:
                self.interact_ANNOTATIONS.rrr()
            ibs = self.ibs
            self.gid = gid
            img = ibs.get_images(self.gid)
            self.aid_list = ibs.get_image_aids(self.gid)
            bbox_list = ibs.get_annot_bboxes(self.aid_list)
            theta_list = ibs.get_annot_thetas(self.aid_list)
            species_list = ibs.get_annot_species_texts(self.aid_list)
            metadata_list = [ibs.get_annot_lazy_dict(aid) for aid in self.aid_list]
            self.interact_ANNOTATIONS.update_image_and_callbacks(
                img,
                bbox_list=bbox_list,
                theta_list=theta_list,
                species_list=species_list,
                metadata_list=metadata_list,
                next_callback=nextcb,
                prev_callback=prevcb,
            )


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.interact.interact_annotations2
        python -m wbia.viz.interact.interact_annotations2 --allexamples
        python -m wbia.viz.interact.interact_annotations2 --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
