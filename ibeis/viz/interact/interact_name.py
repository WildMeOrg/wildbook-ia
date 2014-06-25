from __future__ import absolute_import, division, print_function
import utool
from itertools import izip
from plottool import interact_helpers as ih
from plottool import draw_func2 as df2
from ibeis import viz
from ibeis.viz import viz_helpers as vh
from ibeis.dev import ibsfuncs
from functools import partial
from plottool.abstract_interaction import AbstractInteraction
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_name]', DEBUG=False)

from ibeis.viz import viz_chip


#==========================
# Name Interaction
#==========================


def ishow_name(ibs, nid, sel_rids=[], select_rid_callback=None, fnum=5, **kwargs):
    fig = ih.begin_interaction('name', fnum)

    def _on_name_click(event):
        print_('[inter] clicked name')
        ax = event.inaxes
        if ih.clicked_inside_axis(event):
            viztype = vh.get_ibsdat(ax, 'viztype')
            print_(' viztype=%r' % viztype)
            if viztype == 'chip':
                rid = vh.get_ibsdat(ax, 'rid')
                print('... rid=%r' % rid)
                viz.show_name(ibs, nid, fnum=fnum, sel_rids=[rid])
                if select_rid_callback is not None:
                    select_rid_callback(rid)
        viz.draw()

    viz.show_name(ibs, nid, fnum=fnum, sel_rids=sel_rids)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_name_click)
    pass


class MatchVerificationInteraction(AbstractInteraction):
    def __init__(self, ibs, rid1, rid2, update_callback=None, backend_callback=None, **kwargs):
        print('[matchver] __init__')
        super(MatchVerificationInteraction, self).__init__(**kwargs)
        self.ibs = ibs
        self.rid1 = rid1
        self.rid2 = rid2
        if update_callback is None:
            update_callback = lambda: None
        if backend_callback is None:
            print("backend_callback is None")
            backend_callback = lambda: None
        self.update_callback = update_callback  # if something like qt needs a manual refresh on change
        self.backend_callback = backend_callback
        self.infer_data()
        self.show_page(bring_to_front=True)

    def infer_data(self):
        """ Initialize data related to the input rids """
        ibs = self.ibs
        # The two matching rids
        (rid1, rid2) = (self.rid1, self.rid2)
        self.match_text = ibs.get_match_text(rid1, rid2)
        # The names of the matching rois
        self.nid1, self.nid2 = ibs.get_roi_nids((rid1, rid2))
        self.name1, self.name2 = ibs.get_names((self.nid1, self.nid2))
        # The other rois that belong to these two names
        groundtruth_list = ibs.get_roi_groundtruth((rid1, rid2))
        self.gt_list = [gt + [rid] for gt, rid in izip(groundtruth_list, (rid1, rid2))]
        # A flat list of all the rids we are looking at
        self.rid_list = utool.unique_ordered(utool.flatten(self.gt_list))
        # Original sets of groundtruth we are working with
        self.gt1, self.gt2 = self.gt_list
        # Grid that will fit all the names we need to display
        self.nCols = max(map(len, self.gt_list))
        self.nRows = len(self.gt_list)
        if self.nid1 == self.nid2:
            self.nRows = 1
            self.gt_list = self.gt_list[0:1]  # remove redundant rids

    def prepare_page(self):
        figkw = {'fnum': self.fnum,
                 'doclf': True,
                 'docla': True, }
        self.fig = df2.figure(**figkw)
        ih.disconnect_callback(self.fig, 'button_press_event')
        #ih.connect_callback(self.fig, 'button_press_event', _on_name_click)

    def show_page(self, bring_to_front=False):
        """ Plots all subaxes on a page """
        print('[matchver] show_page()')
        self.prepare_page()
        # Variables we will work with to paint a pretty picture
        ibs = self.ibs
        nRows = self.nRows
        nCols = self.nCols

        # Distinct color for every unique name
        unique_nids = utool.unique_ordered(ibs.get_roi_nids(self.rid_list))
        import ibeis
        unique_colors = df2.distinct_colors(len(unique_nids) + 2)
        self.nid2_color = dict(izip(unique_nids, unique_colors))

        for count, groundtruth in enumerate(self.gt_list):
            offset = count * nCols + 1
            for px, rid in enumerate(groundtruth):
                nid = ibs.get_roi_nids(rid)
                color = self.nid2_color[nid]

                if ibs.is_rid_unknown(rid):
                    color = ibeis.constants.UNKNOWN_PURPLE_RGBA01
                elif nid == self.nid1:
                    color = ibeis.constants.NAME_RED_RGBA01
                elif nid == self.nid2:
                    color = ibeis.constants.NAME_BLUE_RGBA01
                else:
                    color = ibeis.constants.NEW_YELLOW_RGBA01
                self.plot_chip(rid, nRows, nCols, px + offset, color=color)

        self.show_hud()
        df2.adjust_subplots_safe(top=0.85, hspace=0.03)
        self.draw()
        self.show()
        if bring_to_front:
            self.bring_to_front()
        #self.update()

    def plot_chip(self, rid, nRows, nCols, px, **kwargs):
        """ Plots an individual chip in a subaxis """
        ibs = self.ibs
        nid = ibs.get_roi_nids(rid)
        viz_chip_kw = {
            'fnum': self.fnum,
            'pnum': (nRows, nCols, px),
            'nokpts': True,
            'show_name': True,
            'show_gname': False,
            'show_ridstr': True,
            'notitle': True,
        }
        viz_chip.show_chip(ibs, rid, **viz_chip_kw)
        ax = df2.gca()
        df2.draw_border(ax, color=kwargs.get('color'), lw=4)

        if kwargs.get('make_buttons', True):
            divider = df2.ensure_divider(ax)
            butkw = {
                'divider': divider,
                'size': '13%'
            }
        roi_unknown = ibs.is_nid_unknown([nid])[0]
        if not roi_unknown:
            callback = partial(self.unname_roi, rid)
            self.append_button('unname', callback=callback, **butkw)
        if nid != self.nid1 and not ibs.is_nid_unknown([self.nid1])[0]:
            callback = partial(self.rename_roi_nid1, rid)
            text = 'change name to: ' + ibs.get_names(self.nid1)
            self.append_button(text, callback=callback, **butkw)
        if nid != self.nid2 and not ibs.is_nid_unknown([self.nid2])[0]:
            callback = partial(self.rename_roi_nid2, rid)
            text = 'change name to: ' + ibs.get_names(self.nid2)
            self.append_button(text, callback=callback, **butkw)

    def unname_roi(self, rid, event=None):
        print('unname')
        self.ibs.set_roi_nids([rid], [self.ibs.UNKNOWN_NID])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def rename_roi_nid1(self, rid, event=None):
        print('rename nid1')
        self.ibs.set_roi_nids([rid], [self.nid1])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def rename_roi_nid2(self, rid, event=None):
        print('rename nid2')
        self.ibs.set_roi_nids([rid], [self.nid2])
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def show_hud(self):
        """ Creates heads up display """
        # Button positioners
        hl_slot, hr_slot = df2.make_bbox_positioners(y=.02, w=.16,
                                                     h=3 * utool.PHI_B ** 4,
                                                     xpad=.02, startx=0, stopx=1)

        ibs = self.ibs
        name1, name2 = self.name1, self.name2

        nid_list = ibs.get_roi_nids(self.rid_list, distinguish_unknowns=False)

        def next_rect(accum=[-1]):
            accum[0] += 1
            return hr_slot(accum[0])

        is_unknown = ibs.is_nid_unknown(nid_list)

        if not all(is_unknown):
            self.append_button('unname all', callback=self.unname_all,
                               rect=next_rect())
        if all(is_unknown):
            self.append_button('merge all\n into NEW NAME',
                               callback=self.merge_all_into_next_name, rect=next_rect())
        if not name1.startswith('____'):
            self.append_button('join all\n into name2=%s' % name1, callback=self.merge_all_into_nid1, rect=next_rect())
        if name1 != name2 and not name2.startswith('____') and \
           not all([False]):
            self.append_button('join all\n into name2=%s' % name2,
                               callback=self.merge_all_into_nid2, rect=next_rect())
        """ Heads up display """
        self.vsstr = ibsfuncs.vsstr(self.rid1, self.rid2)
        figtitle_fmt = '''
        Match Review Interface
        {match_text}
        {vsstr}
        '''
        figtitle = figtitle_fmt.format(**self.__dict__)  # sexy: using obj dict as fmtkw
        df2.set_figtitle(figtitle)

    def unname_all(self, event=None):
        print('unname')
        self.ibs.set_roi_nids(self.rid_list, [self.ibs.UNKNOWN_NID] * len(self.rid_list))
        self.show_page()

    def merge_all_into_nid1(self, event=None):
        """ All the rois are given nid1 """
        self.ibs.set_roi_nids(self.rid_list , [self.nid1] * len(self.rid_list))
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def merge_all_into_nid2(self, event=None):
        """ All the rois are given nid2 """
        self.ibs.set_roi_nids(self.rid_list , [self.nid2] * len(self.rid_list))
        self.update_callback()
        self.backend_callback()
        self.show_page()

    def merge_all_into_next_name(self, event=None):
        """ All the rois are given nid2 """
        #self.next_name = next_name = ibsfuncs.make_new_name(self.ibs)
        #self.ibs.set_roi_names(self.rid_list , [next_name] * len(self.rid_list))
        ibs.set_roi_names_to_next_name(self.rid_list)
        self.update_callback()
        self.backend_callback()
        self.show_page()
