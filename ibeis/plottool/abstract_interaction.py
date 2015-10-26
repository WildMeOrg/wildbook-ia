# -*- coding: utf-8 -*-
"""
Known Interactions that use AbstractInteraction:
    pt.MatchInteraction2
    pt.MultiImageInteraction
    ibeis.NameInteraction
"""
from __future__ import absolute_import, division, print_function
from plottool import plot_helpers as ph
from plottool import interact_helpers as ih
import six
import re
import utool as ut
import plottool.draw_func2 as df2
from plottool import fig_presenter
import matplotlib as mpl
ut.noinject(__name__, '[abstract_iteract]')

#(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
#'[abstract_iteract]')


# for scoping
__REGISTERED_INTERACTIONS__ = []


def register_interaction(self):
    global __REGISTERED_INTERACTIONS__
    if ut.VERBOSE:
        print('Registering intearction: self=%r' % (self,))
    __REGISTERED_INTERACTIONS__.append(self)


def unregister_interaction(self):
    global __REGISTERED_INTERACTIONS__
    if ut.VERBOSE:
        print('Unregistering intearction: self=%r' % (self,))
    __REGISTERED_INTERACTIONS__


class AbstractInteraction(object):
    """
    An interaction is meant to take up an entire figure

    overwrite either self.plot(fnum, pnum) or self.staic_plot(fnum, pnum) or show_page
    """
    def __init__(self, **kwargs):
        self.fnum = kwargs.get('fnum', None)
        if self.fnum  is None:
            self.fnum  = df2.next_fnum()
        self.interaction_name = kwargs.get('interaction_name', 'AbstractInteraction')
        self.fig = df2.figure(fnum=self.fnum, doclf=True, docla=True)
        ih.connect_callback(self.fig, 'close_event', self.on_close)
        # Careful this might cause memory leaks
        self.scope = []  # for keeping those widgets alive!
        register_interaction(self)

        self.leftbutton_is_down = None

    def clear_parent_axes(self, ax):
        """ for clearing axes that we appended anything to """
        child_axes = ph.get_plotdat(ax, 'child_axes', [])
        ph.set_plotdat(ax, 'child_axes', [])
        for subax in child_axes:
            to_remove = None
            for tup in self.scope:
                if tup[1] is subax:
                    to_remove = tup
                    break
            if to_remove is not None:
                self.scope.remove(to_remove)
            subax.cla()
            self.fig.delaxes(subax)
        ph.del_plotdat(ax, df2.DF2_DIVIDER_KEY)
        ax.cla()

    def clean_scope(self):
        """ Removes any widgets saved in the interaction scope """
        self.scope = []

    def append_button(self, text, divider=None, rect=None, callback=None,
                      size='9%', location='bottom', ax=None, **kwargs):
        """ Adds a button to the current page """

        if rect is not None:
            new_ax = df2.plt.axes(rect)
        if rect is None and divider is None:
            if ax is None:
                ax = df2.gca()
            divider = df2.ensure_divider(ax)
        if divider is not None:
            new_ax = divider.append_axes(location, size=size, pad=.05)
        if callback is not None:
            color, hovercolor = u'.85', u'.95'
        else:
            color, hovercolor = u'.88', u'.88'
            #color, hovercolor = u'.45', u'.45'
        new_but = mpl.widgets.Button(
            new_ax, text, color=color, hovercolor=hovercolor)
        if callback is not None:
            new_but.on_clicked(callback)
        else:
            button_text = new_but.ax.texts[0]
            button_text.set_color('.6')
            #button_text.set_color('r')
            #ut.embed()
            #print('new_but.color = %r' % (new_but.color,))
        #else:
        # TODO: figure ou how to gray out these buttons
        #    new_but.color = u'.1'
        #    new_but.hovercolor = u'.1'
        #    new_but.active = False
        #    print('new_but.color = %r' % (new_but.color,))
        ph.set_plotdat(new_ax, 'viztype', 'button')
        ph.set_plotdat(new_ax, 'text', text)
        #ph.set_plotdat(new_ax, 'parent_axes', ax)
        if ax is not None:
            child_axes = ph.get_plotdat(ax, 'child_axes', [])
            child_axes.append(new_ax)
            ph.set_plotdat(ax, 'child_axes', child_axes)
        for key, val in six.iteritems(kwargs):
            ph.set_plotdat(new_ax, key, val)
        # Keep buttons from losing scrop
        tup = (new_but, new_ax)
        self.scope.append(tup)
        return tup

    # def make_hud(self):

    # def prepare_page(self, pagenum):
    #    self.clean_scope()

    def show_page(self, *args):
        """
        Hack: this function should probably not be defined, but it is for
        convinience of a developer.
        Override this or create static plot function
        (preferably override)
        """
        self.fig = ih.begin_interaction(self.interaction_name, self.fnum)
        if hasattr(self, 'plot'):
            self.plot(self.fnum, (1, 1, 1))
        else:
            self.static_plot(self.fnum, (1, 1, 1))
        self.connect_callbacks()

    def connect_callbacks(self):
        ih.connect_callback(self.fig, 'button_press_event', self.on_click)
        ih.connect_callback(self.fig, 'button_release_event', self.on_click_release)
        ih.connect_callback(self.fig, 'key_press_event', self.on_key_press)
        ih.connect_callback(self.fig, 'motion_notify_event', self.on_motion)

    def bring_to_front(self):
        fig_presenter.bring_to_front(self.fig)

    def draw(self):
        self.fig.canvas.draw()

    def show(self):
        self.fig.show()

    def update(self):
        fig_presenter.update()

    def close(self):
        assert isinstance(self.fig, mpl.figure.Figure)
        fig_presenter.close_figure(self.fig)

    def on_close(self, event=None):
        print('[pt] handling interaction close')
        unregister_interaction(self)

    def on_motion(self, event):
        if self.leftbutton_is_down:
            self.on_drag(event)
        #print('event = %r' % (event.__dict__,))
        pass

    def on_drag(self, event=None):
        # Make sure BLIT (bit block transfer) is used for updates
        #self.fig.canvas.blit(self.fig.ax.bbox)
        pass

    def on_key_press(self, event):
        pass

    def on_click(self, event):
        #raise NotImplementedError('implement yourself')
        if event.button == 1:  # left
            self.leftbutton_is_down = True
        if ih.clicked_inside_axis(event):
            ax = event.inaxes
            self.on_click_inside(event, ax)
        else:
            self.on_click_outside(event)

    def on_click_release(self, event):
        if event.button == 1:  # left
            self.leftbutton_is_down = False

    def on_click_inside(self, event, ax):
        pass

    def on_click_outside(self, event):
        pass

    def show_popup_menu(self, options, event):
        import guitool
        height = self.fig.canvas.geometry().height()
        qpoint = guitool.newQPoint(event.x, height - event.y)
        qwin = self.fig.canvas
        guitool.popup_menu(qwin, qpoint, options)


class AbstractPagedInteraction(AbstractInteraction):

    def __init__(self, nPages=None, **kwargs):
        self.current_pagenum = 0
        assert nPages is not None
        self.nPages = nPages
        self.NEXT_PAGE_HOTKEYS  = ['right', 'pagedown']
        self.PREV_PAGE_HOTKEYS  = ['left', 'pageup']
        super(AbstractPagedInteraction, self).__init__(**kwargs)

    def next_page(self, event):
        #print('next')
        self.show_page(self.current_pagenum + 1)
        pass

    def prev_page(self, event):
        if self.current_pagenum == 0:
            return
        #print('prev')
        self.show_page(self.current_pagenum - 1)
        pass

    def make_hud(self):
        """ Creates heads up display """
        import plottool as pt
        # Button positioning
        #w, h = .08, .04
        #w, h = .14, .08
        w, h = .14, .07
        hl_slot, hr_slot = pt.make_bbox_positioners(y=.02, w=w, h=h,
                                                    xpad=.05, startx=0,
                                                    stopx=1)
        prev_rect = hl_slot(0)
        next_rect = hr_slot(0)
        #print('prev_rect = %r' % (prev_rect,))
        #print('next_rect = %r' % (next_rect,))

        # Create buttons
        prev_callback = None if self.current_pagenum == 0 else self.prev_page
        next_callback = None if self.current_pagenum == self.nPages - 1 else self.next_page
        prev_text = 'prev\n' + pretty_hotkey_map(self.PREV_PAGE_HOTKEYS)
        next_text = 'next\n' + pretty_hotkey_map(self.NEXT_PAGE_HOTKEYS)
        self.append_button(prev_text, callback=prev_callback, rect=prev_rect)
        self.append_button(next_text, callback=next_callback, rect=next_rect)

    def prepare_page(self, fulldraw=True):
        import plottool as pt
        ih.disconnect_callback(self.fig, 'button_press_event')
        ih.disconnect_callback(self.fig, 'button_release_event')
        ih.disconnect_callback(self.fig, 'key_press_event')
        ih.disconnect_callback(self.fig, 'motion_notify_event')

        figkw = {'fnum': self.fnum,
                 'doclf': fulldraw,
                 'docla': fulldraw, }
        if fulldraw:
            self.fig = pt.figure(**figkw)
        self.make_hud()
        ih.connect_callback(self.fig, 'button_press_event', self.on_click)
        ih.connect_callback(self.fig, 'button_release_event', self.on_click_release)
        ih.connect_callback(self.fig, 'key_press_event', self.on_key_press)
        ih.connect_callback(self.fig, 'motion_notify_event', self.on_motion)

    def on_key_press(self, event):
        if matches_hotkey(event.key, self.PREV_PAGE_HOTKEYS):
            if self.current_pagenum != 0:
                self.prev_page(event)
        if matches_hotkey(event.key, self.NEXT_PAGE_HOTKEYS):
            if self.current_pagenum != self.nPages - 1:
                self.next_page(event)
        self.draw()


def pretty_hotkey_map(hotkeys):
    if hotkeys is None:
        return ''
    hotkeys = [hotkeys] if not isinstance(hotkeys, list) else hotkeys
    mapping = {
        #'right': 'right arrow',
        #'left':  'left arrow',
    }
    mapped_hotkeys = [mapping.get(hk, hk) for hk in hotkeys]
    hotkey_str = '(' + ut.conj_phrase(mapped_hotkeys, 'or') + ')'
    return hotkey_str


def matches_hotkey(key, hotkeys):
    hotkeys = [hotkeys] if not isinstance(hotkeys, list) else hotkeys
    #flags = [re.match(hk, '^' + key + '$') for hk in hotkeys]
    flags = [re.match(hk,  key) is not None for hk in hotkeys]
    return any(flags)
