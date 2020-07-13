# -*- coding: utf-8 -*-
"""
Known Interactions that use AbstractInteraction:
    pt.MatchInteraction2
    pt.MultiImageInteraction
    wbia.NameInteraction
"""
from __future__ import absolute_import, division, print_function
import six
import re
import utool as ut
import matplotlib as mpl

ut.noinject(__name__, '[abstract_iteract]')
from . import draw_func2 as df2  # NOQA
from wbia.plottool import fig_presenter  # NOQA
from wbia.plottool import plot_helpers as ph  # NOQA
from wbia.plottool import interact_helpers as ih  # NOQA

# (print, print_, printDBG, rrr, profile) = utool.inject(__name__,
# '[abstract_iteract]')

DEBUG = ut.get_argflag('--debug-interact')
VERBOSE = ut.VERBOSE or True


# for scoping
__REGISTERED_INTERACTIONS__ = []


def register_interaction(self):
    global __REGISTERED_INTERACTIONS__
    if VERBOSE:
        print('[pt] Registering intearction: self=%r' % (self,))
    __REGISTERED_INTERACTIONS__.append(self)
    if VERBOSE:
        print(
            '[pt] There are now %d registered interactions'
            % (len(__REGISTERED_INTERACTIONS__))
        )


def unregister_interaction(self):
    global __REGISTERED_INTERACTIONS__
    if VERBOSE:
        print('[pt] Unregistering intearction: self=%r' % (self,))
    try:
        __REGISTERED_INTERACTIONS__.remove(self)
    except ValueError:
        pass
    if VERBOSE:
        print(
            '[pt] There are now %d registered interactions'
            % (len(__REGISTERED_INTERACTIONS__))
        )


class AbstractInteraction(object):
    """
    An interaction is meant to take up an entire figure

    overwrite either self.plot(fnum, pnum) or self.staic_plot(fnum, pnum) or show_page
    """

    LEFT_BUTTON = 1
    MIDDLE_BUTTON = 2
    RIGHT_BUTTON = 3

    MOUSE_BUTTONS = {
        LEFT_BUTTON: 'left',
        MIDDLE_BUTTON: 'middle',
        RIGHT_BUTTON: 'right',
    }

    def __init__(self, **kwargs):
        debug = kwargs.get('debug', None)
        self.debug = DEBUG if debug is None else debug
        if self.debug:
            print('[pt.a] create interaction')
        self.fnum = kwargs.get('fnum', None)
        if self.fnum is None:
            self.fnum = df2.next_fnum()
        self.interaction_name = kwargs.get('interaction_name', 'AbstractInteraction')
        # Careful this might cause memory leaks
        self.scope = []  # for keeping those widgets alive!
        self.is_down = {}
        self.is_drag = {}
        self.is_running = False

        self.pan_event_list = []
        self.zoom_event_list = []

        self.fig = getattr(self, 'fig', None)

        for button in self.MOUSE_BUTTONS.values():
            self.is_down[button] = None
            self.is_drag[button] = None

        autostart = kwargs.get('autostart', False)
        if autostart:
            self.start()

    def reset_mouse_state(self):
        for key in self.is_down.keys():
            self.is_down[key] = False
        for key in self.is_drag.keys():
            self.is_drag[key] = False

    def enable_pan_and_zoom(self, ax):
        self.enable_zoom(ax)
        self.enable_pan(ax)

    def enable_pan(self, ax):
        from wbia.plottool.interactions import PanEvents

        pan = PanEvents(ax)
        self.pan_event_list.append(pan)

    def enable_zoom(self, ax):
        from wbia.plottool.interactions import zoom_factory

        self.zoom_event_list.append(zoom_factory(ax))

    def _start_interaction(self):
        # self.fig = df2.figure(fnum=self.fnum, doclf=True, docla=True)
        self.fig = df2.figure(fnum=self.fnum, doclf=True)
        ih.connect_callback(self.fig, 'close_event', self.on_close)
        register_interaction(self)
        self.is_running = True

    def _ensure_running(self):
        if not self.is_running:
            self._start_interaction()

    def start(self):
        self._ensure_running()
        self.show_page()
        self.show()

    def print_status(self):
        print('is_down = ' + ut.repr2(self.is_down))
        print('is_drag = ' + ut.repr2(self.is_drag))

    def _preshow_page(self):
        self._ensure_running()
        if self.debug:
            print('[pt.a] show page')
        self.fig = ih.begin_interaction(self.interaction_name, fnum=self.fnum)

    def _postshow_page(self):
        self.connect_callbacks()

    def _show_page(self):
        if hasattr(self, 'plot'):
            self.plot(fnum=self.fnum, pnum=(1, 1, 1))
        else:
            self.static_plot(fnum=self.fnum, pnum=(1, 1, 1))

    def show_page(self, *args):
        """
        Hack: this function should probably not be defined, but it is for
        convinience of a developer.
        Override this or create static plot function
        (preferably override)
        """
        self._preshow_page()
        self._show_page()
        self._postshow_page()

    def connect_callbacks(self):
        if self.debug:
            print('[pt.a] connect_callbacks')
        ih.connect_callback(self.fig, 'button_press_event', self.on_click)
        ih.connect_callback(self.fig, 'button_release_event', self.on_click_release)
        ih.connect_callback(self.fig, 'key_press_event', self.on_key_press)
        ih.connect_callback(self.fig, 'motion_notify_event', self.on_motion)
        ih.connect_callback(self.fig, 'draw_event', self.on_draw)
        ih.connect_callback(self.fig, 'scroll_event', self.on_scroll)

    def bring_to_front(self):
        import utool

        with utool.embed_on_exception_context:
            fig_presenter.bring_to_front(self.fig)

    def draw(self):
        if self.debug > 5:
            print('[pt.a] draw')
        self.fig.canvas.draw()

    def on_draw(self, event=None):
        if self.debug > 5:
            print('[pt.a] on draw')
        pass

    def show(self):
        if self.debug:
            print('[pt.a] show')
        self.fig.show()

    def update(self):
        if self.debug:
            print('[pt.a] update')
        # fig_presenter.update()
        self.fig.canvas.update()
        self.fig.canvas.flush_events()

    def on_scroll(self, event):
        if self.debug:
            print('[pt.a] on_scroll')
            print(ut.repr3(event.__dict__))
        pass

    def close(self):
        assert isinstance(self.fig, mpl.figure.Figure)
        fig_presenter.close_figure(self.fig)

    def on_close(self, event=None):
        print('[pt] handling interaction close')
        unregister_interaction(self)
        self.is_running = False

    def on_motion(self, event):
        if self.debug > 5:
            print('[pt.a] on_motion')
        for button in self.MOUSE_BUTTONS.values():
            if self.is_down[button]:
                if not self.is_drag[button]:
                    self.is_drag[button] = True
                    self.on_drag_start(event)
        if any(self.is_drag.values()):
            self.on_drag(event)

        for pan in self.pan_event_list:
            pan.pan_on_motion(event)
        # print('event = %r' % (event.__dict__,))
        pass

    def on_drag(self, event=None):
        if self.debug > 1:
            print('[pt.a] on_drag')
        if ih.clicked_inside_axis(event):
            self.on_drag_inside(event)
        # Make sure BLIT (bit block transfer) is used for updates
        # self.fig.canvas.blit(self.fig.ax.bbox)
        pass

    def on_drag_inside(self, event=None):
        if self.debug > 1:
            print('[pt.a] on_drag_inside')

    def on_drag_stop(self, event=None):
        if self.debug > 0:
            print('[pt.a] on_drag_stop')
        pass

    def on_drag_start(self, event=None):
        if self.debug > 0:
            print('[pt.a] on_drag_start')
        pass

    def on_key_press(self, event):
        if self.debug > 0:
            print('[pt.a] on_key_press')
        # self.print_status()
        pass

    def on_click(self, event):
        if self.debug > 0:
            print('[pt.a] on_click')
            # print('[pt.a] on_click. event=%r' % (ut.repr2(event.__dict__)))
        # raise NotImplementedError('implement yourself')
        if event.button is not None:
            for button in self.MOUSE_BUTTONS.values():
                if self.MOUSE_BUTTONS[event.button] == button:
                    self.is_down[button] = True
        # if event.button == self.LEFT_BUTTON:
        #    self.is_down['left'] = True
        # if event.button == self.RIGHT_BUTTON:
        #    self.is_down['right'] = True
        # if event.button == self.MIDDLE_BUTTON:
        #    self.is_down['middle'] = True
        if ih.clicked_inside_axis(event):
            ax = event.inaxes
            self.on_click_inside(event, ax)
        else:
            self.on_click_outside(event)

        for pan in self.pan_event_list:
            pan.pan_on_press(event)

    def on_click_release(self, event):
        if self.debug > 0:
            print('[pt.a] on_release')
        for button in self.MOUSE_BUTTONS.values():
            flag = (
                event is None
                or event.button is None
                or self.MOUSE_BUTTONS[event.button] == button
            )
            if flag:
                self.is_down[button] = False
                if self.is_drag[button]:
                    self.is_drag[button] = False
                    self.on_drag_stop(event)
        # if event.button == self.LEFT_BUTTON:
        #    self.is_down['left'] = False
        #    self.is_drag['left'] = False
        # if event.button == self.RIGHT_BUTTON:
        #    self.is_down['right'] = False
        # if event.button == self.MIDDLE_BUTTON:
        #    self.is_down['middle'] = False
        for pan in self.pan_event_list:
            pan.pan_on_release(event)

    def on_click_inside(self, event, ax):
        pass

    def on_click_outside(self, event):
        pass

    def show_popup_menu(self, options, event):
        """
        context menu
        """
        import wbia.guitool as gt

        height = self.fig.canvas.geometry().height()
        qpoint = gt.newQPoint(event.x, height - event.y)
        qwin = self.fig.canvas
        gt.popup_menu(qwin, qpoint, options)

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

    # def make_hud(self):

    # def prepare_page(self, pagenum):
    #    self.clean_scope()

    def clean_scope(self):
        """ Removes any widgets saved in the interaction scope """
        self.scope = []

    def append_button(
        self,
        text,
        divider=None,
        rect=None,
        callback=None,
        size='9%',
        location='bottom',
        ax=None,
        **kwargs,
    ):
        """ Adds a button to the current page """
        if rect is not None:
            new_ax = df2.plt.axes(rect)
        if rect is None and divider is None:
            if ax is None:
                ax = df2.gca()
            divider = df2.ensure_divider(ax)
        if divider is not None:
            new_ax = divider.append_axes(location, size=size, pad=0.05)
        if callback is not None:
            color, hovercolor = '.85', '.95'
        else:
            color, hovercolor = '.88', '.88'
            # color, hovercolor = u'.45', u'.45'
        # if isinstance(text, six.text_type):
        new_but = mpl.widgets.Button(new_ax, text, color=color, hovercolor=hovercolor)
        # elif isinstance(text, (list, tuple)):
        #    labels = [False] * len(text)
        #    labels[0] = True
        #    new_but = mpl.widgets.CheckButtons(new_ax, text, labels)
        # else:
        #    raise ValueError('bad input')

        if callback is not None:
            new_but.on_clicked(callback)
        else:
            button_text = new_but.ax.texts[0]
            button_text.set_color('.6')
            # button_text.set_color('r')
            # ut.embed()
            # print('new_but.color = %r' % (new_but.color,))
        # else:
        # TODO: figure ou how to gray out these buttons
        #    new_but.color = u'.1'
        #    new_but.hovercolor = u'.1'
        #    new_but.active = False
        #    print('new_but.color = %r' % (new_but.color,))
        ph.set_plotdat(new_ax, 'viztype', 'button')
        ph.set_plotdat(new_ax, 'text', text)
        # ph.set_plotdat(new_ax, 'parent_axes', ax)
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


class AbstractPagedInteraction(AbstractInteraction):
    def __init__(self, nPages=None, draw_hud=True, **kwargs):
        self.current_pagenum = 0
        assert nPages is not None
        self.nPages = nPages
        self.draw_hud = draw_hud
        self.NEXT_PAGE_HOTKEYS = ['right', 'pagedown']
        self.PREV_PAGE_HOTKEYS = ['left', 'pageup']
        super(AbstractPagedInteraction, self).__init__(**kwargs)

    def next_page(self, event):
        # print('next')
        self.show_page(self.current_pagenum + 1)
        pass

    def prev_page(self, event):
        if self.current_pagenum == 0:
            return
        # print('prev')
        self.show_page(self.current_pagenum - 1)
        pass

    def make_hud(self):
        """ Creates heads up display """
        import wbia.plottool as pt

        if not self.draw_hud:
            return
        # Button positioning
        # w, h = .08, .04
        # w, h = .14, .08
        w, h = 0.14, 0.07
        hl_slot, hr_slot = pt.make_bbox_positioners(
            y=0.02, w=w, h=h, xpad=0.05, startx=0, stopx=1
        )
        prev_rect = hl_slot(0)
        next_rect = hr_slot(0)
        # print('prev_rect = %r' % (prev_rect,))
        # print('next_rect = %r' % (next_rect,))

        # Create buttons
        prev_callback = None if self.current_pagenum == 0 else self.prev_page
        next_callback = (
            None if self.current_pagenum == self.nPages - 1 else self.next_page
        )
        prev_text = 'prev\n' + pretty_hotkey_map(self.PREV_PAGE_HOTKEYS)
        next_text = 'next\n' + pretty_hotkey_map(self.NEXT_PAGE_HOTKEYS)
        self.append_button(prev_text, callback=prev_callback, rect=prev_rect)
        self.append_button(next_text, callback=next_callback, rect=next_rect)

    def prepare_page(self, fulldraw=True):
        import wbia.plottool as pt

        ih.disconnect_callback(self.fig, 'button_press_event')
        ih.disconnect_callback(self.fig, 'button_release_event')
        ih.disconnect_callback(self.fig, 'key_press_event')
        ih.disconnect_callback(self.fig, 'motion_notify_event')

        figkw = {
            'fnum': self.fnum,
            'doclf': fulldraw,
            'docla': fulldraw,
        }
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


# moved to interactions.py
# def zoom_factory(ax, zoomable_list, base_scale=1.1):
#    """
#    TODO: make into interaction

#    References:
#        https://gist.github.com/tacaswell/3144287
#    """
#    def zoom_fun(event):
#        #print('zooming')
#        # get the current x and y limits
#        cur_xlim = ax.get_xlim()
#        cur_ylim = ax.get_ylim()
#        xdata = event.xdata  # get event x location
#        ydata = event.ydata  # get event y location
#        if xdata is None or ydata is None:
#            return
#        if event.button == 'up':
#            # deal with zoom in
#            scale_factor = 1 / base_scale
#        elif event.button == 'down':
#            # deal with zoom out
#            scale_factor = base_scale
#        else:
#            raise NotImplementedError('event.button=%r' % (event.button,))
#            # deal with something that should never happen
#            scale_factor = 1
#            print(event.button)
#        for zoomable in zoomable_list:
#            zoom = zoomable.get_zoom()
#            new_zoom = zoom / (scale_factor ** (1.2))
#            zoomable.set_zoom(new_zoom)
#        # Get distance from the cursor to the edge of the figure frame
#        x_left = xdata - cur_xlim[0]
#        x_right = cur_xlim[1] - xdata
#        y_top = ydata - cur_ylim[0]
#        y_bottom = cur_ylim[1] - ydata
#        ax.set_xlim([xdata - x_left * scale_factor, xdata + x_right * scale_factor])
#        ax.set_ylim([ydata - y_top * scale_factor, ydata + y_bottom * scale_factor])

#        # ----
#        ax.figure.canvas.draw()  # force re-draw

#    fig = ax.get_figure()  # get the figure of interest
#    # attach the call back
#    fig.canvas.mpl_connect('scroll_event', zoom_fun)

#    #return the function
#    return zoom_fun


def pretty_hotkey_map(hotkeys):
    if hotkeys is None:
        return ''
    hotkeys = [hotkeys] if not isinstance(hotkeys, list) else hotkeys
    mapping = {
        # 'right': 'right arrow',
        # 'left':  'left arrow',
    }
    mapped_hotkeys = [mapping.get(hk, hk) for hk in hotkeys]
    hotkey_str = '(' + ut.conj_phrase(mapped_hotkeys, 'or') + ')'
    return hotkey_str


def matches_hotkey(key, hotkeys):
    hotkeys = [hotkeys] if not isinstance(hotkeys, list) else hotkeys
    # flags = [re.match(hk, '^' + key + '$') for hk in hotkeys]
    flags = [re.match(hk, key) is not None for hk in hotkeys]
    return any(flags)
