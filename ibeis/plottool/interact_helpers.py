from __future__ import absolute_import, division, print_function
from plottool import draw_func2 as df2
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_helpers]',
                                                       DEBUG=False)
#==========================
# HELPERS
#==========================

# RCOS TODO: We should change the fnum, pnum figure layout into one managed by
# gridspec.


def detect_keypress(fig):
    def on_key_press(event):
        if event.key == 'shift':
            shift_is_held = True  # NOQA

    def on_key_release(event):
        if event.key == 'shift':
            shift_is_held = False  # NOQA
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)


def clicked_inside_axis(event):
    in_axis = event is not None and (event.inaxes is not None and event.xdata is not None)
    if not in_axis:
        print(' ...out of axis')
    else:
        print(' ...in axis')
    return in_axis


def clicked_outside_axis(event):
    return not clicked_inside_axis(event)


def begin_interaction(type_, fnum):
    printDBG('\n<<<<  BEGIN %s INTERACTION >>>>' % (str(type_).upper()))
    #printDBG('[inter] starting %s interaction' % type_)
    fig = df2.figure(fnum=fnum, docla=True, doclf=True)
    ax = df2.gca()
    disconnect_callback(fig, 'button_press_event', axes=[ax])
    return fig


def disconnect_callback(fig, callback_type, **kwargs):
    #print('[df2] disconnect %r callback' % callback_type)
    axes = kwargs.get('axes', [])
    for ax in axes:
        ax._hs_viztype = ''
    cbid_type = callback_type + '_cbid'
    cbfn_type = callback_type + '_func'
    cbid = fig.__dict__.get(cbid_type, None)
    cbfn = fig.__dict__.get(cbfn_type, None)
    if cbid is not None:
        fig.canvas.mpl_disconnect(cbid)
    else:
        cbfn = None
    fig.__dict__[cbid_type] = None
    return cbid, cbfn


def connect_callback(fig, callback_type, callback_fn):
    printDBG('[ih] register %r callback' % callback_type)
    if callback_fn is None:
        return
    # Store the callback in the figure diction so it doesnt lose scope
    cbid_type = callback_type + '_cbid'
    cbfn_type = callback_type + '_func'
    fig.__dict__[cbid_type] = fig.canvas.mpl_connect(callback_type, callback_fn)
    fig.__dict__[cbfn_type] = callback_fn


REGIESTERED_INTERACTIONS = []


def register_interaction(interaction):
    global REGIESTERED_INTERACTIONS
    REGIESTERED_INTERACTIONS.append(interaction)
