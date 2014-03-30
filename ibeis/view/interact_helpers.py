from drawtool import draw_func2 as df2
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


def is_event_valid(event, valid_tests=['click']):
    is_valid = event is not None
    if 'click' in valid_tests:
        is_valid = is_valid and (event.inaxes is not None and event.xdata is not None)
    return is_valid


def begin_interaction(type_, fnum):
    print('[inter] starting %s interaction' % type_)
    fig = df2.figure(fnum=fnum, docla=True, doclf=True)
    ax = df2.gca()
    df2.disconnect_callback(fig, 'button_press_event', axes=[ax])
    return fig
