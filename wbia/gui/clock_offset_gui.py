# -*- coding: utf-8 -*-
"""
Small GUI for asking the user to enter the clock time shown, and moving along a gid list if the first image isn't a clock
"""
from __future__ import absolute_import, division, print_function
from functools import partial
from six.moves import range
from time import mktime
from datetime import date, datetime
import utool as ut
import wbia.guitool as gt
from wbia.guitool.__PYQT__ import QtWidgets
from wbia.guitool.__PYQT__.QtCore import Qt
import wbia.plottool as pt

(print, rrr, profile) = ut.inject2(__name__, '[co_gui]')


class ClockOffsetWidget(QtWidgets.QWidget):
    def __init__(co_wgt, ibs, gid_list, parent=None, hack=False):
        print('[co_gui] Initializing')
        print('[co_gui] gid_list = %r' % (gid_list,))

        QtWidgets.QWidget.__init__(co_wgt, parent=parent)

        co_wgt.fnum = pt.next_fnum()

        co_wgt.main_layout = QtWidgets.QVBoxLayout(co_wgt)

        co_wgt.text_layout = gt.newWidget(
            co_wgt, orientation=Qt.Vertical, verticalStretch=10
        )
        co_wgt.main_layout.addWidget(co_wgt.text_layout)

        co_wgt.control_layout = gt.newWidget(
            co_wgt,
            orientation=Qt.Vertical,
            verticalSizePolicy=QtWidgets.QSizePolicy.MinimumExpanding,
        )
        co_wgt.main_layout.addWidget(co_wgt.control_layout)

        co_wgt.button_layout = gt.newWidget(co_wgt, orientation=Qt.Horizontal)
        co_wgt.control_layout.addWidget(co_wgt.button_layout)

        co_wgt.combo_layout = gt.newWidget(co_wgt, orientation=Qt.Horizontal)
        co_wgt.control_layout.addWidget(co_wgt.combo_layout)

        co_wgt.hack = hack  # hack for edting the time of a single image

        co_wgt.imfig = None
        co_wgt.imax = None

        co_wgt.dtime = None

        co_wgt.ibs = ibs
        co_wgt.gid_list = gid_list
        co_wgt.current_gindex = 0
        co_wgt.offset = 0
        # Set image datetime with first image
        co_wgt.get_image_datetime_()
        co_wgt.add_label()
        co_wgt.add_combo_boxes()
        co_wgt.add_buttons()
        co_wgt.update_ui()

    def resizeEvent(co_wgt, event):
        super(ClockOffsetWidget, co_wgt).resizeEvent(event)
        if co_wgt.image_label is not None:
            # Hack use signals and slots
            if hasattr(co_wgt.image_label, '_on_resize_slot'):
                co_wgt.image_label._on_resize_slot()
        # print('resizeEvent')

    def get_image_datetime_(co_wgt):
        # Function that extracts the unixtime from an image and stores it
        utime = co_wgt.ibs.get_image_unixtime(co_wgt.gid_list[co_wgt.current_gindex])
        if not isinstance(utime, (float, int)) or utime is None:
            utime = -1
        utime = float(utime)
        co_wgt.dtime = datetime.fromtimestamp(utime)

    def update_ui(co_wgt):
        if not co_wgt.hack:
            if co_wgt.current_gindex == 0:
                co_wgt.button_list[0].setEnabled(False)
            else:
                co_wgt.button_list[0].setEnabled(True)
            if co_wgt.current_gindex == len(co_wgt.gid_list) - 1:
                co_wgt.button_list[1].setEnabled(False)
            else:
                co_wgt.button_list[1].setEnabled(True)

        def extract_tuple(li, idx):
            return list(zip(*li)[idx])

        # Update option setting, assume datetime has been updated
        co_wgt.combo_list[1].setCurrentIndex(
            extract_tuple(co_wgt.opt_list['year'], 1).index(co_wgt.dtime.year)
        )
        co_wgt.combo_list[3].setCurrentIndex(
            extract_tuple(co_wgt.opt_list['month'], 1).index(co_wgt.dtime.month)
        )
        co_wgt.combo_list[5].setCurrentIndex(
            extract_tuple(co_wgt.opt_list['day'], 1).index(co_wgt.dtime.day)
        )
        co_wgt.combo_list[7].setCurrentIndex(
            extract_tuple(co_wgt.opt_list['hour'], 1).index(co_wgt.dtime.hour)
        )
        co_wgt.combo_list[9].setCurrentIndex(
            extract_tuple(co_wgt.opt_list['minute'], 1).index(co_wgt.dtime.minute)
        )
        co_wgt.combo_list[11].setCurrentIndex(
            extract_tuple(co_wgt.opt_list['second'], 1).index(co_wgt.dtime.second)
        )

        # Redraw image
        if not co_wgt.hack:
            if co_wgt.imfig is not None:
                pt.close_figure(co_wgt.imfig)
            image = co_wgt.ibs.get_images(co_wgt.gid_list[co_wgt.current_gindex])
            figtitle = 'Time Synchronization Picture'
            co_wgt.imfig, co_wgt.imax = pt.imshow(image, fnum=co_wgt.fnum, title=figtitle)
            co_wgt.imfig.show()

    def show_helpmsg(co_wgt):
        msg = ut.textblock(
            """
            This step is for synchronizing the time of the images being imported
            with the actual time of the imageset.

            Use the Previous and Next buttons until the 'Time Synchronization Picture' is of the clock
            taken at the beginning of the imageset.

            Once found, change the date and time in the boxes below to match the time of the clock in
            the image, and correct the date if necessary.

            Once done, click 'Set' and the difference will be applied to all images currently
            being imported.  If you are sure the camera was synchronized, you
            can skip this step by pressing 'Skip'.
            """
        )
        gt.user_info(co_wgt, msg=msg, title='Time Sync Help')

    def add_label(co_wgt):
        # Very simply adds the text
        _LABEL = partial(gt.newLabel, parent=co_wgt)
        if not co_wgt.hack:
            text = ut.codeblock(
                """
                * Find the image of the clock
                * Set the sliders to correspond with the clock
                * Click Set
                * Skip if time synchonization is not relevant to you
                """
            )
            main_label = _LABEL(text=text, align='left')
            co_wgt.text_layout.addWidget(main_label)
        else:
            text = ut.codeblock(
                """
                * Set the time for image %r
                """
                % (co_wgt.gid_list,)
            )
            gpath = co_wgt.ibs.get_image_paths(co_wgt.gid_list[co_wgt.current_gindex])
            image_label = _LABEL(text='', gpath=gpath)  # align='left')
            co_wgt.image_label = image_label
            co_wgt.text_layout.addWidget(image_label)
            # main_label = _LABEL(text=text, align='left')
            # co_wgt.text_layout.addWidget(main_label)

    def add_buttons(co_wgt):
        _BUTTON = partial(gt.newButton, parent=co_wgt)
        if not co_wgt.hack:
            co_wgt.button_list = [
                _BUTTON(text='Previous Image', clicked=co_wgt.go_prev),
                _BUTTON(text='Next Image', clicked=co_wgt.go_next),
                _BUTTON(text='Skip', clicked=co_wgt.cancel),
                _BUTTON(text='Set', clicked=co_wgt.accept),
                _BUTTON(text='Help', clicked=co_wgt.show_helpmsg),
            ]
        else:
            co_wgt.button_list = [
                _BUTTON(text='Accept', clicked=co_wgt.accept),
            ]
        # Copying from inspect_gui

        for button in co_wgt.button_list:
            co_wgt.button_layout.addWidget(button)

    def add_combo_boxes(co_wgt):
        _CBOX = partial(gt.newComboBox, parent=co_wgt)
        _LABEL = partial(gt.newLabel, parent=co_wgt, align='right')

        def to_opt_list(x):
            return [('%02d' % i, i) for i in x]

        year_list = [
            datetime.fromtimestamp(unixtime).year
            for unixtime in co_wgt.ibs.get_image_unixtime(co_wgt.gid_list)
            if unixtime is not None
        ]
        max_year = max(year_list + [date.today().year]) + 1
        min_year = min(year_list + [1970]) - 1
        # max_year = date.today().year + 1
        co_wgt.opt_list = {
            'year': to_opt_list(range(min_year, max_year)),
            'month': to_opt_list(range(1, 13)),
            'day': to_opt_list(
                range(1, 32)
            ),  # yes this will allow invalid dates, but it's not a real issue
            'hour': to_opt_list(range(24)),
            'minute': to_opt_list(range(60)),
            'second': to_opt_list(range(60)),
        }
        co_wgt.combo_list = [
            _LABEL(text='YYYY'),
            _CBOX(
                changed=partial(co_wgt.change_dt, 'year'),
                options=co_wgt.opt_list['year'],
                default=co_wgt.dtime.year,
            ),
            _LABEL(text='MM'),
            _CBOX(
                changed=partial(co_wgt.change_dt, 'month'),
                options=co_wgt.opt_list['month'],
                default=co_wgt.dtime.month,
            ),
            _LABEL(text='DD'),
            _CBOX(
                changed=partial(co_wgt.change_dt, 'day'),
                options=co_wgt.opt_list['day'],
                default=co_wgt.dtime.day,
            ),
            _LABEL(text='HH (24H)'),
            _CBOX(
                changed=partial(co_wgt.change_dt, 'hour'),
                options=co_wgt.opt_list['hour'],
                default=co_wgt.dtime.hour,
            ),
            _LABEL(text='mm'),
            _CBOX(
                changed=partial(co_wgt.change_dt, 'minute'),
                options=co_wgt.opt_list['minute'],
                default=co_wgt.dtime.minute,
            ),
            _LABEL(text='ss'),
            _CBOX(
                changed=partial(co_wgt.change_dt, 'second'),
                options=co_wgt.opt_list['second'],
                default=co_wgt.dtime.second,
            ),
        ]

        for combo in co_wgt.combo_list:
            co_wgt.combo_layout.addWidget(combo)

    @gt.slot_()
    def go_prev(co_wgt):
        # Decrement current gid index, then call update
        co_wgt.current_gindex -= 1
        co_wgt.get_image_datetime_()
        co_wgt.update_ui()

    @gt.slot_()
    def go_next(co_wgt):
        # Increment current gid index, then call update
        co_wgt.current_gindex += 1
        co_wgt.get_image_datetime_()
        co_wgt.update_ui()

    @gt.slot_()
    def cancel(co_wgt):
        # Just close
        pt.close_figure(co_wgt.imfig)
        co_wgt.close()

    @gt.slot_()
    def accept(co_wgt):
        # Calculate offset
        # Get current image's actual time
        image_time = co_wgt.ibs.get_image_unixtime(co_wgt.gid_list[co_wgt.current_gindex])
        # Get unixtime from current dtime
        input_time = mktime(co_wgt.dtime.timetuple())
        # Go through gid list, and add that offset to every unixtime
        offset = input_time - image_time
        print('[co_gui] Unixtime offset is %d' % offset)
        # Set the unixtimes with the new ones
        utimes = co_wgt.ibs.get_image_unixtime(co_wgt.gid_list)
        new_utimes = [time + offset for time in utimes]
        # SHOULD WE BE USING image_timedelta_posix INSTEAD OF THIS?
        co_wgt.ibs.set_image_unixtime(co_wgt.gid_list, new_utimes)
        # Close
        if not co_wgt.hack:
            pt.close_figure(co_wgt.imfig)
        co_wgt.close()

    @gt.slot_(str, int, str)
    def change_dt(co_wgt, attr, val_ind, val):
        co_wgt.dtime = co_wgt.dtime.replace(**{attr: co_wgt.opt_list[attr][val_ind][1]})
        print('[co_gui] Base datetime is %r' % co_wgt.dtime)


def clock_offset_test():
    r"""
    CommandLine:
        python -m wbia.gui.clock_offset_gui --test-clock_offset_test

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.gui.clock_offset_gui import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = clock_offset_test()
        >>> # verify results
        >>> print(result)
    """
    import wbia

    main_locals = wbia.main(db='testdb1')
    ibs = main_locals['ibs']
    gid_list = ibs.get_valid_gids()
    co = ClockOffsetWidget(ibs, gid_list, hack=True)
    co.show()
    wbia.main_loop(main_locals)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.gui.clock_offset_gui
        python -m wbia.gui.clock_offset_gui --allexamples
        python -m wbia.gui.clock_offset_gui --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
