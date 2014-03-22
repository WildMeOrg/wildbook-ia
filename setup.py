from os.path import join, dirname, realpath, splitext
from ibeis.util import util_path
import os
import sys


def compile_ui():
    'Compiles the qt designer *.ui files into python code'
    pyuic4_cmd = {'win32':  'C:\Python27\Lib\site-packages\PyQt4\pyuic4',
                  'linux2': 'pyuic4',
                  'darwin': 'pyuic4'}[sys.platform]
    widget_dir = join(dirname(realpath(__file__)), 'ibeis/view')
    print('[setup] Compiling qt designer files in %r' % widget_dir)
    for widget_ui in util_path.glob(widget_dir, '*.ui'):
        widget_py = splitext(widget_ui)[0] + '.py'
        cmd = ' '.join([pyuic4_cmd, '-x', widget_ui, '-o', widget_py])
        print('[setup] compile_ui()>' + cmd)
        os.system(cmd)


if __name__ == '__main__':
    print('[setup] Entering HotSpotter setup')
    for cmd in iter(sys.argv[1:]):
        # Build PyQt UI files
        if cmd in ['buildui', 'ui', 'compile_ui']:
            compile_ui()
            sys.exit(0)
