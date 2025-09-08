import os
import glob
import sys
try:
    import torch
except Exception as e:
    print('[light-mode] torch not importable, nothing to prune:', e)
    sys.exit(0)
libdir = os.path.join(os.path.dirname(torch.__file__), 'lib')
patterns = ['libtorch_cuda*', 'libnvrtc*', 'libcudnn*', 'libcublas*']
removed = 0
for pat in patterns:
    for path in glob.glob(os.path.join(libdir, pat)):
        try:
            os.remove(path)
            removed += 1
            print('[light-mode] removed', path)
        except OSError as ex:
            print('[light-mode] failed to remove', path, ex)
print(f'[light-mode] total removed {removed} files')
