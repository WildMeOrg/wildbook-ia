import importlib
import sys

try:
    import yaml  # type: ignore
    print('PyYAML_VERSION', yaml.__version__)
except Exception as e:  # pragma: no cover
    print('PyYAML import failed', e, file=sys.stderr)
if importlib.util.find_spec('yorm') is None:
    print('yorm: NOT INSTALLED (expected)')
else:
    print('WARNING: yorm PRESENT', file=sys.stderr)
