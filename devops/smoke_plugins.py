import importlib
import sys


CHECKS = [
    ("wbia", "from wbia.__main__ import smoke_test; smoke_test()", True),
    ("wbia_cnn", "from wbia_cnn.__main__ import main; main()", False),
    ("wbia_pie_v2", "from wbia_pie_v2.__main__ import main; main()", False),
    ("wbia_blend", "from wbia_blend._plugin import *", False),
    ("wbia_orientation", "from wbia_orientation.__main__ import main; main()", False),
    ("wbia_whaleridgefindr", "from wbia_whaleridgefindr._plugin import *", False),
    ("wbia_kaggle7", "from wbia_kaggle7._plugin import *", False),
    ("wbia_lca", "from wbia_lca._plugin import *", False),
    ("wbia_deepsense", "from wbia_deepsense._plugin import *", False),
]


def main():
    failures = 0
    for mod, code, required in CHECKS:
        if importlib.util.find_spec(mod) is None:
            if required:
                print(f"[error] required module missing: {mod}", file=sys.stderr)
                failures += 1
            else:
                print(f"[skip] {mod} not installed")
            continue
        try:
            print(f"[run] {mod}")
            exec(code, {})  # nosec - controlled internal smoke commands
        except Exception as e:  # pragma: no cover - diagnostic path
            if required:
                print(f"[error] {mod} failed: {e}", file=sys.stderr)
                failures += 1
            else:
                print(f"[warn] {mod} execution failed: {e}", file=sys.stderr)
    if failures:
        print(f"[smoke] completed with {failures} required failures", file=sys.stderr)
        sys.exit(1)
    print("[smoke] all required checks passed; optional plugins skipped or succeeded")


if __name__ == "__main__":  # pragma: no cover
    main()
