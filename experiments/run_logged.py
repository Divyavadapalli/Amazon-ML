"""Run a module with stdout/stderr written to files (for fully detached launches, where the
console handles are unavailable). Exceptions are written to the stderr file and re-raised;
the exit code is appended to the stdout file as EXIT=<code>.

Usage: python -m experiments.run_logged <stdout_file> <stderr_file> <module> [args...]
"""
import runpy
import sys
import traceback

if __name__ == "__main__":
    out, err, mod, args = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    fo = open(out, "a", encoding="utf-8", buffering=1)
    fe = open(err, "a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = fo, fe
    sys.argv = [mod, *args]
    code = 0
    try:
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        traceback.print_exc(file=fe)
        code = 1
    fo.write(f"EXIT={code}\n")
    fo.flush()
    fe.flush()
    sys.exit(code)
