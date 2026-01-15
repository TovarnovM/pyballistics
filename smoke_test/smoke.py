from __future__ import annotations

import importlib.metadata as md


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    print("Installed pyballistics:", md.version("pyballistics"))

    # Import surface API
    from pyballistics import (
        get_options_sample,
        get_full_options,
        get_powder_names,
        ozvb_termo,
        ozvb_lagrange,
    )

    # 1) Data file / options helpers
    powders = get_powder_names()
    _assert(isinstance(powders, (list, tuple)) and len(powders) > 0, "get_powder_names() returned empty")
    print("Powders:", len(powders))

    opts = get_options_sample()
    _assert(isinstance(opts, dict) and len(opts) > 0, "get_options_sample() did not return a non-empty dict")

    full_opts = get_full_options(opts)
    _assert(isinstance(full_opts, dict) and len(full_opts) > 0, "get_full_options() did not return a non-empty dict")

    # 2) Core compute entrypoints
    res_t = ozvb_termo(opts)
    _assert(isinstance(res_t, dict), "ozvb_termo() must return dict")
    _assert("stop_reason" in res_t, "ozvb_termo() result missing 'stop_reason'")
    _assert(res_t["stop_reason"] != "error", f"ozvb_termo stop_reason=error: {res_t.get('error', '')}")
    print("ozvb_termo OK. stop_reason:", res_t["stop_reason"])

    res_l = ozvb_lagrange(opts)
    _assert(isinstance(res_l, dict), "ozvb_lagrange() must return dict")
    _assert("stop_reason" in res_l, "ozvb_lagrange() result missing 'stop_reason'")
    _assert(res_l["stop_reason"] != "error", f"ozvb_lagrange stop_reason=error: {res_l.get('error', '')}")
    print("ozvb_lagrange OK. stop_reason:", res_l["stop_reason"])

    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
