#!/usr/bin/env python3
"""What a qualification report must prove before it can promote a release.

`activate` used to require only that a report EXIST. It read `checks_passed` and
`checks_total`, printed them, and never compared them -- so a release whose
evidence said 8/11 activated exactly like one that said 11/11, an empty report
satisfied the condition, and the detached signature then attested to a binary
nobody had qualified. The 2026-09-26 review found it.

These are the negative cases that review asked for: failed, empty, incomplete
and wrong-profile reports. A gate is only worth having if it refuses.

Run: python3 tests/test_mimo26_package_gating.py
"""
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "mimo26_package", HERE.parent / "tools" / "mimo26_package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)

checks = 0


def report(names=None, failed=(), stock=True, weight_reuse=False, empty=False):
    """A synthetic report covering the mandatory set unless told otherwise."""
    if empty:
        entries = []
    else:
        covered = list(package.MANDATORY_CHECKS) + ["soak 20 requests, no faults"]
        entries = [{"check": name, "pass": name not in failed, "detail": ""}
                   for name in (names if names is not None else covered)]
    profile = {"context": 131072, "expert_slots": 160}
    if weight_reuse is not None:
        profile["expert_weight_reuse"] = weight_reuse
    return {"label": "synthetic", "stock_profile": stock,
            "profile": profile, "checks": entries}


def parsed(text):
    return package.parse_expected_profile(text)


def bump():
    global checks
    checks += 1


def case(name, reasons, expect_refused, expect_contains=None):
    global checks
    refused = bool(reasons)
    if refused != expect_refused:
        print(f"FAIL {name}: refused={refused} expected={expect_refused} {reasons}")
        sys.exit(1)
    if expect_contains and not any(expect_contains in r for r in reasons):
        print(f"FAIL {name}: no reason mentioning {expect_contains!r}; got {reasons}")
        sys.exit(1)
    print(f"ok   {name}"
          + (f"  -> {reasons[0][:72]}" if reasons else "  -> accepted"))
    checks += 1


def main():
    short = package.qualification_shortfall

    # The one report that should promote.
    case("complete, all passing, stock", short(report()), False)

    # Failed outcomes. This is the hole the review found.
    case("one check failed",
         short(report(failed={"reasoning answer correct"})), True, "failed")
    case("every check failed",
         short(report(failed=set(package.MANDATORY_CHECKS))), True, "failed")

    # Empty and missing coverage.
    case("no checks at all", short(report(empty=True)), True, "no checks")
    case("checks key absent", short({"stock_profile": True}), True, "no checks")
    case("checks not a list", short({"checks": {}, "stock_profile": True}), True, "no checks")

    # Incomplete: passing, but not the required set.
    case("only the first two checks",
         short(report(names=list(package.MANDATORY_CHECKS[:2]))), True, "mandatory set")
    case("mandatory set but no soak",
         short(report(names=list(package.MANDATORY_CHECKS))), True, "mandatory set")
    case("soak present, one mandatory name missing",
         short(report(names=list(package.MANDATORY_CHECKS[1:])
                      + ["soak 20 requests, no faults"])), True, "mandatory set")

    # Wrong profile. Non-stock is allowed only when the intended profile is
    # STATED -- production runs non-stock deliberately, so refusing it outright
    # would make the real serving profile unpromotable.
    case("non-stock with nothing stated", short(report(stock=False)), True, "stock profile")
    case("non-stock, profile stated and matching",
         short(report(stock=False), expected_profile={"context": 131072}), False)
    case("non-stock, stated profile disagrees",
         short(report(stock=False), expected_profile={"context": 262144}),
         True, "profile mismatch")
    case("stated profile names a field the report lacks",
         short(report(stock=False), expected_profile={"tensor_parallel": 4}),
         True, "absent from the report")
    case("stock report, stated profile still checked",
         short(report(), expected_profile={"expert_slots": 999}), True, "profile mismatch")
    case("experimental kernel, not allowed",
         short(report(weight_reuse=True)), True, "tiled expert")
    case("experimental kernel, allowed deliberately",
         short(report(weight_reuse=True), allow_experimental_kernel=True), False)

    # A report from a binary predating the flag has no such mode to be in.
    case("weight-reuse field absent", short(report(weight_reuse=None)), False)

    # A failing check must be refused even when everything else is right, and a
    # failed check that is NOT in the mandatory set still counts as a failure.
    case("extra non-mandatory check failed",
         short(report(names=list(package.MANDATORY_CHECKS)
                      + ["soak 20 requests, no faults", "some future check"],
                      failed={"some future check"})), True, "failed")

    # The --expect-profile parser itself.
    assert parsed("") == {}, parsed("")
    assert parsed("context=131072") == {"context": 131072}
    assert parsed("kv_prefix_reuse=true,auth=off") == {"kv_prefix_reuse": True, "auth": "off"}
    assert parsed(" a=false , b=-5 ") == {"a": False, "b": -5}
    print("ok   --expect-profile parses ints, bools and strings")
    bump()

    print(f"\nPASS {checks} package gating cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
