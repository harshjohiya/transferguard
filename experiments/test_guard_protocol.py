"""Plain-python checks for guard_protocol.py (no torch, no GPU). Run: python experiments/test_guard_protocol.py"""
import guard_protocol as gp


def test_rule_matches_notebook_values():
    # Notebook 3 reported conditions: 40% -> frozen, 80% -> scratch.
    assert gp.guard_choice(64.2, 70.0, 50.0) == "frozen"
    assert gp.guard_choice(9.8, 10.65, 50.0) == "scratch"


def test_thresholds_are_inclusive():
    assert gp.guard_choice(20.0, 52.0, 50.0) == "frozen"
    assert gp.guard_choice(19.9, 52.0, 50.0) == "scratch"
    assert gp.guard_choice(20.0, 51.9, 50.0) == "scratch"


def test_regret_and_correctness():
    scratch = [40, 45, 48, 50, 52] + [57] * 25
    frozen = [60, 62, 64, 66, 67] + [67] * 25
    r = gp.evaluate_condition(43, 40, 64.0, 0.02, scratch, frozen)
    assert r["guard_choice"] == "frozen" and r["better_method"] == "frozen"
    assert r["guard_regret"] == 0 and r["guard_correct"]
    assert not r["development_condition"] and not r["collapsed"]
    # A wrong choice must show positive regret.
    r2 = gp.evaluate_condition(43, 0, 5.0, 0.02, scratch, frozen)
    assert r2["guard_choice"] == "scratch" and r2["guard_regret"] == 10 and not r2["guard_correct"]


def test_development_flag_and_summary():
    c = [10.65] * 30
    s = [57.0] * 30
    a = gp.evaluate_condition(42, 80, 9.8, 1e-9, s, c)
    b = gp.evaluate_condition(43, 80, 9.7, 1e-9, s, c)
    assert a["development_condition"] and not b["development_condition"] and b["collapsed"]
    kinds = {x["kind"] for x in gp.summarize([a, b])}
    assert kinds == {"development", "independent"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
