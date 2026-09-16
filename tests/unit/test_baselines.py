from driftbound.baselines.change_detection import PageHinkleyDetector
from driftbound.baselines.hmm import DiscreteHMM
from driftbound.baselines.switching_experts import FixedShareExperts
from driftbound.core.random import make_rng


def test_hmm_updates() -> None:
    rng = make_rng(5)
    hmm = DiscreteHMM()
    hmm.reset(rng)
    s0 = hmm.predict_state(0)
    hmm.update(0, rng)
    assert isinstance(s0, int)


def test_page_hinkley_detects_shift() -> None:
    d = PageHinkleyDetector(delta=0.005, threshold=0.05, min_instances=20)
    for _ in range(30):
        d.update(0.1)
    for _ in range(40):
        d.update(0.9)
    assert d.changed


def test_fixed_share_runs() -> None:
    rng = make_rng(6)
    fs = FixedShareExperts()
    fs.reset(rng)
    a = fs.act(0, rng)
    fs.update(0, a, {"best_action": "brick"}, rng)
    assert a in fs.actions
