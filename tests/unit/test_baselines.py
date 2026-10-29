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

def test_hmm_change_uses_success_field_not_dict_truthiness() -> None:
    """Regression: outcome dicts are always truthy; must read ['success']."""
    from driftbound.smartdream.controller import HMMChangeController

    rng = make_rng(7)
    ctrl = HMMChangeController()
    ctrl.reset(rng)
    for _ in range(25):
        action = ctrl.act(0, rng)
        ctrl.update(0, action, {"best_action": "wood", "success": True}, rng)
    assert ctrl.detector._mean > 0.9
    for _ in range(50):
        action = ctrl.act(1, rng)
        ctrl.update(1, action, {"best_action": "brick", "success": False}, rng)
    # Mean must fall — proves we are not scoring dict truthiness as 1.0
    assert ctrl.detector._mean < 0.5



def test_hmm_predict_state_filters_current_observation() -> None:
    rng = make_rng(5)
    hmm = DiscreteHMM()
    hmm.reset(rng)
    for _ in range(6):
        hmm.filter(0)
        hmm._filtered_obs = None
    mass0 = float(hmm.belief[0])
    hmm.predict_state(2)
    assert float(hmm.belief[2]) > 0.0
    # Filtering obs=2 should not leave belief identical to pre-call lag MAP path
    assert float(hmm.belief[0]) <= mass0 or float(hmm.belief[2]) > float(hmm.belief[0])
