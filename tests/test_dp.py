import math

import numpy as np
import pytest

from fedthreat.client.dp import (
    PrivacyAccountant, add_gaussian_noise, clip_update, l2_norm, per_client_noise_std,
    rdp_subsampled_gaussian, rdp_gaussian,
)


def test_accountant_monotonic_and_noise_scaling():
    a = PrivacyAccountant(1e-5)
    assert a.get_epsilon() == 0.0
    a.step(1.0); e1 = a.get_epsilon()
    a.step(1.0); e2 = a.get_epsilon()
    assert 0 < e1 < e2
    lo, hi = PrivacyAccountant(1e-5), PrivacyAccountant(1e-5)
    lo.step(2.0); hi.step(0.5)
    assert lo.get_epsilon() < hi.get_epsilon()


def test_zero_noise_is_rejected_rather_than_reported_as_private():
    with pytest.raises(ValueError):
        PrivacyAccountant().step(0.0)


def test_q1_matches_closed_form_gaussian():
    for alpha in (2, 5, 17):
        assert rdp_subsampled_gaussian(1.0, 0.7, alpha) == pytest.approx(rdp_gaussian(0.7, alpha))


def test_subsampling_amplifies_privacy():
    full = rdp_subsampled_gaussian(1.0, 1.0, 8)
    sub = rdp_subsampled_gaussian(0.01, 1.0, 8)
    assert 0 < sub < full


def test_budget_enforcement_preflight():
    a = PrivacyAccountant(1e-5, max_epsilon=8.0)
    n = a.rounds_remaining(1.0)
    assert n > 0
    for _ in range(n):
        assert a.can_afford(1.0)
        a.step(1.0)
    assert not a.can_afford(1.0)
    assert a.get_epsilon() <= 8.0


def test_matches_google_dp_accounting():
    dpa = pytest.importorskip("dp_accounting")
    from dp_accounting import rdp
    for sigma, q, T in [(1.0, 1.0, 10), (1.1, 0.01, 1000), (0.8, 0.05, 200), (2.0, 1.0, 50)]:
        ours = PrivacyAccountant(1e-5)
        ours.step(sigma, q, count=T)
        ref = rdp.RdpAccountant(orders=list(range(2, 64)) + [64, 80, 96, 128, 192, 256, 512])
        ev = dpa.SelfComposedDpEvent(
            dpa.PoissonSampledDpEvent(q, dpa.GaussianDpEvent(sigma)) if q < 1 else dpa.GaussianDpEvent(sigma), T)
        ref.compose(ev)
        assert ours.get_epsilon() == pytest.approx(ref.get_epsilon(1e-5), rel=0.02), (sigma, q, T)


def test_old_server_config_was_not_actually_budget_safe():
    """Regression for the audit: sigma=0.4, 1 round already costs eps ~ 15 at delta 1e-5."""
    a = PrivacyAccountant(1e-5, max_epsilon=20.0)
    a.step(0.4)
    assert a.get_epsilon() > 10


def test_clip_update_bounds_norm_and_preserves_direction():
    u = {"a": np.full(100, 3.0, np.float32), "b": np.full((5, 5), -4.0, np.float32)}
    c, orig = clip_update(u, 1.0)
    assert orig > 1 and l2_norm(c) == pytest.approx(1.0, rel=1e-4)
    small = {"a": np.full(4, 0.01, np.float32)}
    assert l2_norm(clip_update(small, 1.0)[0]) == pytest.approx(l2_norm(small), rel=1e-5)
    with pytest.raises(ValueError):
        clip_update({"a": np.array([np.nan], np.float32)}, 1.0)


def test_distributed_noise_adds_up_to_sigma_c():
    std = per_client_noise_std(1.5, 2.0, honest_min=9)
    assert std == pytest.approx(1.5 * 2.0 / 3.0)
    rng = np.random.default_rng(0)
    d = 200_000
    total = sum(add_gaussian_noise({"x": np.zeros(d, np.float32)}, std, rng)["x"] for _ in range(9))
    assert total.std() == pytest.approx(3.0, rel=0.02)


def test_noise_rng_not_fixed_seed():
    a = add_gaussian_noise({"x": np.zeros(8, np.float32)}, 1.0)["x"]
    b = add_gaussian_noise({"x": np.zeros(8, np.float32)}, 1.0)["x"]
    assert not np.array_equal(a, b)
