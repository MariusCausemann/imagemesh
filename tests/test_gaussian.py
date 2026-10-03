"""Tests for the Young-van Vliet recursive Gaussian filter."""

import numpy as np
import pytest
import scipy.ndimage as ndi

from imagemesh.gaussian import compute_yvv_coeffs, gaussian, yvv_gaussian_filter_3d


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_compute_yvv_coeffs_dc_gain():
    """The coefficients give a forward+backward DC gain of 1 (B / (1 - b1 - b2 - b3) = 1)."""
    for sigma in [0.6, 1.0, 2.0, 2.5, 4.0, 8.0, 10.0]:
        B, b1, b2, b3 = compute_yvv_coeffs(sigma)
        # DC gain of a single pass: B / (1 - b1 - b2 - b3)
        denom = 1.0 - (b1 + b2 + b3)
        assert np.isclose(B, denom, atol=1e-12)


def test_constant_input_is_preserved():
    image = np.full((20, 22, 24), 5.3, dtype=np.float64)
    out = yvv_gaussian_filter_3d(image, sigma=2.0)
    # IIR filter with unit DC gain and reflecting boundary preserves constants.
    np.testing.assert_allclose(out, image, atol=1e-8)


def test_zero_input_is_preserved():
    image = np.zeros((10, 11, 12), dtype=np.float64)
    out = yvv_gaussian_filter_3d(image, sigma=3.0)
    np.testing.assert_allclose(out, np.zeros_like(image), atol=1e-10)


def test_preserves_total_mass(rng):
    """Gaussian blurring with reflecting boundaries preserves total mass.

    YvV is an IIR approximation, so mass is conserved only to a few parts in 1e4
    rather than to round-off.
    """
    image = rng.random((24, 26, 28))
    out = yvv_gaussian_filter_3d(image, sigma=2.5)
    np.testing.assert_allclose(out.sum(), image.sum(), rtol=1e-3)


def test_small_sigma_is_identity():
    """sigma < 0.5 returns the identity coefficients (B=1, others 0)."""
    rng = np.random.default_rng(1)
    image = rng.random((6, 7, 8))
    out = yvv_gaussian_filter_3d(image, sigma=0.1)
    np.testing.assert_allclose(out, image, atol=1e-12)


@pytest.mark.parametrize("sigma", [1.5, 2.5])
def test_matches_scipy_on_smooth_signal(sigma):
    """On a smooth signal in the interior of a large volume the YvV
    approximation should track scipy's truncated Gaussian closely.

    A pure-noise test would be unfair: YvV and FIR Gaussians have slightly
    different stopband shapes, so their high-frequency residuals don't match
    well even though both are valid low-pass filters with matching sigma.
    """
    N = 60
    x = np.arange(N) - N // 2
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    image = (
        np.exp(-(xx**2 + yy**2 + zz**2) / (2 * 5.0**2))
        + 0.5 * np.exp(-((xx - 8) ** 2 + (yy + 6) ** 2 + zz**2) / (2 * 3.0**2))
    ).astype(np.float64)

    out = yvv_gaussian_filter_3d(image, sigma=sigma)
    expected = ndi.gaussian_filter(image, sigma=sigma, mode="reflect")

    # Compare an interior crop to avoid boundary mismatch.
    crop = int(np.ceil(4 * sigma))
    sl = (slice(crop, -crop),) * 3
    diff = out[sl] - expected[sl]
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(expected[sl] - expected[sl].mean())
    assert rel_l2 < 0.05


def test_blurs_high_frequency_content(rng):
    """Filtering should reduce high-frequency energy, and more so for larger sigma."""
    image = rng.random((24, 24, 24))
    image_var = image.var()
    var_small = yvv_gaussian_filter_3d(image, sigma=1.0).var()
    var_large = yvv_gaussian_filter_3d(image, sigma=3.0).var()
    # Each pass attenuates high-frequency variance.
    assert var_small < image_var
    assert var_large < var_small


def test_impulse_response_is_gaussian_like():
    """The response to a delta function should peak at the impulse location and
    integrate to one — the hallmarks of a Gaussian kernel."""
    shape = (31, 31, 31)
    image = np.zeros(shape, dtype=np.float64)
    image[15, 15, 15] = 1.0
    sigma = 2.0
    out = yvv_gaussian_filter_3d(image, sigma=sigma)

    assert np.argmax(out) == np.ravel_multi_index((15, 15, 15), shape)
    # YvV preserves mass to ~1e-3 — much better than the typical scipy
    # gaussian_filter truncation but not exact.
    np.testing.assert_allclose(out.sum(), 1.0, rtol=2e-3)

    # Compare against the analytical Gaussian peak value 1 / (2*pi*sigma^2)^(3/2).
    # The 3rd-order YvV approximation gives a peak within ~10% of the analytical value.
    expected_peak = 1.0 / (2 * np.pi * sigma**2) ** 1.5
    assert abs(out[15, 15, 15] - expected_peak) / expected_peak < 0.1


def test_separable_axis_application(rng):
    """Filtering an x-only varying signal should match a 1D scipy filter along x."""
    sigma = 1.8
    x = np.linspace(-5, 5, 50)
    line = np.exp(-(x**2))
    image = np.broadcast_to(line[None, None, :], (8, 9, 50)).copy()
    out = yvv_gaussian_filter_3d(image, sigma=sigma)

    expected_line = ndi.gaussian_filter1d(line, sigma=sigma, mode="reflect")
    expected = np.broadcast_to(expected_line[None, None, :], (8, 9, 50))
    crop = int(np.ceil(4 * sigma))
    err = np.abs(out[:, :, crop:-crop] - expected[:, :, crop:-crop]).max()
    assert err < 0.05


def test_gaussian_alias_matches_underlying():
    rng = np.random.default_rng(3)
    image = rng.random((10, 11, 12))
    sigma = 1.5
    np.testing.assert_array_equal(gaussian(image, sigma), yvv_gaussian_filter_3d(image, sigma))


# ---------------------------------------------------------------------------
# Large-sigma regime (sigma=10)
#
# At sigma=10 the YvV burn-in is ~60 voxels, so the volume must be much larger
# than at smaller sigmas. Boundary effects also degrade mass conservation, so
# tolerances are looser than the small-sigma cases above.
# ---------------------------------------------------------------------------
def test_large_sigma_constant_input_preserved():
    """Constants must still be preserved exactly at large sigma — this exercises
    the burn-in logic on a volume just barely larger than 6*sigma."""
    image = np.full((64, 64, 64), 3.7, dtype=np.float64)
    out = yvv_gaussian_filter_3d(image, sigma=10.0)
    np.testing.assert_allclose(out, image, atol=1e-8)


def test_large_sigma_matches_scipy_on_smooth_signal():
    """Interior of a large volume should still track scipy's truncated Gaussian
    at sigma=10. Uses N=140 so a 3-sigma crop on each side leaves a healthy
    interior region."""
    sigma = 10.0
    N = 140
    x = np.arange(N) - N // 2
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    image = (
        np.exp(-(xx**2 + yy**2 + zz**2) / (2 * 15.0**2))
        + 0.5 * np.exp(-((xx - 20) ** 2 + (yy + 15) ** 2 + zz**2) / (2 * 8.0**2))
    ).astype(np.float64)

    out = yvv_gaussian_filter_3d(image, sigma=sigma)
    expected = ndi.gaussian_filter(image, sigma=sigma, mode="reflect")

    crop = int(np.ceil(3 * sigma))
    sl = (slice(crop, -crop),) * 3
    diff = out[sl] - expected[sl]
    rel_l2 = np.linalg.norm(diff) / np.linalg.norm(expected[sl] - expected[sl].mean())
    # Interior relative-L2 error is around 4% at sigma=10.
    assert rel_l2 < 0.08


def test_large_sigma_impulse_response():
    """Delta-function response at sigma=10: peak should land at the impulse and
    match the analytical Gaussian peak height. The volume must be wide enough
    that the kernel decays inside the boundary."""
    sigma = 10.0
    shape = (121, 121, 121)
    image = np.zeros(shape, dtype=np.float64)
    image[60, 60, 60] = 1.0
    out = yvv_gaussian_filter_3d(image, sigma=sigma)

    assert np.argmax(out) == np.ravel_multi_index((60, 60, 60), shape)
    # YvV mass on an impulse is within ~10% of unity at sigma=10.
    np.testing.assert_allclose(out.sum(), 1.0, rtol=0.1)

    expected_peak = 1.0 / (2 * np.pi * sigma**2) ** 1.5
    assert abs(out[60, 60, 60] - expected_peak) / expected_peak < 0.05


def test_large_sigma_preserves_compact_signal_mass():
    """A signal well inside a large enough volume should keep most of its mass
    after blurring at sigma=10. The reflecting boundary causes some leakage
    re-entry, so the tolerance is looser than the small-sigma case."""
    N = 120
    x = np.arange(N) - N // 2
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    image = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * 5.0**2)).astype(np.float64)
    out = yvv_gaussian_filter_3d(image, sigma=10.0)
    np.testing.assert_allclose(out.sum(), image.sum(), rtol=0.05)
