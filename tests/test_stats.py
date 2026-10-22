from glassbox.stats import mcnemar_exact, wilson_interval


def test_wilson_half_width_on_bird_dev_size():
    lo, hi = wilson_interval(767, 1534)          # 50% accuracy
    assert abs((hi - lo) / 2 - 0.025) < 0.001    # about +/-2.5 points


def test_wilson_edges():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 10) == 1.0
    assert mcnemar_exact(20, 0) < 1e-5
    assert abs(mcnemar_exact(30, 15) - mcnemar_exact(15, 30)) < 1e-12
    assert 0.03 < mcnemar_exact(30, 15) < 0.04
