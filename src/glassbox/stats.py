"""The two statistics every comparison in this project uses."""

from __future__ import annotations

from math import comb, sqrt


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for an accuracy of correct/n (Wilson score interval).

    With n = 1534 and accuracy near 50%, the interval is about +/-2.5 points, so two
    runs that differ by less than that cannot be told apart from a single number each.
    """
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided p-value for "A and B are equally accurate" on the same questions.

    Only questions where exactly one run is right carry information. Under the null
    hypothesis each of those is a fair coin flip, so the count follows Binomial(k, 0.5).
    """
    k = only_a + only_b
    if k == 0:
        return 1.0
    tail = sum(comb(k, i) for i in range(min(only_a, only_b) + 1)) / 2**k
    return min(1.0, 2 * tail)
