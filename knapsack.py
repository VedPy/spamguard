def knapsack_01(values, weights, capacity):
    """
    Select whole items to maximize total value
    without exceeding the available capacity.
    """

    if len(values) != len(weights):
        raise ValueError("Values and weights must have equal lengths.")

    if not isinstance(capacity, int) or isinstance(capacity, bool):
        raise ValueError("Capacity must be an integer.")

    if capacity < 0:
        raise ValueError("Capacity cannot be negative.")

    if any(
        not isinstance(v, int) or isinstance(v, bool) or v < 0
        for v in values
    ):
        raise ValueError("Values must be non-negative integers.")

    if any(
        not isinstance(w, int) or isinstance(w, bool) or w <= 0
        for w in weights
    ):
        raise ValueError("Weights must be positive integers.")

    n = len(values)

    # dp[i][c] = maximum value using the first i items
    # with capacity c.
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        value = values[i - 1]
        weight = weights[i - 1]

        for c in range(capacity + 1):
            # Option 1: exclude this item.
            dp[i][c] = dp[i - 1][c]

            # Option 2: include this item if it fits.
            if weight <= c:
                included_value = (
                    value + dp[i - 1][c - weight]
                )
                dp[i][c] = max(
                    dp[i][c],
                    included_value
                )

    # Recover the selected item indices.
    selected = []
    c = capacity

    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected.append(i - 1)
            c -= weights[i - 1]

    selected.reverse()

    return dp[n][capacity], selected
