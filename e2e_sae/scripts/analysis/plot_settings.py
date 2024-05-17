import matplotlib.pyplot as plt

# Runs with constant CE loss increase for each layer. Values represent wandb run IDs.
SIMILAR_CE_RUNS = {
    2: {"local": "ue3lz0n7", "e2e": "ovhfts9n", "downstream": "visi12en"},
    6: {"local": "1jy3m5j0", "e2e": "zgdpkafo", "downstream": "2lzle2f0"},
    10: {"local": "m2hntlav", "e2e": "8crnit9h", "downstream": "cvj5um2h"},
}
# Runs with similar L0 loss increase for each layer. Values represent wandb run IDs.
SIMILAR_L0_RUNS = {
    2: {"local": "6vtk4k51", "e2e": "bst0prdd", "downstream": "e26jflpq"},
    6: {"local": "jup3glm9", "e2e": "tvj2owza", "downstream": "2lzle2f0"},
    10: {"local": "5vmpdgaz", "e2e": "8crnit9h", "downstream": "cvj5um2h"},
}
# Runs with similar alive dictionary elements. Values represent wandb run IDs.
SIMILAR_ALIVE_ELEMENTS_RUNS = {
    2: {"local": "6vtk4k51", "e2e": "0z98g9pf", "downstream": "visi12en"},
    6: {"local": "h9hrelni", "e2e": "tvj2owza", "downstream": "p9zmh62k"},
    10: {"local": "5vmpdgaz", "e2e": "vnfh4vpi", "downstream": "f2fs7hk3"},
}

SIMILAR_RUN_INFO = {
    "CE": SIMILAR_CE_RUNS,
    "l0": SIMILAR_L0_RUNS,
    "alive_elements": SIMILAR_ALIVE_ELEMENTS_RUNS,
}

STYLE_MAP = {
    "local": {"marker": "^", "color": "#f0a70a", "label": "local"},
    "e2e": {"marker": "o", "color": "#518c31", "label": "e2e"},
    "downstream": {"marker": "X", "color": plt.get_cmap("tab20b").colors[2], "label": "e2e+ds"},  # type: ignore[reportAttributeAccessIssue]
}
