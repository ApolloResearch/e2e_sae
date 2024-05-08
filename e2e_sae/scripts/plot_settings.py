import matplotlib.pyplot as plt

# Runs with constant CE loss increase for each layer. Values represent wandb run IDs.
SIMILAR_CE_RUNS = {
    2: {"local": "ue3lz0n7", "e2e": "ovhfts9n", "downstream": "visi12en"},
    6: {"local": "1jy3m5j0", "e2e": "zgdpkafo", "downstream": "2lzle2f0"},
    10: {"local": "m2hntlav", "e2e": "8crnit9h", "downstream": "cvj5um2h"},
}
SIMILAR_L0_RUNS = {
    2: {"local": "6vtk4k51", "e2e": "bst0prdd", "downstream": "e26jflpq"},
    6: {"local": "jup3glm9", "e2e": "tvj2owza", "downstream": "2lzle2f0"},
    10: {"local": "5vmpdgaz", "e2e": "8crnit9h", "downstream": "cvj5um2h"},
}

COLOR_MAP = {
    "local": "#f0a70a",
    "e2e": "#518c31",
    "downstream": plt.get_cmap("tab20b").colors[2],  # type: ignore[reportAttributeAccessIssue]
}

STYLE_MAP = {
    "local": {"marker": "^", "color": COLOR_MAP["local"], "label": "local"},
    "e2e": {"marker": "o", "color": COLOR_MAP["e2e"], "label": "e2e"},
    "downstream": {"marker": "X", "color": COLOR_MAP["downstream"], "label": "e2e+ds"},
}
