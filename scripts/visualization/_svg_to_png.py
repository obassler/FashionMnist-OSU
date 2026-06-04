import os
import shutil

OUT = "../outputs"
THESIS = "../../Thesis/images"

pairs = [
    ("per_class_disagreement.svg", "per_class_disagreement.png"),
    ("barrier_heatmap.svg", "barrier_heatmap.png"),
]


def try_cairosvg():
    try:
        import cairosvg
        for src, dst in pairs:
            cairosvg.svg2png(url=os.path.join(OUT, src),
                             write_to=os.path.join(THESIS, dst),
                             output_width=2400)
            print(f"cairosvg: {src} -> {dst}")
        return True
    except Exception as e:
        print(f"cairosvg failed: {e}")
        return False


def try_matplotlib_replot():
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt
    orig_savefig = plt.Figure.savefig

    def png_savefig(self, fname, *a, **kw):
        kw["format"] = "png"
        kw["dpi"] = 200
        return orig_savefig(self, str(fname).replace(".svg", ".png"), *a, **kw)
    plt.Figure.savefig = png_savefig

    import sys
    sys.path.insert(0, ".")
    from per_class import plot_per_class
    from barrier_heatmap import plot_barrier_heatmap

    d = np.load(os.path.join(OUT, "per_class_data.npz"), allow_pickle=True)
    out_pc = os.path.join(THESIS, "per_class_disagreement.png")
    plot_per_class(d["confusion"], d["disagreement"], out_pc, n_models=100)
    print(f"replot: per_class -> {out_pc}")

    d = np.load(os.path.join(OUT, "barrier_matrix_data.npz"), allow_pickle=True)
    out_bh = os.path.join(THESIS, "barrier_heatmap.png")
    plot_barrier_heatmap(d["barriers"], d["endpoint_loss"], d["names"].tolist(), out_bh)
    print(f"replot: barrier -> {out_bh}")


if not try_cairosvg():
    try_matplotlib_replot()
