"""Cloud Phase RGB (Section 2.5): red 1.6 um over 0-0.4, green 2.25 um over 0-0.4, blue 0.6 um over 0-0.8,
linear, gamma 1."""
import numpy as np


def cloud_phase_rgb(vis16, nir22, vis06):
    rgb = np.dstack([np.clip(vis16 / 0.4, 0, 1), np.clip(nir22 / 0.4, 0, 1), np.clip(vis06 / 0.8, 0, 1)])
    rgb[~np.isfinite(rgb)] = 0.0
    return rgb


def save_png(rgb, path, crop_to_data=True, dpi=300, width_in=6.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if crop_to_data:
        rr, cc = np.where(rgb.sum(axis=2) > 0)
        if rr.size:
            rgb = rgb[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(width_in, width_in * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path, dpi=dpi, facecolor="black")
    plt.close(fig)
