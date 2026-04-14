"""
Figure generators for the paper
"Kernel Embeddings and the Separation of Measure Phenomenon".

Figures 1a–2b are schematic / illustrative plots (Plotly + Matplotlib).
Figure 3 is the Monte Carlo power comparison (KLR vs MMD).
"""

import numpy as np
from pathlib import Path
from scipy.stats import multivariate_normal
import plotly.graph_objects as go
from pypdf import PdfReader, PdfWriter
import seaborn as sns
import matplotlib.pyplot as plt

from .models import AR1_model
from .stats import MC_test_vals
from .plot import res_plot

# Output directory (relative to the repo root, i.e. one level above code/)
_FIG_DIR = Path(__file__).resolve().parents[2] / "fig"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _discretize_colors(values, n_levels=10):
    bins = np.linspace(values.min(), values.max(), n_levels)
    return np.digitize(values, bins) / n_levels


def _make_gaussian_ellipsoid(center, cov, tilt_axis="x", tilt_angle_deg=0,
                              resolution=100, color_levels=10):
    """3D ellipsoid whose cross-section matches a 2D Gaussian, with optional tilt."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes_lengths = np.sqrt(eigvals) * 2

    theta = np.linspace(0, 2 * np.pi, resolution)
    r     = np.linspace(0, 1, resolution // 2)
    theta, r = np.meshgrid(theta, r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    pos2d   = np.stack([x, y], axis=-1)
    shaped  = pos2d @ np.diag(axes_lengths) @ eigvecs.T
    x_, y_  = shaped[..., 0], shaped[..., 1]
    z_      = np.zeros_like(x_)

    angle_rad = np.deg2rad(tilt_angle_deg)
    if tilt_axis == "x":
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle_rad), -np.sin(angle_rad)],
                      [0, np.sin(angle_rad),  np.cos(angle_rad)]])
    elif tilt_axis == "y":
        R = np.array([[ np.cos(angle_rad), 0, np.sin(angle_rad)],
                      [0, 1, 0],
                      [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
    else:
        raise ValueError("tilt_axis must be 'x' or 'y'")

    pts          = np.stack([x_, y_, z_], axis=-1)
    pts_rotated  = pts @ R.T
    x_f = pts_rotated[..., 0] + center[0]
    y_f = pts_rotated[..., 1] + center[1]
    z_f = pts_rotated[..., 2] + center[2]

    pts_unrotated = pts_rotated @ R
    xy_unrotated  = pts_unrotated[..., :2] + center[:2]
    mvn       = multivariate_normal(mean=center[:2], cov=cov)
    pdf_vals  = mvn.pdf(xy_unrotated)
    pdf_vals /= pdf_vals.max()
    colors    = _discretize_colors(pdf_vals ** 3, color_levels)
    return x_f, y_f, z_f, colors


def _trim_pdf(filename, left=50, right=50, top=125, bottom=75):
    """Crop a single-page PDF in-place."""
    reader = PdfReader(filename)
    page   = reader.pages[0]
    page.mediabox.lower_left  = (page.mediabox.left + left,
                                  page.mediabox.bottom + bottom)
    page.mediabox.upper_right = (page.mediabox.right - right,
                                  page.mediabox.top - top)
    writer = PdfWriter()
    writer.add_page(page)
    writer.write(filename)


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------

def figure_1a():
    width = height = 1000

    mu1  = np.array([1, 0])
    x    = np.linspace(-7, 7, 100)
    y    = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    pos  = np.dstack((X, Y))
    cov1 = np.array([[3.0, 0.0], [0.0, 0.0021]])
    Z1   = multivariate_normal(mu1, cov1).pdf(pos)
    _Z1  = np.where(Z1 >= 1e-2, Z1, -10)
    c1   = np.abs(X - 1) / 3; c1 -= c1.min(); c1 /= c1.max(); c1 = (1 - c1) ** 2

    x2   = np.linspace(-1, 1, 20)
    y2   = np.linspace(-7, 7, 100)
    X2, Y2 = np.meshgrid(x2, y2)
    pos2 = np.dstack((X2, Y2))
    cov2 = np.array([[0.0025, 0.0], [0.0, 2.5]])
    Z2   = multivariate_normal(mu1, cov2).pdf(pos2)
    _Z2  = np.where(Z2 >= 1e-2, Z2, -10)
    c2   = np.abs(Y2 / 2.5); c2 = (c2.max() - c2) ** 2

    fig = go.Figure(data=[
        go.Surface(x=X - 1, y=Y, z=_Z1, surfacecolor=c1,
                   colorscale="Blues", opacity=0.7, showscale=False),
        go.Surface(x=np.zeros_like(x2) - 0.5, y=Y2, z=_Z2, surfacecolor=c2,
                   colorscale="Reds", opacity=1, showscale=False),
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, range=[-6, 7]),
            yaxis=dict(showticklabels=False, range=[-5, 6]),
            zaxis=dict(showticklabels=False, range=[0, 2.5]),
            camera=dict(eye=dict(x=0.9, y=1.7, z=0.35),
                        center=dict(x=-0.05, y=0, z=-0.15)),
            aspectmode="data",
        ),
        width=width, height=height, showlegend=False,
    )
    filename = str(_FIG_DIR / "Figure 1a.pdf")
    fig.write_image(filename, width=width, height=height, scale=2)
    _trim_pdf(filename)


def figure_1b():
    width = height = 1000
    mu1  = np.array([1, 0])
    x    = np.linspace(-7, 7, 100)
    y    = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    cov1 = np.array([[3.0, 0.0], [0.0, 0.0025]])
    Z1   = multivariate_normal(mu1, cov1).pdf(np.dstack((X, Y)))
    _Z1  = np.where(Z1 >= 1e-2, Z1, -10)
    c1   = np.abs(X - 1) ** 0.6

    X2, Y2 = np.meshgrid(x, np.linspace(-7, 7, 100))
    cov2   = np.array([[0.5, 0.0], [0.0, 0.5]])
    Z2     = multivariate_normal(mu1, cov2).pdf(np.dstack((X2, Y2)))
    _Z2    = Z2 - 2e-8
    c2     = np.sqrt((X2 - 1) ** 2 + Y2 ** 2) * 0.25
    c2    -= c2.min(); c2 /= c2.max(); c2 = (1 - c2) ** 5

    fig = go.Figure(data=[
        go.Surface(x=X, y=Y, z=_Z1, surfacecolor=c1,
                   colorscale="Blues_r", opacity=0.7, showscale=False),
        go.Surface(x=X2, y=Y2, z=_Z2, surfacecolor=c2,
                   colorscale="Reds", opacity=1, showscale=False),
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, range=[-6, 7]),
            yaxis=dict(showticklabels=False, range=[-5, 6]),
            zaxis=dict(showticklabels=False, range=[0, 1.5]),
            camera=dict(eye=dict(x=1.3, y=1.5, z=0.35),
                        center=dict(x=0, y=0, z=-0.1)),
            aspectmode="data",
        ),
        width=width, height=height, showlegend=False,
    )
    filename = str(_FIG_DIR / "Figure 1b.pdf")
    fig.write_image(filename, width=width, height=height, scale=2)
    _trim_pdf(filename)


def figure_1c():
    width = height = 1000
    mu1  = np.array([-0.5, 0]);   cov1 = np.array([[0.5,  0.2], [0.2,  1.0]])
    mu2  = np.array([ 0.5, -0.5]); cov2 = np.array([[0.5, -0.3], [-0.3, 1.2]])

    x1, y1, z1, c1 = _make_gaussian_ellipsoid(
        np.append(mu1, 0.0), cov1, tilt_axis="x", tilt_angle_deg=-45, color_levels=20)
    x2, y2, z2, c2 = _make_gaussian_ellipsoid(
        np.append(mu2, 0.0), cov2, tilt_axis="x", tilt_angle_deg=35,  color_levels=20)

    COLOR_SCALE = 0.01
    fig = go.Figure(data=[
        go.Surface(x=x1, y=y1, z=z1, surfacecolor=c1 ** COLOR_SCALE,
                   colorscale="Reds",  showscale=False, opacity=1),
        go.Surface(x=x2, y=y2, z=z2, surfacecolor=c2 ** COLOR_SCALE,
                   colorscale="Blues", showscale=False, opacity=0.9),
    ])
    fig.update_layout(
        width=width, height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=True, range=[-3, 2]),
            yaxis=dict(showticklabels=False, showgrid=True, range=[-3, 2]),
            zaxis=dict(showticklabels=False, showgrid=True, range=[-4, 4]),
            aspectmode="data",
            camera=dict(eye=dict(x=2.5, y=1, z=1)),
        ),
    )
    filename = str(_FIG_DIR / "Figure 1c.pdf")
    fig.write_image(filename, width=width, height=height, scale=2)
    _trim_pdf(filename, left=50, right=150, top=250, bottom=50)


def figure_2a():
    np.random.seed(42)
    size    = 1500
    theta   = np.random.uniform(0, 2 * np.pi, size)
    r_ring  = np.random.normal(5, 0.3, size)
    x_ring  = r_ring * np.cos(theta)
    y_ring  = r_ring * np.sin(theta)
    x_red   = np.random.normal(0, 1, size)
    y_red   = np.random.normal(0, 1, size)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
    })
    plt.figure(figsize=(8, 8))
    sns.kdeplot(x=x_red,  y=y_red,  cmap="Reds",  fill=True, alpha=1,   levels=6, thresh=0.05)
    sns.kdeplot(x=x_ring, y=y_ring, cmap="Blues", fill=True, alpha=0.6, levels=6, thresh=0.1)
    plt.axis("equal")
    plt.gca().set_axis_off()
    plt.savefig(str(_FIG_DIR / "Figure 2a.pdf"), format="pdf",
                bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.close()


def figure_2b():
    width = height = 1000
    mu1  = np.array([1, 0])

    x    = np.linspace(-7, 7, 100)
    y    = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    cov1 = np.array([[3.0, 0.0], [0.0, 0.002]])
    Z1   = multivariate_normal(mu1, cov1).pdf(np.dstack((X, Y)))
    _Z1  = np.where(Z1 >= 1e-2, Z1, -10)
    c1   = np.abs(X - 1) / 3; c1 -= c1.min(); c1 /= c1.max(); c1 = (1 - c1) ** 2

    x2   = np.linspace(-1, 1, 20)
    y2   = np.linspace(-7, 7, 100)
    X2, Y2 = np.meshgrid(x2, y2)
    cov2 = np.array([[0.0075, 0.0], [0.0, 2.5]])
    Z2   = multivariate_normal(mu1, cov2).pdf(np.dstack((X2, Y2))) * 1.5
    _Z2  = np.where(Z2 >= 1e-2, Z2, -10)
    c2   = np.abs(Y2 / 2.5); c2 = (c2.max() - c2) ** 2

    fig = go.Figure(data=[
        go.Surface(x=X - 1, y=Y, z=_Z1, surfacecolor=c1,
                   colorscale="Blues", opacity=0.7, showscale=False, name="N_Q"),
        go.Surface(x=np.zeros_like(x2) - 0.5, y=Y2, z=_Z2, surfacecolor=c2,
                   colorscale="Reds",  opacity=1,   showscale=False, name="N_P"),
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, range=[-6, 6], visible=False),
            yaxis=dict(showticklabels=False, range=[-6, 6], visible=False),
            zaxis=dict(showticklabels=False, range=[0, 2]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.15)),
            aspectmode="data",
        ),
        width=width, height=height, showlegend=True,
    )
    filename = str(_FIG_DIR / "Figure 2b.pdf")
    fig.write_image(filename, width=width, height=height, scale=2)
    _trim_pdf(filename, left=100, right=100, top=500, bottom=200)


def figure_3():
    """
    Figure 3: Monte Carlo power comparison of KLR vs MMD
    under the AR1 (decreasing correlation) model.
    """
    model = AR1_model(alpha=0.5, eps=0.25)(d=50)
    Ns    = [75, 125, 175, 225]
    K     = 100
    np.random.seed(0)
    data = MC_test_vals(model, Ns, K, kernel_type="euclidean")
    fig  = res_plot(data)
    out  = str(_FIG_DIR / "Figure 3.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
