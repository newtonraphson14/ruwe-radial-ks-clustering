#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_RENAME_MAP = {
    "RA_ICRS": "ra",
    "DE_ICRS": "dec",
    "Plx": "parallax",
    "e_Plx": "parallax_error",
    "pmRA": "pmra",
    "e_pmRA": "pmra_error",
    "pmDE": "pmdec",
    "e_pmDE": "pmdec_error",
    "Gmag": "phot_g_mean_mag",
    "BP-RP": "bp_rp",
    "RUWE": "ruwe",
}


def infer_cluster_name(path_str: str) -> str:
    name = Path(path_str).stem
    name = name.replace("_Members_Final", "")
    name = name.replace("_", " ")
    return name


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def ensure_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_or_show(fig: plt.Figure, output_path: Optional[Path], show: bool, dpi: int = 300) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def load_members_catalog(path_str: str) -> pd.DataFrame:
    path = resolve_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Members catalog not found: {path}")
    return pd.read_csv(path)


def load_membership_input(path_str: str, sep: str, comment: str) -> pd.DataFrame:
    path = resolve_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")

    df = pd.read_csv(path, sep=sep, comment=comment, on_bad_lines="skip")
    df = df.rename(columns=DEFAULT_RENAME_MAP)
    df = coerce_numeric(
        df,
        (
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "parallax",
            "parallax_error",
            "pmra_error",
            "pmdec_error",
            "ruwe",
            "phot_g_mean_mag",
            "bp_rp",
        ),
    )

    required = (
        "ra",
        "dec",
        "pmra",
        "pmdec",
        "parallax",
        "parallax_error",
        "pmra_error",
        "pmdec_error",
    )
    ensure_columns(df, required, "membership inference")
    return df.dropna(subset=list(required)).copy()


def plot_initial_checks(
    df: pd.DataFrame,
    cluster_name: str,
    pmra_ref: Optional[float],
    pmdec_ref: Optional[float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    output_path: Optional[Path],
    show: bool,
) -> None:
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(df["ra"], df["dec"], s=1, alpha=0.1, color="black")
    ax1.set_title(f"{cluster_name}: initial sky distribution (N={len(df)})")
    ax1.set_xlabel("RA (deg)")
    ax1.set_ylabel("Dec (deg)")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(df["pmra"], df["pmdec"], s=1, alpha=0.1, color="black")
    ax2.set_title("Vector point diagram")
    ax2.set_xlabel("pmRA (mas/yr)")
    ax2.set_ylabel("pmDec (mas/yr)")
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.grid(True, alpha=0.3)

    if pmra_ref is not None:
        ax2.axvline(pmra_ref, linestyle="--", linewidth=0.8, color="red")
    if pmdec_ref is not None:
        ax2.axhline(pmdec_ref, linestyle="--", linewidth=0.8, color="red")

    fig.tight_layout()
    save_or_show(fig, output_path, show)


def plot_membership_results(
    df: pd.DataFrame,
    cluster_name: str,
    threshold: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    scatter_output: Optional[Path],
    hist_output: Optional[Path],
    show: bool,
) -> None:
    members = df[df["prob"] >= threshold]
    field = df[df["prob"] < threshold]

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(field["ra"], field["dec"], s=1, alpha=0.1, color="gray", label="Field")
    ax1.scatter(members["ra"], members["dec"], s=5, alpha=0.8, color="red", label="Members")
    ax1.set_title(f"{cluster_name}: membership spatial distribution")
    ax1.set_xlabel("RA (deg)")
    ax1.set_ylabel("Dec (deg)")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", frameon=True)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(field["pmra"], field["pmdec"], s=1, alpha=0.1, color="gray")
    ax2.scatter(members["pmra"], members["pmdec"], s=5, alpha=0.6, color="red")
    ax2.set_title("Membership in proper motion space")
    ax2.set_xlabel("pmRA (mas/yr)")
    ax2.set_ylabel("pmDec (mas/yr)")
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_or_show(fig, scatter_output, show)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df["prob"], bins=20, edgecolor="black", color="skyblue")
    ax.set_title("Membership probability distribution")
    ax.set_xlabel("Membership probability")
    ax.set_ylabel("Count")
    fig.tight_layout()
    save_or_show(fig, hist_output, show)


def run_membership(args: argparse.Namespace) -> None:
    cluster_name = args.cluster_name or infer_cluster_name(args.input_path)
    cluster_slug = slugify(cluster_name)
    figure_dir = resolve_path(args.figure_dir) if args.figure_dir else None

    df = load_membership_input(args.input_path, args.sep, args.comment)
    print(f"Loaded and cleaned input rows: {len(df)}")

    if figure_dir is not None or args.show:
        initial_output = figure_dir / f"{cluster_slug}_initial_checks.png" if figure_dir else None
        plot_initial_checks(
            df=df,
            cluster_name=cluster_name,
            pmra_ref=args.target_pmra,
            pmdec_ref=args.target_pmdec,
            xlim=(args.vpd_xlim_min, args.vpd_xlim_max),
            ylim=(args.vpd_ylim_min, args.vpd_ylim_max),
            output_path=initial_output,
            show=args.show,
        )

    n_rows = len(df)
    votes = np.zeros(n_rows, dtype=float)
    feature_cols = ["ra", "dec", "parallax", "pmra", "pmdec"]
    rng = np.random.default_rng(args.seed)

    for iteration in range(args.iterations):
        noisy = df[feature_cols].copy()
        noisy["parallax"] += df["parallax_error"].to_numpy() * rng.standard_normal(n_rows)
        noisy["pmra"] += df["pmra_error"].to_numpy() * rng.standard_normal(n_rows)
        noisy["pmdec"] += df["pmdec_error"].to_numpy() * rng.standard_normal(n_rows)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(noisy)
        kmeans = KMeans(
            n_clusters=args.k_clusters,
            n_init=10,
            random_state=None if args.seed is None else args.seed + iteration,
        )
        labels = kmeans.fit_predict(scaled)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)

        dist_sq = (
            (centers[:, 3] - args.target_pmra) ** 2
            + (centers[:, 4] - args.target_pmdec) ** 2
            + args.parallax_weight * (centers[:, 2] - args.target_parallax) ** 2
        )
        best_cluster = int(np.argmin(dist_sq))
        votes[labels == best_cluster] += 1.0

        if (iteration + 1) % max(1, args.iterations // 5) == 0:
            print(f"Iteration {iteration + 1}/{args.iterations} completed.")

    df = df.copy()
    df["prob"] = votes / float(args.iterations)

    members = df[df["prob"] >= args.membership_threshold].copy()
    field = df[df["prob"] < args.membership_threshold].copy()

    output_members = resolve_path(args.output_members)
    output_members.parent.mkdir(parents=True, exist_ok=True)
    members.to_csv(output_members, index=False)
    print(f"Saved members catalog: {output_members}")

    if args.probability_output:
        probability_output = resolve_path(args.probability_output)
        probability_output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(probability_output, index=False)
        print(f"Saved probability catalog: {probability_output}")

    print("Membership summary")
    print(f"  Total rows processed: {len(df)}")
    print(f"  Members (P >= {args.membership_threshold:.2f}): {len(members)}")
    print(f"  Field  (P <  {args.membership_threshold:.2f}): {len(field)}")

    if figure_dir is not None or args.show:
        scatter_output = figure_dir / f"{cluster_slug}_membership.png" if figure_dir else None
        hist_output = figure_dir / f"{cluster_slug}_probabilities.png" if figure_dir else None
        plot_membership_results(
            df=df,
            cluster_name=cluster_name,
            threshold=args.membership_threshold,
            xlim=(args.vpd_xlim_min, args.vpd_xlim_max),
            ylim=(args.vpd_ylim_min, args.vpd_ylim_max),
            scatter_output=scatter_output,
            hist_output=hist_output,
            show=args.show,
        )


def prepare_cmd(df: pd.DataFrame, mag_min: Optional[float], mag_max: Optional[float], distance_modulus: Optional[float]) -> pd.DataFrame:
    df = coerce_numeric(df, ("bp_rp", "phot_g_mean_mag"))
    ensure_columns(df, ("bp_rp", "phot_g_mean_mag"), "CMD plotting")

    cmd = df.dropna(subset=["bp_rp", "phot_g_mean_mag"]).copy()
    if mag_min is not None:
        cmd = cmd[cmd["phot_g_mean_mag"] >= mag_min].copy()
    if mag_max is not None:
        cmd = cmd[cmd["phot_g_mean_mag"] <= mag_max].copy()

    if distance_modulus is not None:
        cmd["plot_mag"] = cmd["phot_g_mean_mag"] - float(distance_modulus)
        ylabel = "Absolute Magnitude (G)"
    else:
        cmd["plot_mag"] = cmd["phot_g_mean_mag"]
        ylabel = "Apparent Magnitude (G)"

    cmd.attrs["ylabel"] = ylabel
    return cmd


def run_cmd(args: argparse.Namespace) -> None:
    cluster_name = args.cluster_name or infer_cluster_name(args.members_csv)
    df = load_members_catalog(args.members_csv)
    cmd = prepare_cmd(df, args.mag_min, args.mag_max, args.distance_modulus)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(cmd["bp_rp"], cmd["plot_mag"], s=args.marker_size, alpha=args.alpha, color="black", label="Members")
    ax.invert_yaxis()
    ax.set_xlabel("Color (G_BP - G_RP)")
    ax.set_ylabel(cmd.attrs["ylabel"])
    ax.set_title(f"{cluster_name}: color-magnitude diagram")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    output = resolve_path(args.output_figure) if args.output_figure else None
    save_or_show(fig, output, args.show)
    print(f"CMD-ready rows: {len(cmd)}")


def run_ruwe_cmd(args: argparse.Namespace) -> None:
    cluster_name = args.cluster_name or infer_cluster_name(args.members_csv)
    df = load_members_catalog(args.members_csv)
    df = coerce_numeric(df, ("bp_rp", "phot_g_mean_mag", "ruwe"))
    ensure_columns(df, ("bp_rp", "phot_g_mean_mag", "ruwe"), "RUWE CMD plotting")

    cmd = df.dropna(subset=["bp_rp", "phot_g_mean_mag", "ruwe"]).copy()
    if args.mag_min is not None:
        cmd = cmd[cmd["phot_g_mean_mag"] >= args.mag_min].copy()
    if args.mag_max is not None:
        cmd = cmd[cmd["phot_g_mean_mag"] <= args.mag_max].copy()

    if args.distance_modulus is not None:
        cmd["plot_mag"] = cmd["phot_g_mean_mag"] - float(args.distance_modulus)
        ylabel = "Absolute Magnitude (G)"
    else:
        cmd["plot_mag"] = cmd["phot_g_mean_mag"]
        ylabel = "Apparent Magnitude (G)"

    cmd = cmd.sort_values("ruwe", ascending=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    scatter = ax.scatter(
        cmd["bp_rp"],
        cmd["plot_mag"],
        s=args.marker_size,
        c=cmd["ruwe"],
        cmap=args.colormap,
        vmin=args.ruwe_vmin,
        vmax=args.ruwe_vmax,
        alpha=args.alpha,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Color (G_BP - G_RP)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{cluster_name}: CMD colored by RUWE")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Gaia RUWE")

    fig.tight_layout()
    output = resolve_path(args.output_figure) if args.output_figure else None
    save_or_show(fig, output, args.show)
    print(f"RUWE CMD rows: {len(cmd)}")


def add_projected_radius(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_numeric(df, ("ra", "dec", "ruwe"))
    ensure_columns(df, ("ra", "dec", "ruwe"), "radial KS test")

    radial = df.dropna(subset=["ra", "dec", "ruwe"]).copy()
    center_ra = float(radial["ra"].mean())
    center_dec = float(radial["dec"].mean())

    delta_ra = (radial["ra"].to_numpy() - center_ra) * np.cos(np.radians(center_dec))
    delta_dec = radial["dec"].to_numpy() - center_dec
    radius_deg = np.sqrt(delta_ra**2 + delta_dec**2)

    radial["radius_deg"] = radius_deg
    radial["radius_arcmin"] = radius_deg * 60.0
    radial.attrs["center_ra"] = center_ra
    radial.attrs["center_dec"] = center_dec
    return radial


def run_radial_ks(args: argparse.Namespace) -> None:
    cluster_name = args.cluster_name or infer_cluster_name(args.members_csv)
    df = load_members_catalog(args.members_csv)
    radial = add_projected_radius(df)

    single = radial[radial["ruwe"] < args.ruwe_single_max].copy()
    binary = radial[radial["ruwe"] > args.ruwe_binary_min].copy()

    if len(single) == 0 or len(binary) == 0:
        stat = np.nan
        p_value = np.nan
    else:
        stat, p_value = ks_2samp(single["radius_arcmin"], binary["radius_arcmin"])

    print("Sample sizes")
    print(f"  Total members: {len(radial)}")
    print(f"  RUWE-low  (< {args.ruwe_single_max:.2f}): {len(single)}")
    print(f"  RUWE-high (> {args.ruwe_binary_min:.2f}): {len(binary)}")
    print("Cluster center")
    print(f"  RA0  = {radial.attrs['center_ra']:.6f} deg")
    print(f"  Dec0 = {radial.attrs['center_dec']:.6f} deg")
    print("KS test")
    if np.isnan(p_value):
        print("  Not computed because one subsample is empty.")
    else:
        print(f"  KS statistic = {stat:.4f}")
        print(f"  p-value      = {p_value:.2e}")

    if args.summary_output:
        summary = pd.DataFrame(
            [
                {
                    "cluster_name": cluster_name,
                    "center_ra_deg": radial.attrs["center_ra"],
                    "center_dec_deg": radial.attrs["center_dec"],
                    "n_total": len(radial),
                    "n_ruwe_low": len(single),
                    "n_ruwe_high": len(binary),
                    "ks_statistic": stat,
                    "ks_p_value": p_value,
                }
            ]
        )
        summary_path = resolve_path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary table: {summary_path}")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(
        single["radius_arcmin"],
        bins=args.bins,
        density=True,
        cumulative=True,
        histtype="step",
        linewidth=2,
        color="blue",
        label=f"RUWE < {args.ruwe_single_max:.2f}",
    )
    ax.hist(
        binary["radius_arcmin"],
        bins=args.bins,
        density=True,
        cumulative=True,
        histtype="step",
        linewidth=2,
        color="red",
        label=f"RUWE > {args.ruwe_binary_min:.2f}",
    )
    title_p = "NA" if np.isnan(p_value) else f"{p_value:.2e}"
    ax.set_title(f"{cluster_name}: radial CDF comparison (KS p-value = {title_p})")
    ax.set_xlabel("Projected distance from center (arcmin)")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    if args.xlim_arcmin is not None:
        ax.set_xlim(0, args.xlim_arcmin)

    fig.tight_layout()
    output = resolve_path(args.output_figure) if args.output_figure else None
    save_or_show(fig, output, args.show)


def run_grid(args: argparse.Namespace) -> None:
    config_path = resolve_path(args.config_csv)
    config = pd.read_csv(config_path)
    ensure_columns(config, ("cluster_name", "members_csv", "xlim_arcmin"), "multicluster grid config")

    figure, axes = plt.subplots(nrows=2, ncols=len(config), figsize=(5 * len(config), 10), constrained_layout=True)
    if len(config) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, row in config.reset_index(drop=True).iterrows():
        cluster_name = str(row["cluster_name"])
        members_path = resolve_path(str(row["members_csv"]), base_dir=config_path.parent)
        xlim_arcmin = float(row["xlim_arcmin"])

        df = load_members_catalog(str(members_path))

        cmd_df = coerce_numeric(df, ("bp_rp", "phot_g_mean_mag", "ruwe"))
        cmd_df = cmd_df.dropna(subset=["bp_rp", "phot_g_mean_mag", "ruwe"]).copy()
        cmd_df = cmd_df.sort_values("ruwe", ascending=True)

        ax_cmd = axes[0, idx]
        scatter = ax_cmd.scatter(
            cmd_df["bp_rp"],
            cmd_df["phot_g_mean_mag"],
            s=args.marker_size,
            c=cmd_df["ruwe"],
            cmap=args.colormap,
            vmin=args.ruwe_vmin,
            vmax=args.ruwe_vmax,
            alpha=args.alpha,
        )
        ax_cmd.invert_yaxis()
        ax_cmd.set_title(cluster_name)
        ax_cmd.set_xlabel("Color (G_BP - G_RP)")
        ax_cmd.set_ylabel("App. Magnitude (G)" if idx == 0 else "")
        ax_cmd.grid(True, alpha=0.3)

        cbar = figure.colorbar(scatter, ax=ax_cmd, pad=0.02)
        cbar.set_label("RUWE", fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        radial = add_projected_radius(df)
        single = radial[radial["ruwe"] < args.ruwe_single_max].copy()
        binary = radial[radial["ruwe"] > args.ruwe_binary_min].copy()
        if len(single) == 0 or len(binary) == 0:
            p_value = np.nan
        else:
            _, p_value = ks_2samp(single["radius_arcmin"], binary["radius_arcmin"])

        ax_cdf = axes[1, idx]
        ax_cdf.hist(
            single["radius_arcmin"],
            bins=args.bins,
            density=True,
            cumulative=True,
            histtype="step",
            linewidth=2,
            color="blue",
            label="RUWE-low",
        )
        ax_cdf.hist(
            binary["radius_arcmin"],
            bins=args.bins,
            density=True,
            cumulative=True,
            histtype="step",
            linewidth=2,
            color="red",
            label="RUWE-high",
        )
        title_p = "NA" if np.isnan(p_value) else f"{p_value:.2e}"
        ax_cdf.set_title(f"KS p-value: {title_p}")
        ax_cdf.set_xlabel("Distance (arcmin)")
        ax_cdf.set_ylabel("Cumulative fraction" if idx == 0 else "")
        ax_cdf.set_xlim(0, xlim_arcmin)
        ax_cdf.grid(True, alpha=0.3)
        if idx == 0:
            ax_cdf.legend(loc="lower right", fontsize=8)

    output = resolve_path(args.output_figure) if args.output_figure else None
    save_or_show(figure, output, args.show)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for RUWE-based cluster membership, CMD plotting, and radial KS analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    membership = subparsers.add_parser("membership", help="Infer cluster membership with Monte Carlo + KMeans voting.")
    membership.add_argument("--input-path", required=True, help="Path to the raw VizieR-like TSV/CSV file.")
    membership.add_argument("--output-members", required=True, help="Path to save the final members CSV.")
    membership.add_argument("--cluster-name", default=None, help="Display name used in plots and logs.")
    membership.add_argument("--probability-output", default=None, help="Optional path for the full catalog with probabilities.")
    membership.add_argument("--sep", default=";", help="Delimiter used by the input table.")
    membership.add_argument("--comment", default="#", help="Comment prefix used by the input table.")
    membership.add_argument("--iterations", type=int, default=50, help="Number of Monte Carlo voting iterations.")
    membership.add_argument("--k-clusters", type=int, default=10, help="Number of KMeans clusters per iteration.")
    membership.add_argument("--membership-threshold", type=float, default=0.50, help="Probability threshold for final members.")
    membership.add_argument("--target-pmra", type=float, required=True, help="Reference literature pmRA value.")
    membership.add_argument("--target-pmdec", type=float, required=True, help="Reference literature pmDec value.")
    membership.add_argument("--target-parallax", type=float, required=True, help="Reference literature parallax value.")
    membership.add_argument("--parallax-weight", type=float, default=10.0, help="Weight applied to parallax in the target-distance metric.")
    membership.add_argument("--vpd-xlim-min", type=float, default=-25.0)
    membership.add_argument("--vpd-xlim-max", type=float, default=5.0)
    membership.add_argument("--vpd-ylim-min", type=float, default=-15.0)
    membership.add_argument("--vpd-ylim-max", type=float, default=5.0)
    membership.add_argument("--figure-dir", default=None, help="Optional directory for sanity-check figures.")
    membership.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")
    membership.add_argument("--show", action="store_true", help="Display plots interactively in addition to saving them.")
    membership.set_defaults(func=run_membership)

    cmd = subparsers.add_parser("cmd", help="Plot a standard color-magnitude diagram from a members catalog.")
    cmd.add_argument("--members-csv", required=True, help="Path to the final members CSV.")
    cmd.add_argument("--cluster-name", default=None)
    cmd.add_argument("--output-figure", default=None, help="Optional path for the output figure.")
    cmd.add_argument("--distance-modulus", type=float, default=None, help="Optional fixed distance modulus for absolute magnitudes.")
    cmd.add_argument("--mag-min", type=float, default=None)
    cmd.add_argument("--mag-max", type=float, default=None)
    cmd.add_argument("--marker-size", type=float, default=2.0)
    cmd.add_argument("--alpha", type=float, default=0.6)
    cmd.add_argument("--show", action="store_true")
    cmd.set_defaults(func=run_cmd)

    ruwe_cmd = subparsers.add_parser("ruwe-cmd", help="Plot a RUWE-colored color-magnitude diagram.")
    ruwe_cmd.add_argument("--members-csv", required=True, help="Path to the final members CSV.")
    ruwe_cmd.add_argument("--cluster-name", default=None)
    ruwe_cmd.add_argument("--output-figure", default=None)
    ruwe_cmd.add_argument("--distance-modulus", type=float, default=None)
    ruwe_cmd.add_argument("--mag-min", type=float, default=None)
    ruwe_cmd.add_argument("--mag-max", type=float, default=None)
    ruwe_cmd.add_argument("--marker-size", type=float, default=3.0)
    ruwe_cmd.add_argument("--alpha", type=float, default=0.8)
    ruwe_cmd.add_argument("--colormap", default="turbo")
    ruwe_cmd.add_argument("--ruwe-vmin", type=float, default=0.8)
    ruwe_cmd.add_argument("--ruwe-vmax", type=float, default=1.4)
    ruwe_cmd.add_argument("--show", action="store_true")
    ruwe_cmd.set_defaults(func=run_ruwe_cmd)

    radial = subparsers.add_parser("radial-ks", help="Run the RUWE-low vs RUWE-high radial KS test.")
    radial.add_argument("--members-csv", required=True, help="Path to the final members CSV.")
    radial.add_argument("--cluster-name", default=None)
    radial.add_argument("--output-figure", default=None)
    radial.add_argument("--summary-output", default=None, help="Optional CSV file for the one-row KS summary.")
    radial.add_argument("--ruwe-single-max", type=float, default=1.1)
    radial.add_argument("--ruwe-binary-min", type=float, default=1.2)
    radial.add_argument("--xlim-arcmin", type=float, default=60.0)
    radial.add_argument("--bins", type=int, default=1000)
    radial.add_argument("--show", action="store_true")
    radial.set_defaults(func=run_radial_ks)

    grid = subparsers.add_parser("grid", help="Build the multi-cluster CMD + KS summary figure.")
    grid.add_argument("--config-csv", default="configs/multicluster_grid.csv", help="CSV config listing cluster names, members CSV files, and x-axis limits.")
    grid.add_argument("--output-figure", default="data/figures/generated/evolution_grid_generated.png", help="Output path for the grid figure.")
    grid.add_argument("--ruwe-single-max", type=float, default=1.1)
    grid.add_argument("--ruwe-binary-min", type=float, default=1.2)
    grid.add_argument("--bins", type=int, default=1000)
    grid.add_argument("--marker-size", type=float, default=2.0)
    grid.add_argument("--alpha", type=float, default=0.8)
    grid.add_argument("--colormap", default="turbo")
    grid.add_argument("--ruwe-vmin", type=float, default=0.8)
    grid.add_argument("--ruwe-vmax", type=float, default=1.4)
    grid.add_argument("--show", action="store_true")
    grid.set_defaults(func=run_grid)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

