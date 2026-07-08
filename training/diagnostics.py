"""
Live training diagnostics.

DiagnosticsCallback regenerates PNG figures every `plot_freq` timesteps
(overwriting in place, so you can keep the files open and refresh):

  <out_dir>/dashboard.png     — loss curves + rollout stats from SB3's progress.csv
  <out_dir>/recent_games.png  — final boards of recently finished training episodes

Requires the model's logger to include a CSV writer (see self_play.train).
"""

import csv
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

# Palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"

# Board rendering
ATK_FILL = "#2b2b2b"
DEF_FILL = "#ffffff"
KING_FILL = "#eda100"
SPECIAL_SQUARE = "#e8e7e0"

DASHBOARD_PANELS = [
    ("rollout/ep_rew_mean", "Episode reward (mean)"),
    ("rollout/ep_len_mean", "Episode length (mean)"),
    ("train/value_loss", "Value loss"),
    ("train/policy_gradient_loss", "Policy gradient loss"),
    ("train/entropy_loss", "Entropy loss"),
    ("train/approx_kl", "Approx KL"),
    ("train/clip_fraction", "Clip fraction"),
    ("train/explained_variance", "Explained variance"),
]

X_KEY = "time/total_timesteps"


class DiagnosticsCallback(BaseCallback):
    """
    Every `plot_freq` timesteps, re-render diagnostic figures from the CSV
    log and the final boards of recently completed episodes.
    """

    def __init__(self, log_dir, out_dir, plot_freq: int = 2048, n_boards: int = 6, verbose: int = 0):
        super().__init__(verbose)
        self.progress_csv = Path(log_dir) / "progress.csv"
        self.out_dir = Path(out_dir)
        self.plot_freq = plot_freq
        self._last_plot = 0
        self._recent_games = deque(maxlen=n_boards)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "final_board" in info:
                self._recent_games.append({
                    "board": info["final_board"],
                    "winner": info.get("winner"),
                    "reason": info.get("reason", "?"),
                    "step": self.num_timesteps,
                })
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_plot < self.plot_freq:
            return
        self._last_plot = self.num_timesteps
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._plot_dashboard()
            self._plot_recent_games()
            if self.verbose:
                print(f"[Diagnostics] Figures updated at step {self.num_timesteps} -> {self.out_dir}")
        except Exception as e:  # never kill training over a plotting hiccup
            print(f"[Diagnostics] Plotting failed at step {self.num_timesteps}: {e}")

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _read_progress(self) -> dict:
        """Read progress.csv into {column: np.ndarray}, keeping rows aligned with X_KEY."""
        if not self.progress_csv.exists():
            return {}
        with open(self.progress_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        cols = {}
        keys = [X_KEY] + [k for k, _ in DASHBOARD_PANELS]
        for key in keys:
            vals = []
            for row in rows:
                v = row.get(key, "")
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            cols[key] = np.array(vals)
        return cols

    def _plot_dashboard(self) -> None:
        cols = self._read_progress()
        if not cols or np.all(np.isnan(cols.get(X_KEY, np.array([np.nan])))):
            return
        x = cols[X_KEY]

        fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor=SURFACE)
        for ax, (key, title) in zip(axes.flat, DASHBOARD_PANELS):
            ax.set_facecolor(SURFACE)
            y = cols[key]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.any():
                ax.plot(x[ok], y[ok], color=SERIES, linewidth=2)
            else:
                ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes,
                        ha="center", va="center", color=MUTED, fontsize=10)
            ax.set_title(title, color=INK_SECONDARY, fontsize=11, loc="left")
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(BASELINE)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 4))
            ax.xaxis.get_offset_text().set_color(MUTED)
            ax.xaxis.get_offset_text().set_fontsize(8)

        fig.suptitle(f"Training diagnostics — {self.num_timesteps:,} timesteps",
                     color=INK, fontsize=13, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.out_dir / "dashboard.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Recent game boards
    # ------------------------------------------------------------------

    def _plot_recent_games(self) -> None:
        if not self._recent_games:
            return
        from gym_tafl.envs.configs import ATK, DEF, DRAW, ATTACKER, DEFENDER, KING, THRONE, CORNER

        games = list(self._recent_games)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8.5), facecolor=SURFACE)
        winner_name = {ATK: "ATK wins", DEF: "DEF wins", DRAW: "Draw"}

        for ax, game in zip(axes.flat, games):
            board = game["board"]
            n = board.shape[0]
            ax.set_facecolor(SURFACE)
            for (r, c) in zip(*np.where((board == THRONE) | (board == CORNER))):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=SPECIAL_SQUARE, zorder=0))
            atk_r, atk_c = np.where(board == ATTACKER)
            def_r, def_c = np.where(board == DEFENDER)
            king_r, king_c = np.where(board == KING)
            ax.scatter(atk_c, atk_r, s=150, c=ATK_FILL, edgecolors=INK, linewidths=0.5, zorder=2)
            ax.scatter(def_c, def_r, s=150, c=DEF_FILL, edgecolors=INK, linewidths=1.2, zorder=2)
            ax.scatter(king_c, king_r, s=210, c=KING_FILL, edgecolors=INK, linewidths=1.2, zorder=2)
            for r, c in zip(king_r, king_c):
                ax.text(c, r, "K", ha="center", va="center", color=INK,
                        fontsize=8, fontweight="bold", zorder=3)

            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5)  # row 0 on top
            ax.set_xticks(np.arange(-0.5, n, 1), minor=False)
            ax.set_yticks(np.arange(-0.5, n, 1), minor=False)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.tick_params(labelbottom=False, labelleft=False, length=0)
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_color(BASELINE)
            title = f"{winner_name.get(game['winner'], 'Unfinished')} — {game['reason']}"
            ax.set_title(f"{title}\n(seen at step {game['step']:,})",
                         color=INK_SECONDARY, fontsize=10)

        for ax in axes.flat[len(games):]:
            ax.axis("off")

        fig.suptitle("Final boards — recent training episodes",
                     color=INK, fontsize=13, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.out_dir / "recent_games.png", dpi=110, facecolor=SURFACE)
        plt.close(fig)
