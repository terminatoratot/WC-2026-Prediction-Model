#!/usr/bin/env python3
"""V14: v13 live-signal model + individual-player data.

Player ratings (FIFA/SoFIFA national-squad snapshots) are turned into per-team,
per-edition line strengths (GK/DEF/MID/ATT, top-11, squad value) by
`build_player_features.py`. This module loads those tables and feeds them into the
prediction pipeline through the engine's `player_lambda_adjuster` hook, so a team's
attack vs the opponent's defence shifts expected goals **asymmetrically**.

Two channels (see model_diagram.md):
  * a data-fitted player-impact layer (a PoissonRegressor trained on historical
    international matches, EXCLUDING the Euro/Copa test tournaments) -- the
    "retrain" channel; and
  * the engine's existing current-strength correction, left intact.

A/B variants: `variant="squad"` aggregates the whole national squad;
`variant="lineup"` uses the FIFA-designated starting XI. `variant="none"` disables
the player layer (i.e. plain v13 -- the baseline).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import v13_live_signal_model as v13
import v11_wcq_results_model as v11

canon_team = v11.canon_team

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Extra aliases mapping FIFA nationality spellings to the engine's canonical form,
# layered on top of canon_team (which already covers Korea/Iran/Ivory Coast/Czechia).
_FIFA_ALIASES = {
    "china pr": "China",
    "china": "China",
}

# Scorer-perspective feature columns used by the player-impact layer.
_SCORER_FEATS = ["att_s", "mid_s", "top11_s", "def_o", "gk_o", "mid_o"]
# Engine FIFA editions we have player data for and their snapshot dates are read
# from the extract at load time.
EXCLUDED_TOURNAMENTS = {"UEFA Euro", "Copa América"}  # the held-out test set
LOG_MULT_CAP = 0.5  # cap |log multiplier| so the player layer can't dominate

# Per-variant shrinkage applied to the player layer's log-multiplier before it
# moves lambda. Diagnosis: the "squad" aggregate pools the whole national squad
# (~40 players) and is noisier than the FIFA starting XI, so its raw lambda moves
# push the argmax scoreline away from reality (worse individual_goal_diff) even
# though they help the win/draw/loss probabilities. Damping the squad channel
# keeps its probabilistic signal while shrinking its lambda-magnitude error.
# "lineup" stays at 1.0 (the XI is the cleaner signal -> left unchanged); "none"
# (baseline) never reaches this layer.
LOG_MULT_SHRINK = {"squad": 0.5, "lineup": 1.0}


def norm_team(name: object) -> str:
    base = canon_team(name)
    return _FIFA_ALIASES.get(str(base).lower(), base)


# --------------------------------------------------------------------------- #
# Player feature tables
# --------------------------------------------------------------------------- #
def load_player_tables(
    variant: str,
    data_dir: Path = DATA_DIR,
) -> Tuple[Dict[Tuple[str, int], Dict[str, float]], Dict[int, pd.Timestamp]]:
    """Return ((team, edition) -> feature dict) and (edition -> snapshot date)."""
    fname = {
        "squad": "squad_player_features.csv",
        "lineup": "lineup_player_features.csv",
    }[variant]
    feats = pd.read_csv(data_dir / fname)
    feats["team"] = feats["team"].map(norm_team)
    lookup: Dict[Tuple[str, int], Dict[str, float]] = {}
    for d in feats.to_dict(orient="records"):  # preserves the 'def' column name
        lookup[(d["team"], int(d["fifa_version"]))] = d

    extract = pd.read_csv(
        data_dir / "player_ratings_international.csv",
        usecols=["fifa_version", "fifa_update_date"],
    )
    extract["fifa_update_date"] = pd.to_datetime(extract["fifa_update_date"], errors="coerce")
    edition_dates = (
        extract.groupby("fifa_version")["fifa_update_date"].min().dropna().to_dict()
    )
    edition_dates = {int(k): pd.Timestamp(v) for k, v in edition_dates.items()}
    return lookup, edition_dates


def date_to_edition(date: Optional[pd.Timestamp], edition_dates: Dict[int, pd.Timestamp]) -> Optional[int]:
    """Latest FIFA edition whose snapshot pre-dates the match (point-in-time)."""
    if not edition_dates:
        return None
    editions = sorted(edition_dates)
    if date is None or pd.isna(date):
        return editions[-1]  # 2026 prediction: use the most recent player data
    date = pd.Timestamp(date)
    eligible = [e for e in editions if edition_dates[e] <= date]
    if not eligible:
        return None  # match predates all player data
    return eligible[-1]


# --------------------------------------------------------------------------- #
# Player-impact layer ("retrain" channel)
# --------------------------------------------------------------------------- #
@dataclass
class PlayerImpact:
    """A PoissonRegressor on player line-mismatch -> goals scored."""

    model: Any = None
    mean: float = 1.25
    std: np.ndarray = None
    center: np.ndarray = None
    baseline: float = 1.25
    n_train: int = 0

    @staticmethod
    def _scorer_row(s: Dict[str, float], o: Dict[str, float]) -> Dict[str, float]:
        return {
            "att_s": s["att"], "mid_s": s["mid"], "top11_s": s["top11_overall"],
            "def_o": o["def"], "gk_o": o["gk"], "mid_o": o["mid"],
        }

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PlayerImpact":
        from sklearn.linear_model import PoissonRegressor

        self.center = X.mean().values
        self.std = X.std(ddof=0).replace(0, 1.0).values
        Xs = (X.values - self.center) / self.std
        self.model = PoissonRegressor(alpha=1.0, max_iter=500).fit(Xs, y)
        self.baseline = float(np.mean(self.model.predict(Xs)))
        self.n_train = len(y)
        return self

    def predicted_goals(self, scorer: Dict[str, float], opp: Dict[str, float]) -> float:
        row = self._scorer_row(scorer, opp)
        x = np.array([[row[c] for c in _SCORER_FEATS]], dtype=float)
        xs = (x - self.center) / self.std
        return float(self.model.predict(xs)[0])

    def log_multiplier(
        self, scorer: Dict[str, float], opp: Dict[str, float], shrink: float = 1.0
    ) -> float:
        pp = self.predicted_goals(scorer, opp)
        lm = np.log(max(pp, 1e-6) / max(self.baseline, 1e-6))
        # Shrink toward 0 (no lambda move) before clipping, so a noisier variant
        # can be damped without changing the cap semantics.
        lm *= shrink
        return float(np.clip(lm, -LOG_MULT_CAP, LOG_MULT_CAP))


def fit_player_impact(
    lookup: Dict[Tuple[str, int], Dict[str, float]],
    edition_dates: Dict[int, pd.Timestamp],
    results_csv: Path,
) -> PlayerImpact:
    """Train the player-impact layer on internationals, excluding the test set."""
    res = pd.read_csv(results_csv)
    res["date"] = pd.to_datetime(res["date"], errors="coerce")
    res = res[res["date"].notna() & res["home_score"].notna() & res["away_score"].notna()]
    res = res[~res["tournament"].isin(EXCLUDED_TOURNAMENTS)]
    earliest = min(edition_dates.values()) if edition_dates else pd.Timestamp("2100-01-01")
    res = res[res["date"] >= earliest]

    rows, targets = [], []
    for r in res.itertuples(index=False):
        edition = date_to_edition(r.date, edition_dates)
        if edition is None:
            continue
        a, b = norm_team(r.home_team), norm_team(r.away_team)
        fa, fb = lookup.get((a, edition)), lookup.get((b, edition))
        if fa is None or fb is None:
            continue
        # team a scores
        rows.append(PlayerImpact._scorer_row(fa, fb)); targets.append(float(r.home_score))
        # team b scores
        rows.append(PlayerImpact._scorer_row(fb, fa)); targets.append(float(r.away_score))

    if len(targets) < 50:
        raise ValueError(f"Not enough player-covered matches to fit impact layer ({len(targets)//2}).")
    X = pd.DataFrame(rows)[_SCORER_FEATS]
    return PlayerImpact().fit(X, np.array(targets))


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class V14PlayerModel:
    def __init__(
        self,
        base_model: "v13.V13LiveSignalModel",
        variant: str,
        lookup: Dict[Tuple[str, int], Dict[str, float]],
        edition_dates: Dict[int, pd.Timestamp],
        impact: Optional[PlayerImpact],
    ):
        self.base_model = base_model          # V13LiveSignalModel
        self.engine = base_model.base_model    # v11 StrongWorldCupModel (has the hook)
        self.variant = variant
        self.lookup = lookup
        self.edition_dates = edition_dates
        self.impact = impact
        self._current_edition: Optional[int] = None
        if variant != "none" and impact is not None:
            self.engine.player_lambda_adjuster = self._adjuster
        else:
            self.engine.player_lambda_adjuster = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_model, name)

    # The hook the engine calls at the lambda stage.
    def _adjuster(self, lam_a: float, lam_b: float, team_a: str, team_b: str):
        edition = self._current_edition
        a, b = norm_team(team_a), norm_team(team_b)
        fa = self.lookup.get((a, edition)) if edition is not None else None
        fb = self.lookup.get((b, edition)) if edition is not None else None
        if fa is None or fb is None:
            return lam_a, lam_b, {"variant": self.variant, "edition": edition, "covered": False}
        shrink = LOG_MULT_SHRINK.get(self.variant, 1.0)
        lma = self.impact.log_multiplier(fa, fb, shrink=shrink)
        lmb = self.impact.log_multiplier(fb, fa, shrink=shrink)
        return (
            lam_a * float(np.exp(lma)),
            lam_b * float(np.exp(lmb)),
            {
                "variant": self.variant, "edition": edition, "covered": True,
                "log_mult_a": lma, "log_mult_b": lmb,
            },
        )

    def predict(
        self,
        team_a: str,
        team_b: str,
        host_a: bool = False,
        host_b: bool = False,
        knockout: bool = False,
        max_goals: int = 10,
        match_date: Optional[Any] = None,
        edition: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._current_edition = (
            edition if edition is not None
            else date_to_edition(pd.Timestamp(match_date) if match_date is not None else None,
                                 self.edition_dates)
        )
        # Re-install our adjuster so several variant wrappers can share one engine.
        self.engine.player_lambda_adjuster = (
            self._adjuster if (self.variant != "none" and self.impact is not None) else None
        )
        return self.base_model.predict(team_a, team_b, host_a, host_b, knockout, max_goals)


def build_from_zip(
    zip_path,
    variant: str = "squad",
    data_dir: Path = DATA_DIR,
    results_csv: Optional[str] = None,
    **kwargs,
) -> Tuple[V14PlayerModel, Any]:
    """Build a V14 model. Extra kwargs pass through to v13/v11 build_from_zip."""
    data_dir = Path(data_dir)
    # Mirror the data-file defaults that v11's main() supplies, so the base model
    # builds with the full qualifier/box/current inputs.
    defaults = {
        "train_csv": str(data_dir / "current_team_features_2026.csv"),
        "box_csv": str(data_dir / "FIFAallMatchBoxData.csv"),
        "former_names_csv": str(data_dir / "former_names.csv"),
    }
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    results_csv = results_csv or str(data_dir / "results.csv")
    base_model, data = v13.build_from_zip(zip_path, results_csv=results_csv, **kwargs)
    results_path = Path(results_csv)

    if variant == "none":
        return V14PlayerModel(base_model, "none", {}, {}, None), data

    lookup, edition_dates = load_player_tables(variant, Path(data_dir))
    impact = fit_player_impact(lookup, edition_dates, results_path)
    return V14PlayerModel(base_model, variant, lookup, edition_dates, impact), data


HOSTS_2026 = {"Canada", "Mexico", "USA", "United States"}


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Predict one match with the v14 player model and write the "
        "v11-style output suite (JSON, scoreline CSVs, and the plots/ folder)."
    )
    p.add_argument("--team-a", required=True)
    p.add_argument("--team-b", required=True)
    p.add_argument("--variant", choices=["squad", "lineup", "none"], default="lineup")
    p.add_argument("--match-date", default=None, help="YYYY-MM-DD; default = latest player data (FC 26)")
    p.add_argument("--zip", default=str(DATA_DIR / "worldcupsai.zip"))
    p.add_argument("--host-a", action="store_true", help="team A plays at home")
    p.add_argument("--host-b", action="store_true", help="team B plays at home")
    p.add_argument("--auto-host", action="store_true",
                   help="set home advantage automatically for 2026 hosts (Canada/Mexico/USA)")
    p.add_argument("--knockout", action="store_true")
    p.add_argument("--outdir", default=None,
                   help="output directory (default: outputs_v14_<a>_<b>)")
    p.add_argument("--no-plots", action="store_true", help="skip chart generation")
    args = p.parse_args()

    host_a = args.host_a or (args.auto_host and args.team_a in HOSTS_2026)
    host_b = args.host_b or (args.auto_host and args.team_b in HOSTS_2026)

    model, data = build_from_zip(args.zip, variant=args.variant)
    pred = model.predict(
        args.team_a, args.team_b,
        host_a=host_a, host_b=host_b,
        knockout=args.knockout, match_date=args.match_date,
    )

    default_dir = (
        f"outputs_v14_{args.team_a}_{args.team_b}".replace(" ", "_").lower()
    )
    out = v11.unique_output_dir(args.outdir or default_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "single_match_prediction.json").write_text(json.dumps(pred, indent=2))
    pd.DataFrame(pred["top_scorelines"]).to_csv(out / "scoreline_probabilities_top.csv", index=False)
    pd.DataFrame(pred["scoreline_probabilities"]).to_csv(out / "scoreline_probabilities.csv", index=False)

    plot_paths = []
    if not args.no_plots:
        # The v14 prediction dict has the same shape as v11's, so the engine's
        # plot suite (result probs, top scorelines, heatmap, spread, totals,
        # over/under, event expectations) renders directly.
        plot_paths = v11.plot_prediction_outputs(pred, out)

    print(f"\n{pred['team_a']} vs {pred['team_b']}  (variant={args.variant})")
    print("  result probs:", {k: round(v, 3) for k, v in pred["result_probabilities"].items()})
    print("  expected goals:", round(pred["lambda_a"], 2), "-", round(pred["lambda_b"], 2))
    print("  player adjustment:", pred.get("player_adjustment"))
    print("  top scoreline:", pred["top_scorelines"][0])
    print(f"\nWrote: {out}/single_match_prediction.json")
    print(f"Wrote: {out}/scoreline_probabilities_top.csv")
    print(f"Wrote: {out}/scoreline_probabilities.csv")
    for plot_path in plot_paths:
        print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    _cli()
