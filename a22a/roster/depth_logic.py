"""Phase 12 — roster depth construction and replacement planning."""

from __future__ import annotations

import pathlib
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd
import yaml

from a22a.units import uer

CONFIG_PATH = pathlib.Path("configs/defaults.yaml")
ARTIFACT_DIR = pathlib.Path("artifacts/roster")


@dataclass(frozen=True)
class DepthConfig:
    min_snaps_for_starter: int = 200
    fallback_by_position: Dict[str, List[str]] | None = None
    qb_replacement_penalty: float = 0.7
    seed: int = 0

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "DepthConfig":
        raw = dict(data or {})
        seed = int(raw.get("seed", 0))
        roster_cfg = raw.get("roster", {}) or {}
        fallback_map = {
            str(pos).upper(): list(values)
            for pos, values in (roster_cfg.get("fallback_by_position", {}) or {}).items()
        }
        return cls(
            min_snaps_for_starter=int(roster_cfg.get("min_snaps_for_starter", 200)),
            fallback_by_position=fallback_map,
            qb_replacement_penalty=float(roster_cfg.get("qb_replacement_penalty", 0.7)),
            seed=seed,
        )


def _load_config(path: pathlib.Path = CONFIG_PATH) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _staged_root(cfg: Mapping[str, Any]) -> pathlib.Path:
    paths = cfg.get("paths", {}) if isinstance(cfg, Mapping) else {}
    return pathlib.Path(paths.get("staged", "./data/staged"))


# ---------------------------------------------------------------------------
# Sample depth chart scaffolding (used when staged inputs absent)
# ---------------------------------------------------------------------------


def _sample_depth_inputs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = [
        # Team BUF
        {"team_id": "BUF", "position": "QB", "player_id": "BUF_QB1", "snaps": 920, "recent_snaps": 64, "depth_chart_order": 1, "is_active": True},
        {"team_id": "BUF", "position": "QB", "player_id": "BUF_QB2", "snaps": 180, "recent_snaps": 24, "depth_chart_order": 2, "is_active": True},
        {"team_id": "BUF", "position": "QB", "player_id": "BUF_QB3", "snaps": 60, "recent_snaps": 6, "depth_chart_order": 3, "is_active": False},
        {"team_id": "BUF", "position": "RB", "player_id": "BUF_RB1", "snaps": 540, "recent_snaps": 42, "depth_chart_order": 1, "is_active": True},
        {"team_id": "BUF", "position": "RB", "player_id": "BUF_RB2", "snaps": 260, "recent_snaps": 28, "depth_chart_order": 2, "is_active": True},
        {"team_id": "BUF", "position": "RB", "player_id": "BUF_RB3", "snaps": 110, "recent_snaps": 12, "depth_chart_order": 3, "is_active": False},
        {"team_id": "BUF", "position": "WR", "player_id": "BUF_WR1", "snaps": 710, "recent_snaps": 48, "depth_chart_order": 1, "is_active": False},
        {"team_id": "BUF", "position": "WR", "player_id": "BUF_WR2", "snaps": 520, "recent_snaps": 52, "depth_chart_order": 2, "is_active": True},
        {"team_id": "BUF", "position": "WR", "player_id": "BUF_WR3", "snaps": 320, "recent_snaps": 35, "depth_chart_order": 3, "is_active": True},
        {"team_id": "BUF", "position": "WR", "player_id": "BUF_WR4", "snaps": 210, "recent_snaps": 28, "depth_chart_order": 4, "is_active": True},
        {"team_id": "BUF", "position": "TE", "player_id": "BUF_TE1", "snaps": 430, "recent_snaps": 36, "depth_chart_order": 1, "is_active": True},
        {"team_id": "BUF", "position": "TE", "player_id": "BUF_TE2", "snaps": 240, "recent_snaps": 22, "depth_chart_order": 2, "is_active": True},
        {"team_id": "BUF", "position": "TE", "player_id": "BUF_TE3", "snaps": 120, "recent_snaps": 14, "depth_chart_order": 3, "is_active": False},
        # Team KC
        {"team_id": "KC", "position": "QB", "player_id": "KC_QB1", "snaps": 950, "recent_snaps": 65, "depth_chart_order": 1, "is_active": True},
        {"team_id": "KC", "position": "QB", "player_id": "KC_QB2", "snaps": 210, "recent_snaps": 30, "depth_chart_order": 2, "is_active": True},
        {"team_id": "KC", "position": "RB", "player_id": "KC_RB1", "snaps": 560, "recent_snaps": 46, "depth_chart_order": 1, "is_active": True},
        {"team_id": "KC", "position": "RB", "player_id": "KC_RB2", "snaps": 240, "recent_snaps": 18, "depth_chart_order": 2, "is_active": False},
        {"team_id": "KC", "position": "RB", "player_id": "KC_RB3", "snaps": 150, "recent_snaps": 20, "depth_chart_order": 3, "is_active": True},
        {"team_id": "KC", "position": "WR", "player_id": "KC_WR1", "snaps": 690, "recent_snaps": 54, "depth_chart_order": 1, "is_active": True},
        {"team_id": "KC", "position": "WR", "player_id": "KC_WR2", "snaps": 470, "recent_snaps": 38, "depth_chart_order": 2, "is_active": True},
        {"team_id": "KC", "position": "WR", "player_id": "KC_WR3", "snaps": 280, "recent_snaps": 26, "depth_chart_order": 3, "is_active": True},
        {"team_id": "KC", "position": "WR", "player_id": "KC_WR4", "snaps": 190, "recent_snaps": 24, "depth_chart_order": 4, "is_active": True},
        {"team_id": "KC", "position": "TE", "player_id": "KC_TE1", "snaps": 560, "recent_snaps": 48, "depth_chart_order": 1, "is_active": True},
        {"team_id": "KC", "position": "TE", "player_id": "KC_TE2", "snaps": 220, "recent_snaps": 20, "depth_chart_order": 2, "is_active": False},
        {"team_id": "KC", "position": "TE", "player_id": "KC_TE3", "snaps": 140, "recent_snaps": 18, "depth_chart_order": 3, "is_active": True},
    ]
    for entry in base:
        entry.update({"season": 2023, "week": 3})
        rows.append(entry)
    return pd.DataFrame(rows)


def _list_parquet(root: pathlib.Path, folder: str) -> list[pathlib.Path]:
    target = root / folder
    if not target.exists():
        return []
    return sorted(target.rglob("*.parquet"))


def _load_depth_inputs(cfg: Mapping[str, Any]) -> pd.DataFrame:
    staged_root = _staged_root(cfg)
    candidates = _list_parquet(staged_root, "depth_charts")
    frames: list[pd.DataFrame] = []
    for path in candidates:
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if not combined.empty:
            return combined
    return _sample_depth_inputs()


# ---------------------------------------------------------------------------
# Depth ordering helpers
# ---------------------------------------------------------------------------


def _role_from_index(idx: int) -> str:
    if idx == 0:
        return "starter"
    if idx == 1:
        return "primary_backup"
    if idx == 2:
        return "secondary_backup"
    return f"depth_{idx + 1}"


def _assign_depth_roles(df: pd.DataFrame, config: DepthConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (team, position), group in df.groupby(["team_id", "position"], sort=False):
        ordered_idx = (
            group.sort_values(["depth_chart_order", "snaps"], ascending=[True, False]).index.tolist()
        )
        starter_candidates = group[group["snaps"] >= config.min_snaps_for_starter]
        if starter_candidates.empty:
            starter_index = ordered_idx[0]
        else:
            starter_index = starter_candidates.sort_values("snaps", ascending=False).index[0]
        if starter_index in ordered_idx:
            ordered_idx.remove(starter_index)
        ordered_idx.insert(0, starter_index)
        ordered_group = group.loc[ordered_idx].copy()
        ordered_group["depth_order"] = range(1, len(ordered_group) + 1)
        ordered_group["depth_role"] = ordered_group["depth_order"].apply(lambda i: _role_from_index(i - 1))
        starter_player_id = ordered_group.iloc[0]["player_id"]
        starter_active = bool(ordered_group.iloc[0]["is_active"])
        active_candidates = ordered_group[ordered_group["is_active"]]
        effective_starter_id: Optional[str]
        effective_starter_id = active_candidates.iloc[0]["player_id"] if not active_candidates.empty else None
        ordered_group["starter_player_id"] = starter_player_id
        ordered_group["starter_is_active"] = starter_active
        ordered_group["effective_starter_id"] = effective_starter_id
        ordered_group["effective_starter_is_backup"] = effective_starter_id not in (None, starter_player_id)
        ordered_group["lineup_id"] = f"{team}_{position}"
        rows.append(ordered_group)
    combined = pd.concat(rows, ignore_index=True)
    combined.sort_values(["team_id", "position", "depth_order"], inplace=True)
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Replacement logic helpers
# ---------------------------------------------------------------------------


def _parse_fallback_tag(tag: str) -> tuple[str, Optional[int]]:
    tag = str(tag).upper()
    match = re.match(r"([A-Z]+)(\d+)?", tag)
    if not match:
        return tag, None
    pos = match.group(1)
    depth = match.group(2)
    return pos, int(depth) if depth else None


def _team_strength_from_uer(teams: Iterable[str]) -> Dict[str, float]:
    table = uer.run()
    if table.is_empty():
        return {team: 1.0 for team in teams}
    df = pd.DataFrame(table.to_dicts())
    if df.empty:
        return {team: 1.0 for team in teams}
    split = df["unit_id"].str.rsplit("_", n=2, expand=True)
    df["uer_team"] = split[0]
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    available_teams = sorted(df["uer_team"].unique())
    requested = sorted(set(teams))
    mapping = {team: available_teams[i % len(available_teams)] for i, team in enumerate(requested)}
    strength: Dict[str, float] = {}
    for team in requested:
        mapped = mapping[team]
        subset = df[df["uer_team"] == mapped]
        if subset.empty or not mean_cols:
            strength[team] = 1.0
            continue
        base = float(subset[mean_cols].mean().mean())
        strength[team] = 1.0 + base
    for team in teams:
        strength.setdefault(team, 1.0)
    return strength


def _find_fallback_candidate(
    depth_table: pd.DataFrame,
    team: str,
    current_player: str,
    fallback_tag: str,
) -> Optional[pd.Series]:
    position, depth = _parse_fallback_tag(fallback_tag)
    candidates = depth_table[(depth_table["team_id"] == team) & (depth_table["position"] == position)]
    if candidates.empty:
        return None
    candidates = candidates.sort_values("depth_order")
    for _, row in candidates.iterrows():
        if row["player_id"] == current_player:
            continue
        if depth is not None and row["depth_order"] < depth:
            continue
        if row["is_active"]:
            return row
    if depth is None:
        for _, row in candidates.iterrows():
            if row["player_id"] == current_player:
                continue
            if row["is_active"]:
                return row
    return None


def _apply_replacement_logic(df: pd.DataFrame, config: DepthConfig) -> pd.DataFrame:
    depth_df = df.copy()
    team_strength = _team_strength_from_uer(depth_df["team_id"].unique())

    depth_df["starter_uer"] = depth_df["team_id"].map(team_strength).fillna(1.0)
    player_effective_map: Dict[int, float] = {}
    qb_penalty_map: Dict[int, bool] = {}
    for _, group in depth_df.groupby(["team_id", "position"], sort=False):
        starter_snaps = float(max(group.iloc[0]["snaps"], 1.0))
        for idx, row in group.iterrows():
            snap_share = float(row["snaps"]) / starter_snaps if starter_snaps else 0.0
            snap_share = min(max(snap_share, 0.0), 1.0)
            applied_penalty = False
            if row["depth_role"] != "starter" and row["position"] == "QB":
                snap_share *= config.qb_replacement_penalty
                applied_penalty = True
            if row["depth_role"] == "starter":
                snap_share = 1.0
            player_effective_map[idx] = snap_share * row["starter_uer"]
            qb_penalty_map[idx] = applied_penalty
    depth_df["player_effective_uer"] = (
        depth_df.index.to_series().map(player_effective_map).fillna(0.0).astype(float)
    )
    depth_df["qb_penalty_applied"] = (
        depth_df.index.to_series().map(qb_penalty_map).fillna(False).astype(bool)
    )
    depth_df["uer_fraction"] = depth_df["player_effective_uer"] / depth_df["starter_uer"].replace(0, np.nan)
    depth_df["uer_fraction"] = depth_df["uer_fraction"].fillna(0.0)

    player_uer_lookup = depth_df.set_index("player_id")["player_effective_uer"].to_dict()

    replacement_ids: list[Optional[str]] = []
    replacement_types: list[Optional[str]] = []
    replacement_roles: list[Optional[str]] = []
    replacement_positions: list[Optional[str]] = []
    replacement_is_active: list[Optional[bool]] = []
    replacement_effective: list[float] = []

    fallback_cfg = config.fallback_by_position or {}

    for idx, row in depth_df.iterrows():
        group_mask = (depth_df["team_id"] == row["team_id"]) & (depth_df["position"] == row["position"])
        group = depth_df[group_mask].sort_values("depth_order")
        internal_candidate = group[group["depth_order"] > row["depth_order"]]
        internal_candidate = internal_candidate[internal_candidate["is_active"]]
        chosen: Optional[pd.Series] = None
        chosen_type: Optional[str] = None
        if not internal_candidate.empty:
            chosen = internal_candidate.iloc[0]
            chosen_type = "depth"
        else:
            fallback_tags = fallback_cfg.get(row["position"], [])
            for tag in fallback_tags:
                candidate = _find_fallback_candidate(depth_df, row["team_id"], row["player_id"], tag)
                if candidate is not None:
                    chosen = candidate
                    chosen_type = "fallback"
                    break
        if chosen is not None:
            replacement_ids.append(str(chosen["player_id"]))
            replacement_types.append(chosen_type)
            replacement_roles.append(str(chosen.get("depth_role", None)))
            replacement_positions.append(str(chosen.get("position", None)))
            replacement_is_active.append(bool(chosen.get("is_active", True)))
            replacement_effective.append(float(player_uer_lookup.get(chosen["player_id"], 0.0)))
        else:
            replacement_ids.append(None)
            replacement_types.append(None)
            replacement_roles.append(None)
            replacement_positions.append(None)
            replacement_is_active.append(None)
            replacement_effective.append(0.0)

    depth_df["replacement_player_id"] = replacement_ids
    depth_df["replacement_type"] = replacement_types
    depth_df["replacement_depth_role"] = replacement_roles
    depth_df["replacement_position"] = replacement_positions
    depth_df["replacement_is_active"] = replacement_is_active
    depth_df["replacement_effective_uer"] = replacement_effective

    depth_df["season"] = depth_df.get("season", 2023)
    depth_df["week"] = depth_df.get("week", depth_df["week"].max() if "week" in depth_df else 1)
    depth_df["season"] = depth_df["season"].astype(int)
    depth_df["week"] = depth_df["week"].astype(int)
    depth_df["recent_snaps"] = depth_df.get("recent_snaps", depth_df["snaps"])
    depth_df["depth_chart_order"] = depth_df.get("depth_chart_order", depth_df["depth_order"])
    columns = [
        "season",
        "week",
        "team_id",
        "position",
        "lineup_id",
        "player_id",
        "depth_order",
        "depth_role",
        "depth_chart_order",
        "snaps",
        "recent_snaps",
        "is_active",
        "starter_player_id",
        "starter_is_active",
        "effective_starter_id",
        "effective_starter_is_backup",
        "starter_uer",
        "player_effective_uer",
        "uer_fraction",
        "qb_penalty_applied",
        "replacement_player_id",
        "replacement_type",
        "replacement_position",
        "replacement_depth_role",
        "replacement_is_active",
        "replacement_effective_uer",
    ]
    depth_df = depth_df[columns]
    depth_df.sort_values(["team_id", "position", "depth_order"], inplace=True)
    depth_df.reset_index(drop=True, inplace=True)
    return depth_df


def _build_lineups(config: DepthConfig, cfg_map: Mapping[str, Any]) -> pd.DataFrame:
    raw_depth = _load_depth_inputs(cfg_map)
    if raw_depth.empty:
        raise ValueError("depth chart input is empty")
    assigned = _assign_depth_roles(raw_depth, config)
    enriched = _apply_replacement_logic(assigned, config)
    return enriched


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()
    cfg_map = _load_config()
    config = DepthConfig.from_config(cfg_map)

    lineups = _build_lineups(config, cfg_map)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = ARTIFACT_DIR / f"lineups_{stamp}.parquet"
    try:
        lineups.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.with_suffix(".csv")
        lineups.to_csv(out_path, index=False)

    duration = time.time() - start
    qb_pen = config.qb_replacement_penalty
    print(
        "[depth] wrote %s with %d rows (starter_threshold=%d, qb_penalty=%.2f) in %.2fs"
        % (out_path.name, len(lineups), config.min_snaps_for_starter, qb_pen, duration)
    )


if __name__ == "__main__":
    main()
