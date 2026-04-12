"""Horodatage des runs CLI, fraîcheur des résultats, légende pour le notebook."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MANIFEST_FILENAME = "run_manifest.json"


def _parse_iso_utc(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def discover_manifest_runs(repo_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Retourne [(dossier, manifest), ...] pour chaque run-* contenant un manifest terminé."""
    repo_root = repo_root.resolve()
    out: list[tuple[Path, dict[str, Any]]] = []
    if not repo_root.is_dir():
        return out
    for p in sorted(repo_root.glob("run-*")):
        if not p.is_dir():
            continue
        mf = p / MANIFEST_FILENAME
        if not mf.is_file():
            continue
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not data.get("finished_at_utc"):
            continue
        out.append((p, data))
    return out


def _run_matches_source(manifest: dict[str, Any], data_source_filter: str | None) -> bool:
    return manifest.get("data_source") == data_source_filter


def runs_for_source(repo_root: Path, data_source_filter: str | None) -> list[tuple[Path, dict[str, Any]]]:
    """Filtre les runs terminés pour une source de données donnée (None = simulation)."""
    return [
        (p, m)
        for p, m in discover_manifest_runs(repo_root)
        if _run_matches_source(m, data_source_filter)
    ]


def latest_finished_manifest(
    repo_root: Path,
    data_source_filter: str | None = None,
) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    if data_source_filter is None:
        runs = discover_manifest_runs(repo_root)
    else:
        runs = runs_for_source(repo_root, data_source_filter)
    if not runs:
        return None, None
    best = max(runs, key=lambda x: _parse_iso_utc(x[1]["finished_at_utc"]))
    return best[0], best[1]


def age_seconds_since_finished(manifest: dict[str, Any]) -> float:
    fin = manifest.get("finished_at_utc")
    if not fin:
        return float("inf")
    end = _parse_iso_utc(str(fin))
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - end).total_seconds()


def _run_cli_pipeline(repo_root: Path, data_source: str | None) -> None:
    prev = os.getcwd()
    try:
        os.chdir(repo_root)
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        code_dir = repo_root / "code"
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        import firewall_svm as fw  # noqa: WPS433

        fw.main(data_source=data_source)
    finally:
        os.chdir(prev)


def ensure_fresh_cli_pipeline(
    repo_root: Path,
    *,
    max_age_seconds: float,
    data_source: str | None,
) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    """Compat legacy: garantit 1 run frais pour la source demandée."""
    latest_path, manifest, _ = ensure_minimum_fresh_runs(
        repo_root,
        min_runs=1,
        max_age_seconds=max_age_seconds,
        data_source=data_source,
    )
    return latest_path, manifest


def ensure_minimum_fresh_runs(
    repo_root: Path,
    *,
    min_runs: int,
    max_age_seconds: float,
    data_source: str | None,
) -> tuple[Optional[Path], Optional[dict[str, Any]], int]:
    """
    Vérifie qu'il existe au moins `min_runs` runs terminés pour `data_source`.
    Si non, relance la pipeline autant de fois que nécessaire.
    Puis garantit que le plus récent est plus jeune que `max_age_seconds`.

    Retourne (latest_path, latest_manifest, count_for_source).
    """
    repo_root = repo_root.resolve()

    runs = runs_for_source(repo_root, data_source)
    print(
        f"[run_timestamps] Runs trouvés pour source={data_source!r}: {len(runs)} "
        f"(minimum requis={min_runs})"
    )

    while len(runs) < min_runs:
        before_count = len(runs)
        print(
            f"[run_timestamps] Relance pipeline pour atteindre {min_runs} runs "
            f"(actuel={before_count})…"
        )
        _run_cli_pipeline(repo_root, data_source)
        runs = runs_for_source(repo_root, data_source)
        if len(runs) <= before_count:
            raise RuntimeError(
                "Aucun nouveau run manifesté après exécution pipeline; vérifier les logs."
            )

    latest_path, manifest = latest_finished_manifest(repo_root, data_source)
    stale = manifest is None or age_seconds_since_finished(manifest) > max_age_seconds
    if stale:
        print(
            f"[run_timestamps] Dernier run trop ancien (> {max_age_seconds:.0f} s), "
            "relance pipeline…"
        )
        _run_cli_pipeline(repo_root, data_source)
        latest_path, manifest = latest_finished_manifest(repo_root, data_source)
        runs = runs_for_source(repo_root, data_source)
    else:
        print(
            f"[run_timestamps] Dernier run encore « frais » (< {max_age_seconds:.0f} s) : "
            f"{latest_path.name} — fin UTC {manifest.get('finished_at_utc')}"
        )

    return latest_path, manifest, len(runs)


def print_latest_run_legend(
    repo_root: Path,
    data_source_filter: str | None = None,
) -> None:
    """Affiche le run horodaté le plus récent (global ou filtré par source)."""
    p, m = latest_finished_manifest(repo_root.resolve(), data_source_filter)
    print("\n" + "=" * 70)
    print("  Référence run CLI (`python firewall_svm.py`) — horodatage")
    print("=" * 70)
    if data_source_filter is not None:
        print(f"  Filtre source : {data_source_filter!r}")
    if not m:
        print("  Aucun dossier run-* avec run_manifest.json terminé trouvé.")
        print("  Exécuter : python firewall_svm.py [log2.csv] depuis la racine du dépôt.")
        return
    print(f"  Dossier      : {p.name}/")
    print(f"  Début (UTC)  : {m.get('started_at_utc', '?')}")
    print(f"  Fin (UTC)    : {m.get('finished_at_utc', '?')}")
    print(f"  Source données: {m.get('data_source', '?')!r}")
    print(f"  Meilleur noyau (F1 macro) : {m.get('best_kernel_display_name', '?')}")
    rb = (m.get("f1_per_class_best_kernel_pct") or {}).get("reset-both", "?")
    print(f"  F1 classe « reset-both » (meilleur noyau, avant grille) : {rb} %")
    print(f"  Fichier trace : {p / 'resultats.txt'}")
    print(f"  Manifest JSON : {p / MANIFEST_FILENAME}")
