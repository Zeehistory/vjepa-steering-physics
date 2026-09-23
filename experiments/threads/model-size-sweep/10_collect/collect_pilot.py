#!/usr/bin/env python
"""Collect the decoder-free model-size pilot into one comparison table.

Reads, per (quantity, encoder) cell under $SWEEP/pilot/:
  artifacts/subspace_summary.json      subspace structure
  artifacts/cmd_operator_meta.json     HELD-OUT latent gate: cos(pred dH, true dH) + controls
  out/accel_probe_summary.json         probe R^2

Reports at MATCHED RELATIVE DEPTH (mid and late), because layer index is not comparable across
backbones -- layer 23 is 96% of ViT-L's depth but only 58% of ViT-g's.

Read the gate as steerability-in-latent-space, and read it ONLY against its own controls: a raw
cos looks impressive at any model size because dH is not isotropic. What matters is
cos(real command) - cos(shuffled command).
"""
import argparse
import json
from pathlib import Path

PARAMS_M = {"vjepa2_large": 300, "vjepa2_huge": 600, "vjepa2_giant": 1000}
DEPTH = {  # encoder -> {relative position: layer index}
    "vjepa2_large": {"mid": 12, "late": 23},
    "vjepa2_huge": {"mid": 16, "late": 31},
    "vjepa2_giant": {"mid": 20, "late": 38},
}
ORDER = ["vjepa2_large", "vjepa2_huge", "vjepa2_giant"]
SHORT = {"vjepa2_large": "ViT-L", "vjepa2_huge": "ViT-H", "vjepa2_giant": "ViT-g"}


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def probe_row(probe: dict | None, layer: int, rep: str = "pool") -> dict:
    """probe_accel writes summary['decodability'][rep][str(layer)] = {r2_ax, r2_ay, ...}."""
    if not probe:
        return {}
    return (probe.get("decodability", {}).get(rep, {}) or {}).get(str(layer), {}) or {}


def gate_row(cmd: dict | None, layer: int) -> dict:
    """fit_command_operators_accel writes summary['per_layer'][str(layer)]."""
    if not cmd:
        return {}
    return (cmd.get("per_layer", {}) or {}).get(str(layer), {}) or {}


def g(d: dict, key: str):
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def fmt(x, nd=4):
    return "  --  " if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/vjepa_sweep/pilot")
    ap.add_argument("--quantity", default="accel2d_mixed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root) / args.quantity
    rows, raw = [], {}
    for enc in ORDER:
        cell = root / enc
        if not cell.exists():
            continue
        probe = jload(cell / "out" / "accel_probe_summary.json")
        cmd = jload(cell / "artifacts" / "cmd_operator_meta.json")
        sub = jload(cell / "artifacts" / "subspace_summary.json")
        raw[enc] = {"probe": probe, "cmd": cmd, "subspace": sub}
        for pos, layer in DEPTH[enc].items():
            gr = gate_row(cmd, layer)
            for rep in ("pool", "temporal"):
                pr = probe_row(probe, layer, rep)
                r2ax, r2ay = g(pr, "r2_ax"), g(pr, "r2_ay")
                rows.append({
                    "encoder": enc, "depth": pos, "layer": layer, "rep": rep,
                    "params_M": PARAMS_M[enc],
                    "probe_r2": None if r2ax is None or r2ay is None else 0.5 * (r2ax + r2ay),
                    "probe_r2_mag": g(pr, "r2_mag"),
                    "probe_angle_deg": g(pr, "angle_err_deg"),
                    "probe_ctrl_shuffled_r2": g(pr, "ctrl_shuffled_r2_ax"),
                    "gate_cmdU_cos": g(gr, "cmdU_cos"),
                    "gate_rich_cos": g(gr, "rich_cos"),
                    "gate_coord_cos": g(gr, "coord_cos"),
                    "gate_shuffled_cos": g(gr, "cmdU_cos_shuffled"),
                    "gate_lift": g(gr, "cmdU_cos_lift"),
                    "n_pairs": gr.get("n_pairs"),
                })

    if not rows:
        print(f"no pilot cells found under {root}")
        return 1

    w = 78
    print("=" * w)
    print(f"DECODER-FREE MODEL-SIZE PILOT  --  {args.quantity}")
    print("compared at MATCHED RELATIVE DEPTH (index is not comparable across backbones)")
    print("=" * w)
    print("\n--- PROBE: is acceleration linearly encoded?  (R2 = mean of ax/ay, held out) ---")
    print(f"{'enc':6s} {'depth':5s} {'L':>3s} {'rep':9s} {'R2':>8s} {'R2|a|':>8s} {'ang':>7s} {'shufctl':>8s}")
    for pos in ("mid", "late"):
        for r in [r for r in rows if r["depth"] == pos]:
            print(f"{SHORT[r['encoder']]:6s} {r['depth']:5s} {r['layer']:3d} {r['rep']:9s} "
                  f"{fmt(r['probe_r2'], 3):>8s} {fmt(r['probe_r2_mag'], 3):>8s} "
                  f"{fmt(r['probe_angle_deg'], 2):>7s} {fmt(r['probe_ctrl_shuffled_r2'], 3):>8s}")

    print("\n--- GATE: does the command->edit operator point the right way?  (no decoder) ---")
    print(f"{'enc':6s} {'depth':5s} {'L':>3s} {'cmdU cos':>9s} {'shuffled':>9s} {'LIFT':>8s} {'rich':>8s} {'coord':>8s}")
    seen = set()
    for pos in ("mid", "late"):
        for r in [r for r in rows if r["depth"] == pos]:
            key = (r["encoder"], r["layer"])
            if key in seen:
                continue
            seen.add(key)
            print(f"{SHORT[r['encoder']]:6s} {r['depth']:5s} {r['layer']:3d} "
                  f"{fmt(r['gate_cmdU_cos'], 3):>9s} {fmt(r['gate_shuffled_cos'], 3):>9s} "
                  f"{fmt(r['gate_lift'], 3):>8s} {fmt(r['gate_rich_cos'], 3):>8s} "
                  f"{fmt(r['gate_coord_cos'], 3):>8s}")

    # Scaling deltas vs ViT-L at the same relative depth -- the actual question.
    print("\n" + "=" * w)
    print("SCALING vs ViT-L at matched relative depth  (positive = bigger model is better)")
    print("=" * w)
    for pos in ("mid", "late"):
        base = next((r for r in rows if r["depth"] == pos and r["encoder"] == "vjepa2_large"
                     and r["rep"] == "pool"), None)
        if not base:
            continue
        for enc in ORDER[1:]:
            r = next((x for x in rows if x["depth"] == pos and x["encoder"] == enc
                      and x["rep"] == "pool"), None)
            if not r:
                continue
            d_r2 = (None if r["probe_r2"] is None or base["probe_r2"] is None
                    else r["probe_r2"] - base["probe_r2"])
            d_lift = (None if r["gate_lift"] is None or base["gate_lift"] is None
                      else r["gate_lift"] - base["gate_lift"])
            print(f"  {pos:5s} {SHORT[enc]:6s} vs ViT-L :  "
                  f"d probe R2 = {fmt(d_r2, 4):>8s}   d gate LIFT = {fmt(d_lift, 4):>8s}")

    print("\nNOTE: the gate is a LATENT-space proxy for steerability -- it says the command->edit")
    print("operator points the right way, NOT that decoded pixels move. Pixel steering needs the")
    print("per-encoder decoders (the full sweep). A flat gate here is still informative: it would")
    print("make a pixel-level size effect unlikely.")

    out = Path(args.out) if args.out else root / "pilot_table.json"
    out.write_text(json.dumps({"quantity": args.quantity, "rows": rows, "raw": raw}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
