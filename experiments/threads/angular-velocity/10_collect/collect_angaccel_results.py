#!/usr/bin/env python
"""Collect every angular-acceleration / robustness result into one publication-grade table.

Reads the JSONs written by steer_fourier.py and ablate_fourier.py and prints (a) the headline transfer
table, (b) the ablation grid, (c) the model-size sweep. Reports each steer against BOTH of its references
-- the random-command control (is it above chance?) and the decoder's own ceiling (how much of what the
instrument can render did the operator capture?) -- because the ceiling, not the operator, is a co-limiter
for angular acceleration and a bare rho would hide that.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def fmt(x, n=3):
    return "  --  " if x is None else f"{x:+.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/analysis/moving_ball_angaccel2d")
    ap.add_argument("--angvel_dir", default="outputs/analysis/moving_ball_angvel2d/polar")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    D = Path(args.dir)
    summary = {}

    print("\n" + "=" * 104)
    print("  ANGULAR ACCELERATION: zero-shot transfer of the Fourier-in-orientation operator")
    print("=" * 104)
    print(f"  {'run':<34} {'fit on':<16} {'gate':>7} {'shuf':>7} {'rho':>7} {'sign':>6} {'mag':>7} {'ceil':>7} {'ctrl':>7}")
    print("  " + "-" * 100)
    runs = [
        ("zero-shot (ViT-L, w-decoder)", "fourier_zeroshot_decode.json", "constant-omega"),
        ("matched fit (ViT-L, w-decoder)", "fourier_matched_decode.json", "angaccel"),
        ("zero-shot (ViT-L, alpha-decoder)", "fourier_zeroshot_adec.json", "constant-omega"),
        ("zero-shot order-8 (val-selected)", "fourier_zeroshot_o8_decode.json", "constant-omega"),
        ("matched fit (ViT-L, alpha-decoder)", "fourier_matched_adec.json", "angaccel"),
        ("zero-shot (ViT-g, matched depth)", "fourier_zeroshot_vitg_gate.json", "constant-omega"),
        ("zero-shot (ViT-g, pixel-verified)", "fourier_zeroshot_vitg_decode.json", "constant-omega"),
        ("[ref] angvel ViT-g (matched depth)", "fourier_angvel_vitg_gate_matcheddepth.json", "constant-omega"),
        ("zero-shot mixed appearance", "fourier_zeroshot_mixed.json", "constant-omega"),
    ]
    for label, fn, fit in runs:
        r = load(D / fn)
        if not r:
            continue
        g = r.get("gate", {})
        best = max(g.values(), key=lambda d: d["cos"]) if g else {}
        h = r.get("heldout", {}); c = r.get("ceiling", {}); rc = r.get("random_command", {})
        print(f"  {label:<34} {fit:<16} {fmt(best.get('cos'))} {fmt(best.get('cos_shuffled'))} "
              f"{fmt(h.get('rho'))} {h.get('sign_acc', float('nan')):>6.2f} {fmt(h.get('mag_ratio'))} "
              f"{fmt(c.get('rho'))} {fmt(rc.get('rho'))}")
        summary[label] = dict(gate=best.get("cos"), rho=h.get("rho"), ceiling=c.get("rho"),
                              control=rc.get("rho"), sign=h.get("sign_acc"), mag=h.get("mag_ratio"))
        if h.get("rho") and c.get("rho"):
            summary[label]["frac_of_ceiling"] = h["rho"] / c["rho"]

    # angular-velocity reference (the solved result this generalizes)
    av = load(Path(args.angvel_dir) / "fourier_decode_o4.json")
    if av:
        h = av.get("heldout", {})
        print(f"  {'[ref] angular VELOCITY (in-domain)':<34} {'constant-omega':<16} "
              f"{fmt(av.get('gate', {}).get('6', {}).get('cos'))} {'':>7} {fmt(h.get('rho'))} "
              f"{h.get('sign_acc', float('nan')):>6.2f} {fmt(h.get('mag_ratio'))}")

    for fn, title in (("ablation_zeroshot.json", "ABLATION GRID (zero-shot alpha, held-out latent gate)"),
                      ("ablation_vitg.json", "ABLATION GRID (ViT-g)")):
        ab = load(D / fn)
        if not ab:
            continue
        print("\n" + "=" * 104)
        print(f"  {title}")
        print("=" * 104)
        L0 = ab["layers"][0]
        print(f"  {'axis':<12} {'setting':<28} {'L' + str(L0) + ' cos':>9} {'best cos':>9} {'shuffled':>9}  note")
        print("  " + "-" * 100)
        NOTE = {
            "order=0": "DC only -> must be null: a constant cannot express a rotation",
            "order=1": "fundamental only (marker's 2pi cue)",
            "order=2": "adds the EVEN harmonic = the bar's pi-symmetry",
            "harmonics=even": "bar content only",
            "harmonics=odd": "marker content only",
            "canon=False": "no center-canonicalization -> per-scene centre confound",
            "basis=orientation_rate": "adds rate-interacted harmonics",
        }
        for r in ab["results"]:
            setting = ("(operator default)" if r["axis"] == "default" else
                       f"order={r['order']}" if r["axis"] == "order" else
                       f"harmonics={r['harm']}" if r["axis"] == "harmonics" else
                       f"canon={r['canon']}" if r["axis"] == "canon" else
                       f"basis={r['basis']}" if r["axis"] == "basis" else
                       f"ridge={r['ridge']}" if r["axis"] == "ridge" else
                       f"n_train_scenes={r['n_train']}")
            print(f"  {r['axis']:<12} {setting:<28} {r['gate'][str(L0)]['cos']:>+9.3f} "
                  f"{r['best_cos']:>+9.3f} {r['best_shuffled']:>+9.3f}  {NOTE.get(setting, '')}")
        summary[fn] = {(r["axis"] + ":" + str(r.get(r["axis"], ""))): r["best_cos"] for r in ab["results"]}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
