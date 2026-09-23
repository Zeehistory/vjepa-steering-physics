import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base="outputs/analysis/moving_ball_v2d/interp"
I=json.load(open(f"{base}/interp_summary.json"))["layers_detail"]["L12"]
X=json.load(open(f"{base}/cross_object_summary.json"))["layers_detail"]["L12"]

fig,ax=plt.subplots(1,3,figsize=(15,4.6))

# ---- panel 1: read >> write, within disk -----------------------------------------------------------
m=I["methods"]
order=["full_delta","subspace_U8","ridge_global","random8"]
labels=["per-pair\n(on-manifold)","shared U8\nsubspace","global linear\noperator","random\n(control)"]
ach=[m[k]["achieved_frac_of_command"] for k in order]
cols=["#2a7","#49c","#e8a23d","#bbb"]
ax[0].bar(labels,ach,color=cols)
ax[0].axhline(1.0,ls="--",c="k",lw=.8); ax[0].set_ylim(0,1.25)
ax[0].set_ylabel("fraction of commanded velocity\ninstalled in v-jepa's own readout")
ax[0].set_title("read ≫ write\n(probe reads velocity at R²=0.998; writing it back is partial)",fontsize=10.5)
for i,v in enumerate(ach): ax[0].text(i,v+0.03,f"{v:.2f}",ha="center",fontsize=9)

# ---- panel 2: the edit is mostly scene-local -------------------------------------------------------
shared=I["delta_U8_energy_frac_mean"]; local=1-shared
ax[1].bar(["per-pair velocity edit"],[shared],color="#49c",label=f"shared, transferable subspace ({shared*100:.0f}%)")
ax[1].bar(["per-pair velocity edit"],[local],bottom=[shared],color="#d99",label=f"scene-local ({local*100:.0f}%)")
ax[1].set_ylim(0,1); ax[1].set_ylabel("fraction of edit energy")
ax[1].legend(loc="upper center",bbox_to_anchor=(.5,-.08),fontsize=9,frameon=False)
ax[1].set_title("why a single vector doesn't transfer:\nmost of the edit is scene-specific",fontsize=10.5)

# ---- panel 3: cross-object transfer (disk -> square) -----------------------------------------------
rd=X["read"]; al=X["align"]
read_vals=[np.mean(rd["disk_probe_on_disk_test_r2"]),
           np.mean(rd["disk_probe_on_square_test_r2"]),
           np.mean(rd["oracle_square_probe_on_square_test_r2"])]
xr=np.arange(3)
ax[2].bar(xr,read_vals,color=["#2a7","#c5527a","#888"],width=.6)
ax[2].set_xticks(xr); ax[2].set_xticklabels(["disk probe\non disk","disk probe\non SQUARE","oracle\n(square probe)"],fontsize=9)
ax[2].set_ylim(0,1.1); ax[2].set_ylabel("velocity read-out R² (mean vx,vy)")
ax[2].set_title(f"partially object-agnostic\n(disk↔square subspace {al['principal_angle_diskU8_vs_squareU8']['mean_deg']:.0f}°; random 90°)",fontsize=10.5)
for i,v in enumerate(read_vals): ax[2].text(i,v+0.02,f"{v:.2f}",ha="center",fontsize=9)

fig.suptitle("frozen v-jepa2 velocity: genuinely encoded, partly abstract — the steer is on-manifold interpolation riding a thin object-agnostic axis",
             fontsize=11.5,y=1.02)
fig.tight_layout()
import os; os.makedirs("outputs/figures",exist_ok=True)
out="outputs/figures/vjepa_velocity_interp.png"
fig.savefig(out,dpi=150,bbox_inches="tight")
print("saved",out)
