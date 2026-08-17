# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import fior_sim as F
print("worst-direction discrepancy t = max_dir |P[n]-med|/mad, evaluator A0, 3 seeds")
print(f"{'shift':>6} {'same-cluster':>22} {'cross-cluster':>22} {'fabricator':>22}")
for shift in (3.0, 10.0, 30.0):
    cls = {0:[],1:[],2:[]}
    for seed in range(3):
        w = F.build_world(F.Config(attack="targeted", attack_shift=shift, seed=seed))
        C = F.Corroboration(w.sites); A = 0
        p = np.full(w.N, 0.5)
        for n in range(w.N):
            if n == A: continue
            wts = p.copy(); wts[n]=0.0; wts[A]=0.0
            P = C.P[n]; W = np.repeat(wts[:,None], P.shape[1], axis=1)
            med = F._wmedian(P, W)
            mad = np.maximum(F._wmedian(np.abs(P-med), W), 1e-9)
            k = 2 if w.label[n]==2 else (0 if w.label[n]==w.label[A] else 1)
            cls[k].append((np.abs(P[n]-med)/mad).max())
    f = lambda v: f"med {np.median(v):6.1f} max {np.max(v):7.1f}"
    print(f"{shift:6.0f} {f(cls[0]):>22} {f(cls[1]):>22} {f(cls[2]):>22}")
