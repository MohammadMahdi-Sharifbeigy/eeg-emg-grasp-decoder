import json, subprocess, numpy as np, torch, sys
sys.path.insert(0, ".")
from main import preprocess_kinematics_from_config, build_kg_gt_from_config, kin_feature_names
from main.dataloader import load_hs
from main.losses import build_loss_from_config
for name in ("main.ipynb", "main_1stage.ipynb"):
    nb = json.loads(subprocess.run(["git","show",f"HEAD:{name}"],capture_output=True,text=True,encoding="utf8").stdout)
    bad = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code": continue
        s = "\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith(("%","!")))
        try: compile(s, f"{name}:{i}", "exec")
        except SyntaxError as e: bad += 1; print("SYNTAX", name, i, e)
    src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code" and "".join(c["source"]).lstrip().startswith("CONFIG = {"))
    ns = {}; exec("import copy\n" + src, ns); C = ns["CONFIG"]
    k = C["preprocessing"]["kinematics"]; l = C["training"]["loss"]
    print(name, "| compile errors", bad, "| n_kin", C["data"]["n_kin_features"], "drop", k["drop_indices"], "gat", C["model"]["gat"].get("kin_gat_dim"),
          "skip", C["model"]["gat"].get("kin_skip_indices"), "| asym", l["asymmetry"], "huber", l["huber_delta"], "w_height", l.get("w_height"),
          "| cache", C["data"]["cache_dir"], "| stage2_lr", C["training"].get("stage2_lr"), "ckpt", C["training"].get("checkpoint_dir"))
CONFIG = C
hs = load_hs("data/way-eeg/raw/P1/HS_P1_S4.mat")
kt = preprocess_kinematics_from_config(hs["kin"], float(hs["fs_kin"]), CONFIG["preprocessing"]["kinematics"])
keep = [i for i in range(kt.shape[1]) if i not in CONFIG["preprocessing"]["kinematics"]["drop_indices"]]
x = kt[:, keep]; print("kin after drop", x.shape, "finite", bool(np.isfinite(x).all()))
cached = np.load("data/cache_main_stride500_square_pose_force_sg51/P1_test_S4.npz")["kin"]
print("cache cols", cached.shape[1], "equal to fresh:", bool(np.allclose(cached, kt[:len(cached)], atol=1e-5)))
names = kin_feature_names(True, False, CONFIG["preprocessing"]["kinematics"]["drop_indices"], gated_force=CONFIG["preprocessing"]["kinematics"]["gated_force"])
print(len(names), names[10:12], names[22:])
model = build_kg_gt_from_config(CONFIG, input_dim=32, kin_dim=x.shape[1])
print("gat", model.gat_kin_dim, "skip proj in", model.kin_skip_proj[0].in_features)
kin = torch.tensor((x[:1000]-x.mean(0))/(x.std(0)+1e-6)).float().view(2,500,-1)
model.eval()
with torch.no_grad(): out = model(torch.randn(2,500,32), kin)
print("forward", tuple(out.shape), bool(torch.isfinite(out).all()))
t = torch.zeros(1,500,5); t[:,100:200,:] = 3.0
f = build_loss_from_config(CONFIG["training"], channel_weights=[1]*5)
for n, p in {"perfect": t.clone(), "burst at half height": t.clone(), "burst 1.5x": t.clone()}.items():
    if n == "burst at half height": p[:,100:200,:] = 1.5
    if n == "burst 1.5x": p[:,100:200,:] = 4.5
    f(p, t); c = f.last_components; print(f"  {n:22s} total {c['total']:.3f} height {c.get('height',float('nan')):.3f} peak_mse {c['peak_mse']:.3f}")
