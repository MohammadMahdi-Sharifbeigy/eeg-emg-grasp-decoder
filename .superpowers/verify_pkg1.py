import json, copy, subprocess, numpy as np, torch, sys
sys.path.insert(0, ".")
nb = json.loads(subprocess.run(["git","show","HEAD:main.ipynb"],capture_output=True,text=True,encoding="utf8").stdout)
cfg_src = next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).lstrip().startswith('CONFIG = {'))
ns = {}; exec("import copy\n" + cfg_src, ns); CONFIG = ns["CONFIG"]
from main.dataloader import load_hs, series_conditions
from main import (preprocess_kinematics_from_config, build_kg_gt_from_config, kin_feature_names)
from main.losses import build_loss_from_config
import main.losses as L
hs = load_hs("data/way-eeg/raw/P1/HS_P1_S4.mat")
kc = CONFIG["preprocessing"]["kinematics"]
kt = preprocess_kinematics_from_config(hs["kin"], float(hs["fs_kin"]), kc)
print("cached kin shape:", kt.shape, "finite:", bool(np.isfinite(kt).all()))
keep = [i for i in range(kt.shape[1]) if i not in kc["drop_indices"]]
x = kt[:, keep]; print("after drop:", x.shape)
names = kin_feature_names(kc["include_velocity"], kc["include_acceleration"], kc["drop_indices"], gated_force=kc.get("gated_force"))
print(len(names), names)
# gated forces must be 0 when hand far (check a parked stretch) and nonzero near
iF = names.index("F_L_gated"); print("F_L_gated max", float(x[:, iF].max()), "frac nonzero", float((np.abs(x[:, iF]) > 1e-6).mean()))
model = build_kg_gt_from_config(CONFIG, input_dim=32, kin_dim=x.shape[1])
print("gat_kin_dim", model.gat_kin_dim, "skip dims", x.shape[1] - model.gat_kin_dim)
eeg = torch.randn(2, 500, 32); kin = torch.tensor((x[:1000] - x.mean(0)) / (x.std(0) + 1e-6)).float().view(2, 500, -1)
with torch.no_grad(): out = model(eeg, kin)
print("forward:", tuple(out.shape), bool(torch.isfinite(out).all()))
# loss with auto channel weights
z = torch.randn(4000, 5).abs() * torch.tensor([1, 1, 1, 1, 3.0])
w = L.compute_channel_weights(z.numpy()) if hasattr(L, "compute_channel_weights") else None
print("channel weights:", None if w is None else np.round(np.asarray(w), 3))
loss_fn = build_loss_from_config(CONFIG, channel_weights=w)
pred = torch.randn(2, 500, 5).abs(); tgt = torch.randn(2, 500, 5).abs()
l = loss_fn(pred, tgt); print("loss:", float(l), {k: round(float(v), 4) for k, v in getattr(loss_fn, "last_components", {}).items()})
l.backward() if l.requires_grad else None
print("OK")
