import numpy as np, scipy.io as sio, pandas as pd, glob, sys
sys.path.insert(0,'.')
from main.dataloader import stratified_split, series_conditions
P=1; C='data/cache_main_stride500_square_pose_force_sg51'; FS=500
M=['FDI','APB','ADM','ECR','FCR']
m=sio.loadmat(f'data/way-eeg/raw/P{P}/P{P}_AllLifts.mat',squeeze_me=True,struct_as_record=False)['P']
L=pd.DataFrame(np.asarray(m.AllLifts),columns=[str(x) for x in np.atleast_1d(m.ColNames)])
sp=stratified_split(P); cond=series_conditions(P)
rows=[]
for split in ('train','val','test'):
    for sid in sp[split]:
        d=np.load(f'{C}/P{P}_{split}_S{sid}.npz'); emg=d['emg']; kin=d['kin']
        for _,l in L[L.Run==sid].iterrows():
            t0=l.StartTime-l.LEDOn
            a=int((t0+l.tLiftOff+0.3)*FS); b=int((t0+l.tReplace-0.3)*FS)
            r0=int((t0+0.2)*FS); r1=int((t0+l.LEDOn-0.2)*FS)
            if b-a<100 or b>len(emg): continue
            row=dict(split=split,sid=sid,cond=cond[sid],W=int(l.CurW),S=int(l.CurS),FL=kin[a:b,10].mean(),FG=kin[a:b,11].mean(),dur=(b-a)/FS)
            for i,n in enumerate(M): row[n]=emg[a:b,i].mean(); row[n+'_rest']=emg[max(r0,0):r1,i].mean()
            rows.append(row)
D=pd.DataFrame(rows); D.to_csv('.superpowers/lifts_P1.csv',index=False)
pd.set_option('display.width',200); pd.set_option('display.float_format',lambda x:f'{x:.4f}')
print("lifts per split:",D.groupby('split').size().to_dict())
print("\n== hold-phase EMG mean per series (all weights) and gated load force")
print(D.groupby(['split','sid','cond'])[M+['FL','FG']].mean())
print("\n== hold-phase EMG by weight, train vs test series 4 (W) ")
print(D[D.cond=='W'].groupby(['W','split','sid'])[M+['FL']].mean())
# linear model fit on train lifts: EMG ~ 1 + FL + FG, predict val/test lifts
tr=D[D.split=='train']
X=lambda d: np.c_[np.ones(len(d)),d.FL,d.FG]
print("\n== ideal linear force model (fit on train lifts): height ratio = sum(pred)/sum(actual) per series")
out=[]
for n in M:
    w,*_=np.linalg.lstsq(X(tr),tr[n].values,rcond=None)
    for (split,sid),g in D.groupby(['split','sid']):
        p=X(g)@w
        out.append(dict(muscle=n,split=split,sid=sid,cond=cond[sid],ratio=p.sum()/g[n].sum(),r=np.corrcoef(p,g[n])[0,1]))
O=pd.DataFrame(out)
print(O.pivot_table(index=['split','sid','cond'],columns='muscle',values='ratio')[M])
print("\n per-lift r of that model (does force explain lift-to-lift magnitude?)")
print(O.pivot_table(index=['split','sid','cond'],columns='muscle',values='r')[M])
print("\n== lift-to-lift variability of hold EMG (CV) within each series")
print(D.groupby(['split','sid','cond'])[M].agg(lambda x:x.std()/x.mean()))
