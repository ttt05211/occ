from __future__ import annotations
import numpy as np
from ..protocol import DYNAMIC_CLASS_IDS,REPORT_HORIZONS_S
def raw_counts(pred,gt,moving_support=None,valid_mask=None,free_label=17):
    p=np.asarray(pred); g=np.asarray(gt); mask=np.ones_like(g,dtype=bool) if valid_mask is None else np.asarray(valid_mask,dtype=bool); po=(p!=free_label)&mask; go=(g!=free_label)&mask; out={'occ_inter':int((po&go).sum()),'occ_union':int((po|go).sum()),'sem_inter':{},'sem_union':{},'mov_inter':{},'mov_union':{}}
    for c in range(17):
        pc=(p==c)&mask; gc=(g==c)&mask; out['sem_inter'][c]=int((pc&gc).sum()); out['sem_union'][c]=int((pc|gc).sum())
    mov=np.zeros_like(g,dtype=bool) if moving_support is None else np.asarray(moving_support,dtype=bool)&mask
    for c in DYNAMIC_CLASS_IDS:
        pc=(p==c)&mov; gc=(g==c)&mov; out['mov_inter'][c]=int((pc&gc).sum()); out['mov_union'][c]=int((pc|gc).sum())
    return out
def add_counts(a,b):
    if a is None: return {'occ_inter':b['occ_inter'],'occ_union':b['occ_union'],'sem_inter':dict(b['sem_inter']),'sem_union':dict(b['sem_union']),'mov_inter':dict(b['mov_inter']),'mov_union':dict(b['mov_union'])}
    a['occ_inter']+=b['occ_inter']; a['occ_union']+=b['occ_union'];
    for k in ('sem_inter','sem_union','mov_inter','mov_union'):
        for c,v in b[k].items(): a[k][c]=a[k].get(c,0)+int(v)
    return a
def metrics_from_horizon_counts(by_h):
    per={}; occ=[]; sem=[]; macro=[]; micro=[]
    for h in REPORT_HORIZONS_S:
        r=by_h[float(h)]; oi=100*r['occ_inter']/r['occ_union'] if r['occ_union'] else float('nan'); svals=[r['sem_inter'][c]/r['sem_union'][c] for c in range(17) if r['sem_union'][c]]; mvals=[r['mov_inter'][c]/r['mov_union'][c] for c in DYNAMIC_CLASS_IDS if r['mov_union'][c]]; mi=sum(r['mov_inter'].values()); mu=sum(r['mov_union'].values()); mic=100*mi/mu if mu else float('nan'); sm=100*float(np.mean(svals)) if svals else float('nan'); mm=100*float(np.mean(mvals)) if mvals else float('nan'); sem_pc={c:(100*r['sem_inter'][c]/r['sem_union'][c] if r['sem_union'][c] else float('nan')) for c in range(17)}; mov_pc={c:(100*r['mov_inter'][c]/r['mov_union'][c] if r['mov_union'][c] else float('nan')) for c in DYNAMIC_CLASS_IDS}; per[h]={'IoU':oi,'mIoU':sm,'MacroMoving':mm,'MicroMoving':mic,'semantic_per_class':sem_pc,'moving_per_class':mov_pc,'raw':r}; occ.append(oi); sem.append(sm); macro.append(mm); micro.append(mic)
    return {'IoU':float(np.mean(occ)),'mIoU':float(np.mean(sem)),'MacroMoving':float(np.mean(macro)),'MicroMoving':float(np.mean(micro)),'per_horizon':per}
