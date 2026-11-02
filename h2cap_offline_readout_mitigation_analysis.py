import json,numpy as np
from scipy.optimize import nnls
rng=np.random.default_rng(12345)
G=json.load(open('h2cap_ibm_hardware_20261013_070039_groups.json'));A=json.load(open('h2cap_ibm_independent_audit_20261013.json'));pc=A['per_circuit_audit']
RO=json.load(open('stage8L1_readout_db6o9sqmb58s738c4cj0.json'))
ETA=1.30446839e-5;THR=-1.423580144198899;HE=27.211386245988
states=['%03d'%int(bin(i)[2:]) for i in range(8)]
idx={s:i for i,s in enumerate(states)}
# calibration counts per triple
cal={}
for m,c in zip(RO['meta'],RO['counts']):
    t=tuple(m['triple']);cal.setdefault(t,np.zeros((8,8)))
    for b,k in c.items(): cal[t][idx[m['label']],idx[b]]+=k
for t in [(43,44,45),(20,21,36)]:
    C=cal[t];R=C/C.sum(1,keepdims=True)
    print(t,'diag',np.round(np.diag(R),4),'mean assign err %.4f'%(1-np.mean(np.diag(R))))
# sign vectors per label
def sgn(lab):
    v=np.ones(8)
    for s in states:
        for p,ch in enumerate(lab):
            if ch!='I' and s[p]=='1': v[idx[s]]*=-1
    return v
def mitigate(p,R,mode):
    if mode=='none': return p
    if mode=='inv': return np.linalg.solve(R.T,p)
    x,_=nnls(R.T,p); return x/x.sum()
def run(counts_list,Rs,mode):
    out={}
    for t in (0,1):
        R=Rs[t];Hs=0;Ws=0
        for g in range(54):
            i=t*54+g;c=counts_list[i];p=np.zeros(8)
            for b,k in c.items():p[idx[b]]=k
            p/=p.sum();q=mitigate(p,R,mode)
            est=sum(co*(sgn(l)@q) for l,co in zip(G[i]['pauli_labels'],G[i]['coefficients']))
            if G[i]['observable']=='H':Hs+=est
            else:Ws+=est
        out[t]=(Hs,Ws)
    return out
def conv(H,W): return (H-THR)*HE, 2*ETA*W*HE
trip=[(43,44,45),(20,21,36)]
Rhat=[cal[t]/cal[t].sum(1,keepdims=True) for t in trip]
raw=[pc[i]['raw_counts'] for i in range(108)]
res={}
for mode in ('inv',):
    o=run(raw,Rhat,mode)
    # MC
    S=[]
    for k in range(2000):
        rc=[]
        for i in range(108):
            c=raw[i];keys=list(c);n=sum(c.values());pr=np.array([c[x] for x in keys])/n
            s=rng.multinomial(n,pr);rc.append({x:int(v) for x,v in zip(keys,s)})
        Rs=[]
        for t in trip:
            C=cal[t];Rb=np.array([rng.multinomial(int(C[r].sum()),C[r]/C[r].sum()) for r in range(8)],float)
            Rs.append(Rb/Rb.sum(1,keepdims=True))
        oo=run(rc,Rs if mode!='none' else Rhat,mode)
        S.append([*conv(*oo[0]),*conv(*oo[1])])
    S=np.array(S);se=S.std(0)
    res[mode]=(o,se)
    for t in (0,1):
        E,Gm=conv(*o[t]);print(mode,trip[t],'Er %.5f +/- %.5f  Gamma %.5f +/- %.5f'%(E,se[2*t],Gm,se[2*t+1]))
    # combined & layout difference
    E0,G0=conv(*o[0]);E1,G1=conv(*o[1])
    print('   Gamma diff (43-20) %.5f ; z=%.2f'%(G0-G1,(G0-G1)/np.hypot(se[1],se[3])))
import csv
o,se=res['inv']
with open('mitigation_results.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['triple','mode','Er_eV','Er_SE','Gamma_eV','Gamma_SE'])
    for t in (0,1):
        E,Gm=conv(*o[t]);w.writerow([trip[t],'readout-mitigated (inverse confusion matrix, MC errors incl. calibration)',E,se[2*t],Gm,se[2*t+1]])
