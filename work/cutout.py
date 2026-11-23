import cv2,numpy as np
from scipy import ndimage as ndi
im=cv2.imread('work/photo_comp.png').astype(np.float32)[...,::-1]  # RGB
H,W=im.shape[:2]
Bc=np.array([253,253,253],np.float32)
dist=np.abs(im-Bc).max(2)
# background candidate: close to white, connected to border
cand=dist<11
lab,n=ndi.label(cand)
sizes=ndi.sum(cand,lab,range(1,n+1))
qm=np.zeros((H,W),np.uint8)
cv2.fillConvexPoly(qm,np.array([(391,567),(653,523),(849,1100),(573,1160)],np.int32),1)
qm=ndi.binary_dilation(qm,iterations=6)
inq=ndi.sum(qm,lab,range(1,n+1))>0
border=set(np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])))-{0}
good=[i+1 for i in range(n) if (sizes[i]>=25 and not inq[i]) or (i+1) in border]
bg=np.isin(lab,good)
fg_def=ndi.binary_erosion(~ndi.binary_dilation(bg,iterations=5),iterations=1)
# unknown = neither
unk=~bg&~fg_def
# nearest fg colour for unknown
idx=ndi.distance_transform_edt(~fg_def,return_distances=False,return_indices=True)
F=im[idx[0],idx[1]]
d=F-Bc;C=im-Bc
den=(d*d).sum(2)
alpha=np.where(den>900,(C*d).sum(2)/np.maximum(den,1),0.5)
alpha=np.clip(alpha,0,1)
# for weak-contrast unknowns keep alpha from dilation distance
a=np.where(bg,0.0,np.where(fg_def,1.0,alpha))
weak=unk&(den<=900)
dd=ndi.distance_transform_edt(~bg)
a[weak]=np.clip(dd[weak]/3.0,0,1)
a=np.clip(a,0,1)
# decontaminate
a3=np.maximum(a,1e-3)[...,None]
Fo=np.where(unk[...,None],(im-(1-a3)*Bc)/a3,im)
Fo=np.clip(Fo,0,255)
out=np.dstack([Fo,a*255]).astype(np.uint8)
cv2.imwrite('work/photo_nt_smart.png',cv2.cvtColor(out,cv2.COLOR_RGBA2BGRA))
# preview on dark + brown
for name,col in (('d',(60,30,20)),('c',(255,246,237))):
    bgc=np.ones((H,W,3),np.float32)*np.array(col,np.float32)
    pv=Fo*a[...,None]/255*0+ (Fo*a[...,None]+bgc*(1-a[...,None]))
    cv2.imwrite(f'/tmp/pv_{name}.png',cv2.cvtColor(pv.astype(np.uint8),cv2.COLOR_RGB2BGR))
