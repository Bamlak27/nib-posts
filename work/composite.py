import cv2,numpy as np
P='/mnt/user-data/uploads/'
im=cv2.imread(P+'a9e810cf-image.png').astype(np.float32)
H,W=im.shape[:2]
ui=cv2.imread(P+'c5bd3eef-image.png',cv2.IMREAD_UNCHANGED)
ui=ui[:2418,:,:3].astype(np.float32)
uh,uw=ui.shape[:2]
# rounded mask in source space
R=160;inset=5
m=np.zeros((uh,uw),np.float32)
cv2.rectangle(m,(R+inset,inset),(uw-R-inset,uh-inset),1,-1)
cv2.rectangle(m,(inset,R+inset),(uw-inset,uh-R-inset),1,-1)
for cx,cy in [(R+inset,R+inset),(uw-R-inset,R+inset),(R+inset,uh-R-inset),(uw-R-inset,uh-R-inset)]:
    cv2.circle(m,(cx,cy),R,1,-1)
Q=np.array([(388,568.5),(655,522.5),(852,1101),(571,1162)],np.float32)
S=np.array([(0,0),(uw,0),(uw,uh),(0,uh)],np.float32)
K=3
M=cv2.getPerspectiveTransform(S,Q*K)
hsv=cv2.cvtColor(ui.astype(np.uint8),cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[...,1]=np.clip(hsv[...,1]*1.28,0,255);hsv[...,2]=np.clip(hsv[...,2]*1.03+4,0,255)
ui=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR).astype(np.float32)
ui=cv2.GaussianBlur(ui,(0,0),0.6)  # source is 4x larger; downsample softly
wu=cv2.resize(cv2.warpPerspective(ui,M,(W*K,H*K),flags=cv2.INTER_LINEAR),(W,H),interpolation=cv2.INTER_AREA)
wm=cv2.resize(cv2.warpPerspective(m,M,(W*K,H*K),flags=cv2.INTER_LINEAR),(W,H),interpolation=cv2.INTER_AREA)
wm=np.clip(wm,0,1)
# quad mask for occlusion
qm=np.zeros((H,W),np.uint8);cv2.fillConvexPoly(qm,Q.astype(np.int32),255)
# skin mask
b,g,r=[im[...,i] for i in (0,1,2)]
skin=((r-g)>=6)&((g-b)<1.6*(r-g)+10)&(r>120)&(r>b+8)
skin=skin.astype(np.uint8)*255
skin=cv2.morphologyEx(skin,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
n,lab,stats,_=cv2.connectedComponentsWithStats(skin,connectivity=8)
outside=cv2.dilate(qm,np.ones((3,3),np.uint8))==0
keep=np.zeros(n,bool)
ids=np.unique(lab[outside&(skin>0)])
for i in ids:
    if i==0: continue
    if stats[i,cv2.CC_STAT_AREA]>2500: keep[i]=True
fin=keep[lab]&(qm>0)
cv2.imwrite('work/fingers_mask.png',fin.astype(np.uint8)*255)
fm=cv2.GaussianBlur(fin.astype(np.float32),(0,0),0.8)
a=wm*(1-fm)
a3=a[...,None]
out=im*(1-a3)+wu*a3
cv2.imwrite('work/photo_comp.png',np.clip(out,0,255).astype(np.uint8))
