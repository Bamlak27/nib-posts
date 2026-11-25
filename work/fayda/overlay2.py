#!/usr/bin/env python3
"""Warp an app screenshot onto the phone screen in a photo, keeping fingers in front.
usage: phone_overlay.py photo.png screenshot.png out.png "[(TLx,TLy),(TRx,TRy),(BRx,BRy),(BLx,BLy)]" [crop_bottom_px]
Find the quad by drawing candidate corners on a zoomed crop; nudge 1-3px until a thin dark bezel shows on every side."""
import cv2,numpy as np,sys
photo,shot,out,quad=sys.argv[1:5]; crop_bottom=int(sys.argv[5]) if len(sys.argv)>5 else 0
im=cv2.imread(photo).astype(np.float32); H,W=im.shape[:2]
ui=cv2.imread(shot,cv2.IMREAD_UNCHANGED)[...,:3]
if crop_bottom: ui=ui[:ui.shape[0]-crop_bottom]
ui=ui.astype(np.float32); uh,uw=ui.shape[:2]
Q=np.array(eval(quad),np.float32); K=3
R=int(uw*float(sys.argv[6]) if len(sys.argv)>6 else uw*0.137); inset=5
m=np.zeros((uh,uw),np.float32)
cv2.rectangle(m,(R+inset,inset),(uw-R-inset,uh-inset),1,-1); cv2.rectangle(m,(inset,R+inset),(uw-inset,uh-R-inset),1,-1)
for cx,cy in [(R+inset,R+inset),(uw-R-inset,R+inset),(R+inset,uh-R-inset),(uw-R-inset,uh-R-inset)]: cv2.circle(m,(cx,cy),R,1,-1)
hsv=cv2.cvtColor(ui.astype(np.uint8),cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[...,1]=np.clip(hsv[...,1]*float(sys.argv[7]) if len(sys.argv)>7 else hsv[...,1]*1.28,0,255)
ui=cv2.GaussianBlur(cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR).astype(np.float32),(0,0),0.6)
S=np.array([(0,0),(uw,0),(uw,uh),(0,uh)],np.float32); M=cv2.getPerspectiveTransform(S,Q*K)
wu=cv2.resize(cv2.warpPerspective(ui,M,(W*K,H*K)),(W,H),interpolation=cv2.INTER_AREA)
wm=np.clip(cv2.resize(cv2.warpPerspective(m,M,(W*K,H*K)),(W,H),interpolation=cv2.INTER_AREA),0,1)
qm=np.zeros((H,W),np.uint8); cv2.fillConvexPoly(qm,Q.astype(np.int32),255)
b,g,r=[im[...,i] for i in (0,1,2)]
skin=(((r-g)>=6)&((g-b)<1.6*(r-g)+10)&(r>120)&(r>b+8)).astype(np.uint8)*255
import os
if os.environ.get('SKINBOX'):
    x0,y0,x1,y1=map(int,os.environ['SKINBOX'].split(','));bx=np.zeros_like(skin);bx[y0:y1,x0:x1]=1;skin=skin*bx
skin=cv2.morphologyEx(skin,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
n,lab,stats,_=cv2.connectedComponentsWithStats(skin,connectivity=8)
outside=cv2.dilate(qm,np.ones((3,3),np.uint8))==0
keep=np.zeros(n,bool)
for i in np.unique(lab[outside&(skin>0)]):
    if i and stats[i,cv2.CC_STAT_AREA]>(300 if os.environ.get('SKINBOX') else 2500): keep[i]=True
fm=cv2.GaussianBlur((keep[lab]&(qm>0)).astype(np.float32),(0,0),0.8)
a=(wm*(1-fm))[...,None]
cv2.imwrite(out,np.clip(im*(1-a)+wu*a,0,255).astype(np.uint8))
