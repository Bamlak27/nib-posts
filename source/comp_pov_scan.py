import cv2,numpy as np,sys
photo,out=sys.argv[1],sys.argv[2]
im=cv2.imread(photo); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
white=(((im.min(axis=2)>232)&(hsv[...,1]<28))*255).astype(np.uint8)
green=((hsv[...,0]>45)&(hsv[...,0]<80)&(hsv[...,1]>150)&(hsv[...,2]>120)).astype(np.uint8)*255
def biggest(m):
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m); i=1+np.argmax(st[1:,4]); return ((lab==i)*255).astype(np.uint8)
mA,mB=biggest(white),biggest(green)
def pts_of(m):
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); return max(cs,key=cv2.contourArea).reshape(-1,2).astype(np.float32)
def fit(p):
    v=cv2.fitLine(p,cv2.DIST_HUBER,0,0.01,0.01).ravel(); return v
def inter(a,b):
    A=np.array([[a[0],-b[0]],[a[1],-b[1]]]);r=np.array([b[2]-a[2],b[3]-a[3]]);t=np.linalg.solve(A,r);return a[2:]+t[0]*a[:2]
def at_y(l,y): return np.array([l[2]+(y-l[3])*l[0]/l[1],y],np.float32)
PA,PB=pts_of(mA),pts_of(mB)
# ---- A (white, woman's phone): top, left, right visible; bottom hidden partly
x0,y0,w,h=cv2.boundingRect(PA.astype(np.int32)); print('A bbox',x0,y0,w,h)
ymid=y0+h
LA=fit(PA[(PA[:,0]<x0+4)&(PA[:,1]>y0+70)&(PA[:,1]<y0+h*0.75)])
RA=fit(PA[(PA[:,0]>x0+w-5)&(PA[:,1]>y0+70)&(PA[:,1]<y0+h-70)])
TA=fit(PA[(PA[:,1]<y0+3)&(PA[:,0]>x0+45)&(PA[:,0]<x0+w-45)])
BA=np.array([TA[0],TA[1],0,y0+h-2],np.float32)  # parallel to top, at lowest visible y
qA=np.array([inter(TA,LA),inter(TA,RA),inter(BA,RA),inter(BA,LA)],np.float32)
# ---- B (green, scanner phone): top, left, right visible; bottom extended
bx,by,bw,bh=cv2.boundingRect(PB.astype(np.int32)); print('B bbox',bx,by,bw,bh)
LB=fit(PB[(PB[:,0]<bx+bw*0.6)&(PB[:,1]>by+90)&(PB[:,1]<by+bh-30)&(PB[:,0]<np.interp(PB[:,1],[by,by+bh],[bx+60,bx+60])) ])
# robust: use min-x points per row for left, max-x per row for right
rows=range(by+90,by+bh-30,4); lp=[];rp=[]
for y in rows:
    xs=np.where(mB[y]>0)[0]
    if len(xs): lp.append((xs.min(),y)); rp.append((xs.max(),y))
LB=fit(np.array(lp,np.float32)); RB=fit(np.array(rp,np.float32))
TB=fit(PB[(PB[:,1]<by+3)&(PB[:,0]>bx+60)&(PB[:,0]<bx+bw-60)])
HB=float(sys.argv[3]) if len(sys.argv)>3 else 700
tl,tr=inter(TB,LB),inter(TB,RB); ytop=(tl[1]+tr[1])/2
qB=np.array([tl,tr,at_y(RB,ytop+HB),at_y(LB,ytop+HB)],np.float32)
print('qA',qA.tolist()); print('qB',qB.tolist())
def wh(q): return (np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2,(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
wa,ha=wh(qA); wb,hb=wh(qB); print('aspects',wa/ha,wb/hb)
# UI A: real NIBTap screen, crop columns to the apparent aspect
uiA=cv2.imread('/tmp/claude-tool-results/fb7e0372-0a9b-5f6d-83b5-c36d3f83c346/project-file-f0dfecef-1f9c-4480-9ac5-0529878fa8b0-NIBTap_Interface-615e93d1.PNG')
H0,W0=uiA.shape[:2]; asp=wa/ha
if asp<W0/H0:
    nw=int(round(H0*asp)); c=(W0-nw)//2; uiA=uiA[:,c:c+nw]
else:
    nh=int(round(W0/asp)); uiA=np.vstack([uiA[:1650],uiA[1650+(H0-nh):]])
# UI B: scanner viewfinder built from the same real screen, QR window high on the screen
def viewfinder(W,H,cyf=0.30):
    ui=cv2.imread('/tmp/claude-tool-results/fb7e0372-0a9b-5f6d-83b5-c36d3f83c346/project-file-f0dfecef-1f9c-4480-9ac5-0529878fa8b0-NIBTap_Interface-615e93d1.PNG')
    z=1.12*W/1170; cx,cy=586,865; tx,ty=W/2,H*cyf
    feed=cv2.warpAffine(ui,np.float32([[z,0,tx-cx*z],[0,z,ty-cy*z]]),(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    feed=cv2.GaussianBlur(feed,(0,0),1.0)
    hw=int(0.5*712*z+14*W/1170); L,T,R,B=int(tx-hw),int(ty-hw),int(tx+hw),int(ty+hw)
    o=(feed*0.20).astype(np.uint8); o[T:B,L:R]=feed[T:B,L:R]
    s=W/1170; col=(255,255,255); ln,th,off=int(120*s),max(2,int(14*s)),int(10*s)
    def br(x,y,dx,dy):
        cv2.line(o,(x,y),(x+dx*ln,y),col,th,cv2.LINE_AA); cv2.line(o,(x,y),(x,y+dy*ln),col,th,cv2.LINE_AA)
    br(L-off,T-off,1,1);br(R+off,T-off,-1,1);br(L-off,B+off,1,-1);br(R+off,B+off,-1,-1)
    ov=np.zeros_like(o); cv2.rectangle(ov,(L+10,int(ty)-int(5*s)),(R-10,int(ty)+int(5*s)),(25,185,253),-1)
    glow=cv2.GaussianBlur(ov,(0,0),12*s); o=np.clip(o.astype(int)+glow*1.2,0,255).astype(np.uint8); o[int(ty)-int(5*s):int(ty)+int(5*s)+1,L+10:R-10]=(25,185,253)
    return o
uiB=viewfinder(1170,int(round(1170/(wb/hb))))
S=4; big=cv2.resize(im,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC); res=big.astype(np.float32)
for u,q,m in ((uiA,qA,mA),(uiB,qB,mB)):
    H,W=u.shape[:2]
    M=cv2.getPerspectiveTransform(np.float32([[0,0],[W,0],[W,H],[0,H]]),q*S)
    wu=cv2.warpPerspective(u,M,(1254*S,1254*S),flags=cv2.INTER_AREA)
    am=cv2.warpPerspective(np.full((H,W),255,np.uint8),M,(1254*S,1254*S))
    mk=cv2.resize(m,None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)); mk=cv2.erode(mk,np.ones((3,3),np.uint8))
    a=(cv2.min(am,mk).astype(np.float32)/255); a=cv2.GaussianBlur(a,(0,0),1.2)[...,None]
    res=res*(1-a)+wu*a
o=cv2.resize(np.clip(res,0,255).astype(np.uint8),(1254,1254),interpolation=cv2.INTER_AREA)
ring=(cv2.dilate(mB,np.ones((13,13),np.uint8))>0)
oo=o.astype(np.int32); mx=np.maximum(oo[...,2],oo[...,0]); g=oo[...,1]
oo[...,1]=np.where(ring&(g>mx),mx,g); o=np.clip(oo,0,255).astype(np.uint8)
cv2.imwrite(out,o)
