import cv2,numpy as np,sys
photo,out=sys.argv[1],sys.argv[2]
im=cv2.imread(photo); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
white=(((im.min(axis=2)>230)&(hsv[...,1]<30))*255).astype(np.uint8)
green=((hsv[...,0]>45)&(hsv[...,0]<80)&(hsv[...,1]>150)&(hsv[...,2]>120)).astype(np.uint8)*255
def biggest(m):
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m); i=1+np.argmax(st[1:,4]); return ((lab==i)*255).astype(np.uint8)
def inter(a,b):
    A=np.array([[a[0],-b[0]],[a[1],-b[1]]]);r=np.array([b[2]-a[2],b[3]-a[3]]);t=np.linalg.solve(A,r);return a[2:]+t[0]*a[:2]
def quad(mask):
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    pts=max(cs,key=cv2.contourArea).reshape(-1,2).astype(np.float32)
    box=cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float32)
    # sides: find the two short ones
    sides=[(box[k],box[(k+1)%4]) for k in range(4)]
    L=[np.linalg.norm(b-a) for a,b in sides]
    short=sorted(range(4),key=lambda k:L[k])[:2]
    mids=[(sides[k][0]+sides[k][1])/2 for k in short]
    top_k=short[int(np.argmin([m[1] for m in mids]))]
    # rotate list so that side top_k is the top side a->b, ordering corners cyclic: a,b then the others
    a,b=sides[top_k]; c=sides[(top_k+1)%4][1]; d=sides[(top_k+2)%4][1]
    cyc=[a,b,c,d]
    topmid=(a+b)/2; botmid=(c+d)/2; dd=botmid-topmid; r=np.array([dd[1],-dd[0]])
    # TL = top corner whose offset has negative projection on r
    if np.dot(a-topmid,r)<0: TL,TR,BR,BL=a,b,c,d
    else: TL,TR,BR,BL=b,a,d,c
    corners=[TL,TR,BR,BL]; lines=[]
    for k in range(4):
        p0,p1=corners[k],corners[(k+1)%4]; dv=p1-p0; Ln=np.linalg.norm(dv); u=dv/Ln; nrm=np.array([-u[1],u[0]])
        t=(pts-p0)@u; s=(pts-p0)@nrm
        sel=(t>0.25*Ln)&(t<0.75*Ln)&(np.abs(s)<30)
        p=pts[sel]; v=cv2.fitLine(p,cv2.DIST_HUBER,0,0.01,0.01).ravel()
        for _ in range(3):
            n=np.array([-v[1],v[0]]); dist=np.abs((pts-v[2:])@n); tt=(pts-p0)@u
            pp=pts[(dist<2.5)&(tt>0.2*Ln)&(tt<0.8*Ln)]
            if len(pp)<10: break
            v=cv2.fitLine(pp,cv2.DIST_L2,0,0.01,0.01).ravel()
        lines.append(v)
    q=np.array([inter(lines[3],lines[0]),inter(lines[0],lines[1]),inter(lines[1],lines[2]),inter(lines[2],lines[3])],np.float32)
    return q
mG,mW=biggest(green),biggest(white)
qG,qW=quad(mG),quad(mW)
def wh(q): return (np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2,(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
wg,hg=wh(qG); ww,hw=wh(qW); print('qG',qG.round(1).tolist(),wg/hg); print('qW',qW.round(1).tolist(),ww/hw)
UI='/tmp/claude-tool-results/fb7e0372-0a9b-5f6d-83b5-c36d3f83c346/project-file-f0dfecef-1f9c-4480-9ac5-0529878fa8b0-NIBTap_Interface-615e93d1.PNG'
uiA=cv2.imread(UI); H0,W0=uiA.shape[:2]; asp=ww/hw
if asp<W0/H0:
    nw=int(round(H0*asp)); c=(W0-nw)//2; uiA=uiA[:,c:c+nw]
else:
    nh=int(round(W0/asp)); uiA=np.vstack([uiA[:1650],uiA[1650+(H0-nh):]])
def viewfinder(W,H,cyf=0.45):
    ui=cv2.imread(UI)
    z=1.12*W/1170; cx,cy=586,865; tx,ty=W/2,H*cyf
    feed=cv2.warpAffine(ui,np.float32([[z,0,tx-cx*z],[0,z,ty-cy*z]]),(W,H),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    feed=cv2.GaussianBlur(feed,(0,0),1.0)
    hw_=int(0.5*712*z+14*W/1170); L,T,R,B=int(tx-hw_),int(ty-hw_),int(tx+hw_),int(ty+hw_)
    o=(feed*0.20).astype(np.uint8); o[T:B,L:R]=feed[T:B,L:R]
    s=W/1170; col=(255,255,255); ln,th,off=int(120*s),max(2,int(14*s)),int(10*s)
    def br(x,y,dx,dy):
        cv2.line(o,(x,y),(x+dx*ln,y),col,th,cv2.LINE_AA); cv2.line(o,(x,y),(x,y+dy*ln),col,th,cv2.LINE_AA)
    br(L-off,T-off,1,1);br(R+off,T-off,-1,1);br(L-off,B+off,1,-1);br(R+off,B+off,-1,-1)
    ov=np.zeros_like(o); cv2.rectangle(ov,(L+10,int(ty)-int(5*s)),(R-10,int(ty)+int(5*s)),(25,185,253),-1)
    glow=cv2.GaussianBlur(ov,(0,0),12*s); o=np.clip(o.astype(int)+glow*1.2,0,255).astype(np.uint8); o[int(ty)-int(5*s):int(ty)+int(5*s)+1,L+10:R-10]=(25,185,253)
    return o
uiB=viewfinder(1170,int(round(1170/(wg/hg))),0.45)
S=4; big=cv2.resize(im,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC); res=big.astype(np.float32)
for u,q,m in ((uiA,qW,mW),(uiB,qG,mG)):
    H,W=u.shape[:2]
    M=cv2.getPerspectiveTransform(np.float32([[0,0],[W,0],[W,H],[0,H]]),q*S)
    wu=cv2.warpPerspective(u,M,(1254*S,1254*S),flags=cv2.INTER_AREA)
    am=cv2.warpPerspective(np.full((H,W),255,np.uint8),M,(1254*S,1254*S))
    mk=cv2.resize(m,None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)); mk=cv2.erode(mk,np.ones((3,3),np.uint8))
    a=(cv2.min(am,mk).astype(np.float32)/255); a=cv2.GaussianBlur(a,(0,0),1.2)[...,None]
    res=res*(1-a)+wu*a
o=cv2.resize(np.clip(res,0,255).astype(np.uint8),(1254,1254),interpolation=cv2.INTER_AREA)
ring=(cv2.dilate(mG,np.ones((13,13),np.uint8))>0)
oo=o.astype(np.int32); mx=np.maximum(oo[...,2],oo[...,0]); g=oo[...,1]
oo[...,1]=np.where(ring&(g>mx),mx,g); o=np.clip(oo,0,255).astype(np.uint8)
cv2.imwrite(out,o)
