import cv2,numpy as np,sys
photo,out=sys.argv[1],sys.argv[2]
im=cv2.imread(photo); hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
white=(((im.min(axis=2)>232)&(hsv[...,1]<28))*255).astype(np.uint8)
green=((hsv[...,0]>45)&(hsv[...,0]<80)&(hsv[...,1]>150)&(hsv[...,2]>120)).astype(np.uint8)*255
def biggest(m):
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m); i=1+np.argmax(st[1:,4]); return ((lab==i)*255).astype(np.uint8)
def order(c):
    c=c[np.argsort(c[:,1])]; top=c[:2][np.argsort(c[:2,0])]; bot=c[2:][np.argsort(c[2:,0])]
    return np.array([top[0],top[1],bot[1],bot[0]],np.float32)
def quad(mask):
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    pts=max(cs,key=cv2.contourArea).reshape(-1,2).astype(np.float32)
    rect=cv2.minAreaRect(pts); box=cv2.boxPoints(rect).astype(np.float32); box=order(box)
    lines=[]
    for k in range(4):
        a,b=box[k],box[(k+1)%4]; d=b-a; L=np.linalg.norm(d); u=d/L; nrm=np.array([-u[1],u[0]])
        t=(pts-a)@u; s=np.abs((pts-a)@nrm)
        sel=(t>0.25*L)&(t<0.75*L)&(s<30)
        p=pts[sel]
        # keep the points nearest to this side only (outer boundary), then refit tightly
        v=cv2.fitLine(p,cv2.DIST_HUBER,0,0.01,0.01).ravel()
        for _ in range(3):
            n=np.array([-v[1],v[0]]); dd=np.abs((pts-v[2:])@n); tt=(pts-a)@u
            p=pts[(dd<2.5)&(tt>0.2*L)&(tt<0.8*L)]
            if len(p)<10: break
            v=cv2.fitLine(p,cv2.DIST_L2,0,0.01,0.01).ravel()
        lines.append(v)
    def inter(a,b):
        A=np.array([[a[0],-b[0]],[a[1],-b[1]]]);r=np.array([b[2]-a[2],b[3]-a[3]]);t=np.linalg.solve(A,r);return a[2:]+t[0]*a[:2]
    q=np.array([inter(lines[3],lines[0]),inter(lines[0],lines[1]),inter(lines[1],lines[2]),inter(lines[2],lines[3])],np.float32)
    return q
mA,mB=biggest(white),biggest(green)
qA,qB=quad(mA),quad(mB); print('A',qA.tolist()); print('B',qB.tolist())
def cropui(ui,asp):
    H,W=ui.shape[:2]; nh=int(round(W/asp)); cut=H-nh
    if cut<=0: return ui
    return np.vstack([ui[:1650],ui[1650+cut:]]) if H==2532 and 1650+cut<H else ui[:nh]
uiA=cv2.imread('/tmp/claude-tool-results/fb7e0372-0a9b-5f6d-83b5-c36d3f83c346/project-file-f0dfecef-1f9c-4480-9ac5-0529878fa8b0-NIBTap_Interface-615e93d1.PNG')
uiB=cv2.imread('scanner_ui.png')
asp=0.491
S=4; big=cv2.resize(im,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC)
res=big.astype(np.float32)
for ui,q,m in ((uiA,qA,mA),(uiB,qB,mB)):
    u=cropui(ui,asp); H,W=u.shape[:2]
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
g2=np.where(ring&(g>mx),mx,g); oo[...,1]=g2; o=np.clip(oo,0,255).astype(np.uint8)
cv2.imwrite(out,o); np.save('qA.npy',qA); np.save('qB.npy',qB)
