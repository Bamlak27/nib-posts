import cv2,numpy as np
sc=cv2.imread('fayda_scene4.png')
X0,Y0,X1,Y1=520,150,960,770
cr=sc[Y0:Y1,X0:X1].copy()
# inpaint the ChatGPT ID-card corner intruding bottom-right
m=np.zeros(cr.shape[:2],np.uint8); m[758-Y0:,711-X0:]=255
cr=cv2.inpaint(cr,m,5,cv2.INPAINT_TELEA)
h,w=cr.shape[:2]
gm=np.full((h,w),cv2.GC_PR_BGD,np.uint8)
gm[:12,:]=cv2.GC_BGD; gm[:,:6]=cv2.GC_BGD; gm[:,-6:]=cv2.GC_BGD
gm[30:h,30:w-30]=cv2.GC_PR_FGD
gm[:150,:60]=cv2.GC_BGD; gm[:150,-60:]=cv2.GC_BGD
cv2.ellipse(gm,(220,210),(110,150),0,0,360,cv2.GC_FGD,-1)
gm[h-60:,90:w-60]=cv2.GC_FGD
bg=np.zeros((1,65));fg=np.zeros((1,65))
cv2.grabCut(cr,gm,None,bg,fg,6,cv2.GC_INIT_WITH_MASK)
mask=((gm==cv2.GC_FGD)|(gm==cv2.GC_PR_FGD)).astype(np.uint8)
n,lab,st,_=cv2.connectedComponentsWithStats(mask)
k=1+np.argmax(st[1:,4]); mask=(lab==k).astype(np.uint8)
inv=(1-mask); n,lab,st,_=cv2.connectedComponentsWithStats(inv)
for i in range(1,n):
    x,y,ww,hh,a=st[i]
    if a<4000 and x>0 and y>0 and x+ww<w and y+hh<h: mask[lab==i]=1
mask=cv2.GaussianBlur(mask.astype(np.float32),(0,0),1.2)
cv2.imwrite('face4_mask.png',(mask*255).astype(np.uint8)); cv2.imwrite('face4_crop.png',cr)
card=cv2.imread('id_face3.png')
bgc=card[160:170,106:112].reshape(-1,3).mean(0)
S=0.62
nw,nh=int(w*S),int(h*S)
g=cv2.cvtColor(cr,cv2.COLOR_BGR2GRAY); g=cv2.resize(g,(nw,nh),interpolation=cv2.INTER_AREA)
mk=cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_AREA)
# gentle contrast
g=np.clip((g.astype(np.float32)-110)*1.12+118,0,255)
bgg=float(bgc.mean())
photo=np.full((nh,nw),bgg,np.float32)
photo=g*mk+photo*(1-mk)
# placement
left=219-int(220*S); top=131
box=np.zeros(card.shape[:2],np.float32)
rr=np.zeros(card.shape[:2],np.uint8); cv2.rectangle(rr,(98+14,148),(340-14,516),255,-1); cv2.rectangle(rr,(98,148+14),(340,516-14),255,-1)
for cx,cy in [(112,162),(326,162),(112,502),(326,502)]: cv2.circle(rr,(cx,cy),14,255,-1)
rr=cv2.GaussianBlur(rr.astype(np.float32)/255,(0,0),1.5)
# bottom fade
fade=np.ones(card.shape[:2],np.float32)
for y in range(480,516): fade[y,:]=1-(y-480)/36*0.0
layer=np.full(card.shape[:2],bgg,np.float32)
y0=max(top,0); 
sub=photo[y0-top:y0-top+(card.shape[0]-y0),:]
x0=left
H=min(sub.shape[0],card.shape[0]-y0); W=min(nw,card.shape[1]-x0)
layer[y0:y0+H,x0:x0+W]=sub[:H,:W]
out=card.astype(np.float32)
for c in range(3): out[...,c]=out[...,c]*(1-rr)+layer*rr
cv2.imwrite('id_face4.png',np.clip(out,0,255).astype(np.uint8))
