# Builds a physically-plausible foot/leg shadow from the man's silhouette (sun upper-left, ground seen from above)
import numpy as np, cv2
from PIL import Image
S=1254
man=Image.open('source/photo_wed_man.png').convert('RGBA').resize((953,953),Image.LANCZOS)
canvas=Image.new('RGBA',(S,S),(0,0,0,0));canvas.alpha_composite(man,(350,128))
A=np.array(canvas)[...,3].astype(np.float32)/255
# sole contact lines (canvas px): left shoe toe->heel, right shoe heel->toe
lines={'L':((792,1066),(870,1030),(0,945)),'R':((1048,1028),(1098,1046),(945,S))}
SH=0.40;VS=0.40;DMAX=240.0
acc=np.zeros((S,S),np.float32);dist=np.zeros((S,S),np.float32)
for k,((x0,y0),(x1,y1),(xa,xb)) in lines.items():
    m=(y1-y0)/(x1-x0)
    ys,xs=np.mgrid[0:S,0:S]
    cy=y0+m*(xs-x0)               # contact line y at each column
    d=cy-ys                        # height above the contact line (vertical px)
    sel=(A>0.02)&(xs>=xa)&(xs<xb)&(d>=-3)&(d<=DMAX)&(xs>=min(x0,x1)-70)&(xs<=max(x0,x1)+85)
    X=xs[sel].astype(np.float32);Y=ys[sel];D=np.clip(d[sel],0,None);W=A[sel]
    tx=X+SH*D                      # cast away from sun: to the right
    ty=cy[sel]+VS*D                # foreshortened toward the viewer (down)
    ix=np.clip(np.round(tx).astype(int),0,S-1);iy=np.clip(np.round(ty).astype(int),0,S-1)
    fade=(1-D/DMAX)**1.25
    np.add.at(acc,(iy,ix),W*fade);np.add.at(dist,(iy,ix),D*W*fade)
cov=np.clip(acc,0,1);dn=np.where(acc>1e-3,dist/np.maximum(acc,1e-3),0)/DMAX
sharp=cv2.GaussianBlur(cov,(0,0),2.2);soft=cv2.GaussianBlur(cov,(0,0),11)
t=np.clip(cv2.GaussianBlur(dn,(0,0),6)*1.6,0,1)
sh=(1-t)*sharp+t*soft
# ambient occlusion hugging the soles
ao=np.zeros((S,S),np.float32)
for (x0,y0),(x1,y1),_ in lines.values():
    cv2.line(ao,(x0,y0),(x1,y1),1.0,10,cv2.LINE_AA)
ao=cv2.GaussianBlur(ao,(0,0),7)*0.55
al=np.clip(np.maximum(sh*0.78,ao),0,0.85)
out=np.zeros((S,S,4),np.uint8);out[...,0]=42;out[...,1]=20;out[...,2]=5;out[...,3]=(al*255).astype(np.uint8)
Image.fromarray(out,'RGBA').save('source/shadow_wed.png');print('ok',al.max())
