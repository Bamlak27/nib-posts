import cv2,numpy as np,sys
im=cv2.imread('/mnt/user-data/uploads/a9e810cf-image.png')
Q=np.array(eval(sys.argv[1]),dtype=np.float32)
o=im.copy()
cv2.polylines(o,[Q.astype(np.int32)],True,(0,0,255),1)
for p in Q: cv2.circle(o,tuple(int(v) for v in p),3,(255,0,0),-1)
c=o[480:1200,340:880]
cv2.imwrite('/tmp/q.png',cv2.resize(c,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC))
