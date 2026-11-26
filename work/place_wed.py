import re,subprocess,numpy as np
from PIL import Image
P='source/nib-world-elite-debit.html'
def setcss(top,rib):
    h=open(P,encoding='utf-8').read()
    h=re.sub(r'(h1\{position:absolute;left:62px;)top:\d+px;(font-family:NokiaB;font-weight:400;)font-size:\d+px',r'\1top:%dpx;\2font-size:84px'%top,h)
    h=re.sub(r'(\.rib\{position:absolute;left:66px;)top:\d+px;','\\1top:%dpx;'%rib,h)
    h=re.sub(r'(font-family:NokiaB;)font-size:\d+px(;line-height:1;-webkit-text-stroke:1.2px #6e3a14;)padding:[^;]+;','\\1font-size:32px\\2padding:10px 32px 12px;',h)
    open(P,'w',encoding='utf-8').write(h)
def render():
    subprocess.run(['python3','source/render.py',P,'posts/nib-world-elite-debit.png'],check=True)
    return np.array(Image.open('posts/nib-world-elite-debit.png').convert('RGB')).astype(int)
def white_rows(a):
    reg=a[105:470,40:470]
    m=(reg.min(2)>245)
    rows=np.where(m.sum(1)>2)[0]+105
    return rows.min(),rows.max()
G=60
top=140
for _ in range(3):
    setcss(top,520);a=render();t,b=white_rows(a);print('headline visual',t,b)
    top+=(100+G)-t
setcss(top,520);a=render();t,b=white_rows(a)
# yellow ካርድ line extends headline; find lowest yellow row of h1 (honey #fdb913) in x 40-300 above y 470
reg=a[105:520,40:300];ym=(abs(reg[...,0]-253)<6)&(abs(reg[...,1]-185)<8)&(reg[...,2]<40)
yr=np.where(ym.sum(1)>3)[0]+105;print('yellow rows',yr.min(),yr.max())
bottom=max(b,yr.max());print('headline top/bottom',t,bottom,'top css',top)
pill_top=bottom+2*G;print('pill top',pill_top)
setcss(top,pill_top);a=render()
