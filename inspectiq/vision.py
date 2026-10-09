import json,io,base64
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
LABELS=['clean','scratch','spot']
ROOT=Path(__file__).resolve().parents[1]
def sample(label,rng):
    im=Image.fromarray((np.clip(rng.normal(.65,.035,(32,32)),0,1)*255).astype('uint8')); d=ImageDraw.Draw(im)
    x,y=map(int,rng.integers(4,20,2))
    if label==1: d.line((x,y,x+int(rng.integers(5,12)),y+int(rng.integers(5,12))),fill=30,width=int(rng.integers(1,3)))
    if label==2:
        r=int(rng.integers(2,5)); d.ellipse((x,y,x+r,y+r),fill=30)
    return np.asarray(im,dtype=np.float32)/255
class CNN:
    def __init__(self,weights=None):
        if weights:
            for key,val in weights.items(): setattr(self,key,np.array(val))
        else:
            r=np.random.default_rng(7); self.k=r.normal(0,.25,(12,3,3)); self.b=np.zeros(12); self.w=r.normal(0,.1,(24,3)); self.c=np.zeros(3)
    def forward(self,x):
        patches=np.lib.stride_tricks.sliding_window_view(x,(3,3),axis=(1,2))
        z=np.einsum('nhwij,fij->nhwf',patches,self.k)+self.b; h=np.maximum(z,0)
        feat=np.concatenate([h.mean((1,2)),h.max((1,2))],axis=1); logits=feat@self.w+self.c
        e=np.exp(logits-logits.max(1,keepdims=True)); p=e/e.sum(1,keepdims=True)
        return p,(patches,z,h,feat)
    def step(self,x,y,lr=.15):
        p,(patches,z,h,feat)=self.forward(x); loss=-np.log(p[np.arange(len(y)),y]+1e-9).mean()
        dz=p.copy(); dz[np.arange(len(y)),y]-=1; dz/=len(y); df=dz@self.w.T
        dh=np.broadcast_to(df[:,None,None,:12]/900,h.shape).copy(); m=h==h.max((1,2),keepdims=True)
        dh+=m*df[:,None,None,12:]/m.sum((1,2),keepdims=True); dh*=z>0
        grads={'w':feat.T@dz,'c':dz.sum(0),'k':np.einsum('nhwf,nhwij->fij',dh,patches),'b':dh.sum((0,1,2))}
        for key,g in grads.items(): setattr(self,key,getattr(self,key)-lr*g)
        return float(loss)
    def save(self,path): Path(path).write_text(json.dumps({k:getattr(self,k).tolist() for k in ['k','b','w','c']}))
    @classmethod
    def load(cls): return cls(json.loads((ROOT/'artifacts/cnn.json').read_text()))
    def predict(self,im):
        x=np.asarray(im.convert('L').resize((32,32)),dtype=np.float32)/255; p,(_,_,h,_)=self.forward(x[None])
        heat=h[0].max(-1); heat=(heat-heat.min())/(np.ptp(heat)+1e-8)
        a=np.zeros((30,30,3),dtype=np.uint8); a[:,:,0]=(heat*255).astype('uint8'); a[:,:,2]=80
        buf=io.BytesIO(); Image.fromarray(a).resize((240,240)).save(buf,format='PNG')
        return {'label':LABELS[int(p.argmax())],'probabilities':dict(zip(LABELS,p[0].round(4).tolist())),'confidence':float(p.max()),'heatmap':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),'heatmap_method':'Maximum convolution activation, not Grad-CAM','model_scope':'Synthetic surface patterns only; unvalidated on real product photos.'}
