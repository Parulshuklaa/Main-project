import json,time
from pathlib import Path
import numpy as np
from inspectiq.vision import CNN,sample,LABELS
def dataset(n,seed):
    r=np.random.default_rng(seed); y=np.arange(n)%3
    return np.stack([sample(int(v),r) for v in y]),y
def main():
    start=time.time(); x,y=dataset(900,11); vx,vy=dataset(300,22); tx,ty=dataset(300,33); model=CNN(); r=np.random.default_rng(44); best=-1
    for epoch in range(55):
        order=r.permutation(len(y)); losses=[]
        for i in range(0,len(y),45):
            idx=order[i:i+45]; losses.append(model.step(x[idx],y[idx]))
        vp,_=model.forward(vx); acc=float((vp.argmax(1)==vy).mean())
        if acc>best: best=acc; model.save('artifacts/cnn.json')
        if epoch%10==0: print(epoch,round(np.mean(losses),3),acc,flush=True)
    p,_=CNN.load().forward(tx); pred=p.argmax(1); cm=np.zeros((3,3),int)
    for a,b in zip(ty,pred): cm[a,b]+=1
    metrics={'dataset':'Procedural synthetic surfaces; independent seeds, same generator distribution','train_count':900,'validation_count':300,'test_count':300,'test_accuracy':float((pred==ty).mean()),'validation_accuracy':best,'confusion_matrix':cm.tolist(),'labels':LABELS,'seconds':round(time.time()-start,2),'architecture':'3x3 Conv(12) -> ReLU -> mean + max pooling -> Dense(3) -> Softmax','limitations':['No real-image validation','Softmax scores are uncalibrated','Same-generator evaluation may overestimate generalization','Activation map is not a causal explanation']}
    Path('artifacts/metrics.json').write_text(json.dumps(metrics,indent=2)); print(metrics)
if __name__=='__main__': main()
