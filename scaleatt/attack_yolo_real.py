import numpy as np, time
from PIL import Image
from scaling.CVScaler import CVScaler
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from attack.QuadrScaleAttack import QuadraticScaleAttack

BASE = "/home/zeinebdridi/projects/persec"
OUTD = BASE + "/attacks/scaling/out_yolo_real"
import os; os.makedirs(OUTD, exist_ok=True)

SRC, DST = 3840, 640   # 6x ratio, target = YOLO's real input size

def load(p,s): return np.asarray(Image.open(p).convert("RGB").resize((s,s), Image.LANCZOS), np.uint8)
cat = load(BASE+"/data/cat.png", SRC)
dog = load(BASE+"/data/dog.png", DST)
scaler = CVScaler(SuppScalingAlgorithms.LINEAR, cat.shape, dog.shape)

t0=time.time()
a,_,_ = QuadraticScaleAttack(eps=1, verbose=False).attack(src_image=cat, target_image=dog, scaler_approach=scaler)
a8=np.clip(a,0,255).astype(np.uint8); ds=scaler.scale_image(xin=a8)
Image.fromarray(a8).save(OUTD+"/attack_full.png")
Image.fromarray(np.clip(ds,0,255).astype(np.uint8)).save(OUTD+"/attack_downscaled.png")
Image.fromarray(cat).save(OUTD+"/source_cat.png")
def mae(x,y): return round(float(np.abs(x.astype(float)-y.astype(float)).mean()),2)
print("ratio", round(SRC/DST,1),"x took",round(time.time()-t0),"s")
print("full attack vs cat (want SMALL):", mae(a8,cat))
print("downscaled vs dog (want SMALL):", mae(ds,dog))