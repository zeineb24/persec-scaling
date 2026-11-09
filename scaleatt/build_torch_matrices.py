import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

SRC = 512
DST = 224
OUT = "/home/zeinebdridi/projects/persec/attacks/scaling"

def torch_resize(img_hwc_uint8):
    t = torch.from_numpy(img_hwc_uint8).permute(2,0,1).float()
    r = F.resize(t, [DST, DST], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return r.permute(1,2,0).numpy()

# Vertical operator CL (DST x SRC): how output rows blend input rows
def recover_vertical():
    CL = np.zeros((DST, SRC))
    for i in range(SRC):
        imp = np.zeros((SRC, SRC, 3), np.float32)
        imp[i, :, 0] = 255.0
        out = torch_resize(imp.astype(np.uint8))[:, :, 0]
        CL[:, i] = out.mean(axis=1) / 255.0
    return CL

# Horizontal operator CR (SRC x DST): how output cols blend input cols
def recover_horizontal():
    CR = np.zeros((SRC, DST))
    for j in range(SRC):
        imp = np.zeros((SRC, SRC, 3), np.float32)
        imp[:, j, 0] = 255.0
        out = torch_resize(imp.astype(np.uint8))[:, :, 0]
        CR[j, :] = out.mean(axis=0) / 255.0
    return CR

CL = recover_vertical()
CR = recover_horizontal()
np.save(f"{OUT}/CL_torch.npy", CL)
np.save(f"{OUT}/CR_torch.npy", CR)
# verify: CL @ ones(SRC,SRC) @ CR should reproduce a resize of a flat image
test = np.full((SRC, SRC), 128.0)
approx = CL @ test @ CR
print("CL", CL.shape, "CR", CR.shape)
print("flat-image check (want ~128):", round(float(approx.mean()), 2))
print("saved matrices")