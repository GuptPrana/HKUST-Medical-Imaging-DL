import torch
import numpy as np
import math
from glob import glob

from dataset import read_h5


if __name__ == '__main__':
    model = None # load your model here
    patch_size = (112, 112, 80)
    stride_xy = 18
    stride_z = 4

    path_list = glob('./datas/test/*.h5')
    model.eval()
    for path in path_list:
        image, label = read_h5(path)

        w, h, d = image.shape
        sx = math.ceil((w - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((h - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((d - patch_size[2]) / stride_z) + 1

        scores = np.zeros((2, ) + image.shape).astype(np.float32)
        counts = np.zeros(image.shape).astype(np.float32)
        
        # inference all windows (patches)
        for x in range(0, sx):
            xs = min(stride_xy * x, w - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, h - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, d - patch_size[2])

                    # extract one patch for model inference
                    test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    with torch.no_grad():
                        test_patch = torch.from_numpy(test_patch).cuda() # if use cuda
                        test_patch = test_patch.unsqueeze(0).unsqueeze(0) # [1, 1, w, h, d]
                        out = model(test_patch)
                        out = torch.softmax(out, dim=1)
                        out = out.cpu().data.numpy() # [1, 2, w, h, d]
                    
                    # record the predicted scores
                    scores[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out[0, ...]
                    counts[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
        
        scores = scores / np.expand_dims(counts, axis=0)
        predictions = np.argmax(scores, axis = 0) # final prediction: [w, h, d]
        metrics = None
