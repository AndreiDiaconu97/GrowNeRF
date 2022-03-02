import cv2
import glob
import numpy as np
import os
import shutil
import torch
from torchmetrics.functional import ssim
import re

outDir = "_output"
root = "logs"
folders = [
    "_D4W64 vs D2 W128",
    "_baselines",
    "_boosting",
    "_DEBUG",
    "_DEBUG_activations_no_corrective",
    "_hierarchical",
    "_post_activation_sum",
    "_propagation",
    "_ratio",
    "_uncategorized",
    "_old\\_activation_fn",
    "_old\\_boosting",
    "_old\\_decay",
    "_old\\_hierarchical",
    "_old\\_lr",
    "_old\\_networks",
    "_old\\_ratio",
]

if __name__ == '__main__':

    for folder in folders:
        experiments = glob.glob(f"{root}\{folder}\*")

        for exp in experiments:

            # FIND LAST RESULTS
            ensVal_files = glob.glob(f"{exp}\\*ensVal*")
            # ensVal_names = [os.path.basename(x) for x in ensVal_files]
            # last_stage = max(map(lambda x: re.findall(r'\d+', x), ensVal_names))
            ensVal_files.sort(key=lambda x: [int(i) for i in re.findall(r'\d+', x)[-2:]])

            # COPY TO NEW FOLDER
            last_files = ensVal_files[-6:]
            new_files = []
            if not os.path.isdir(f"{exp}\\{outDir}"):
                os.mkdir(f"{exp}\\{outDir}")
            for f in last_files:
                new_files.append(f"{exp}\\{outDir}\\{os.path.basename(f)}")
                shutil.copyfile(f, new_files[-1])

            # LOAD IMAGES
            imgs = {
                "disp_coarse": {"path": new_files[0]},
                "disp_fine": {"path": new_files[1]},
                "grad": {"path": new_files[2]},
                "rgb_coarse": {"path": new_files[3]},
                "rgb_fine": {"path": new_files[4]},
                "target": {"path": new_files[5]}
            }

            for k in imgs:
                imgs[k]["data"] = cv2.imread(imgs[k]["path"]) / 255

            # SAVE ABSOLUTE ERROR IMAGES
            imgs["abs_error"] = {}
            imgs["abs_error"]["data"] = (np.abs(imgs["target"]["data"] - imgs["rgb_fine"]["data"]))
            imgs["abs_error"]["path"] = imgs["rgb_fine"]["path"].removesuffix("_rgb_fine.png") + "_absErr_fine.png"
            cv2.imwrite(imgs["abs_error"]["path"], imgs["abs_error"]["data"] * 255)

            # COMPUTE SSIM
            tmp_shape = [1, imgs["target"]["data"].shape[-1], *imgs["target"]["data"].shape[:2]]
            ssim_fine = ssim(torch.tensor(imgs["rgb_fine"]["data"]).view(tmp_shape), torch.tensor(imgs["target"]["data"]).view(tmp_shape))
            ssim_coarse = ssim(torch.tensor(imgs["rgb_coarse"]["data"]).view(tmp_shape), torch.tensor(imgs["target"]["data"]).view(tmp_shape))

            f = open(f"{exp}\\{outDir}\\SSIM.txt", "w")
            f.write(f"ssim_coarse:\t{ssim_coarse}\n")
            f.write(f"ssim_fine:\t\t{ssim_fine}")
            f.close()

            # STRETCH DEPTH MAPS
            alpha = 5.0
            beta = -1
            imgs["disp_coarse"]["data"] = alpha * imgs["disp_coarse"]["data"] + beta
            imgs["disp_fine"]["data"] = alpha * imgs["disp_fine"]["data"] + beta
            cv2.imwrite(imgs["disp_coarse"]["path"], imgs["disp_coarse"]["data"] * 255)
            cv2.imwrite(imgs["disp_fine"]["path"], imgs["disp_fine"]["data"] * 255)

            # SAVE CROPS
            rx = 185
            ry = 147
            s = 120
            for k in imgs:
                cv2.imwrite(os.path.dirname(imgs[k]["path"]) + "\\crop_" + os.path.basename(imgs[k]["path"]), imgs[k]["data"][rx:rx + s, ry:ry + s] * 255)
