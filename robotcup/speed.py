import time
import os
# from os.path import dirname

import torch
import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

# VOC_BASE_DIR = dirname(dirname(os.path.realpath(__file__))) + "/__collection__/VOCdevkit/VOC2007"

# VAL_TXT = f"{VOC_BASE_DIR}/ImageSets/Main/val.txt"

IMG_DIR = "/home/vtol/src/YOLOv6/VOCdevkit/images/val"

def time_sync():
    '''Waits for all kernels in all streams on a CUDA device to complete if cuda is available.'''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time() * 1000

def main():
    print("Running speed test...\n")

    print(" - Stage 1: Loading model")
    start_t = time_sync()
    model = YOLO()
    end_t = time_sync()
    print(f"    Time used = {end_t - start_t:.3f}ms")

    print(" - Stage 2: Inference")

    print("    Warming up...")
    for _ in range(10):
        img = Image.fromarray(np.zeros((600, 600, 3), dtype=np.uint8))
        model.detect_image(img, False, draw_img=False)

    print("    Running model on val...")
    # run model on all images in `IMG_DIR`
    total = 0
    total_t = 0
    max_t = 0
    min_t = 100000
    # with open(VAL_TXT, 'r') as f:
    #     paths = f.readlines()
    #     for path in paths:
    #         path = path.strip()
    #         if path == "":
    #             continue
    for filename in os.scandir(IMG_DIR):
        if not filename.is_file():
            continue

        path = filename.path
        total += 1

        img = Image.open(path)
        # width, height = img.size
        # box = (100, 100, 550, 350)
        # region = img.crop(box)

        start_t = time_sync()
        model.detect_image(img, False, draw_img=False)
        end_t = time_sync()
        t = end_t - start_t
        total_t += t
        max_t = max(max_t, t)
        min_t = min(min_t, t)
    avg_t = total_t / total
    print(f"    Total time = {total_t:.3f}ms")
    print(f"    Avg time = {avg_t:.3f}ms")
    print(f"    Max time = {max_t:.3f}ms")
    print(f"    Min time = {min_t:.3f}ms")

    print("\nSpeed test completed.")

if __name__ == "__main__":
    main()
