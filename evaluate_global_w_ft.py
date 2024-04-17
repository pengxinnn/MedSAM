import os
from os import listdir, makedirs
from os.path import join, isfile, basename, dirname, isdir
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
import nibabel as nb
import multiprocessing as mp

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from matplotlib import pyplot as plt
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('--testdata_path', type=str,default = '/home/xinpeng/MedSAM/data/npy/CT_Abd_test')
parser.add_argument('--model_path', type=str,default = '/home/xinpeng/MedSAM/work_dir/SlimSAM-Global-50-Gaussian-20240416-0659/medsam_model_best.pth')
parser.add_argument('--reference_model_path', type=str,default = '/home/xinpeng/SlimSAM/checkpoints/SlimSAM-50.pth')
parser.add_argument('--output_save_path', type=str,default = '/home/xinpeng/MedSAM/results')
args, unparsed = parser.parse_known_args()

neighbour_code_to_normals = [
  [[0,0,0]],
  [[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125]],
  [[-0.5,0.0,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[0.5,0.0,0.0]],
  [[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,0.25],[-0.125,-0.125,-0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.375,0.375,0.375],[0.0,-0.25,0.25],[-0.25,0.0,0.25]],
  [[0.125,-0.125,-0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.375,0.375,0.375],[0.0,0.25,-0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.125,0.125,0.125]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25]],
  [[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25]],
  [[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,0.375],[-0.0,0.25,0.25],[0.125,0.125,-0.125],[-0.25,-0.0,-0.25]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,0.25,-0.25],[0.25,0.25,-0.25],[0.125,0.125,-0.125],[-0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125],[0.125,-0.125,0.125]],
  [[0.0,0.25,-0.25],[0.375,-0.375,-0.375],[-0.125,0.125,0.125],[0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,0.5,0.0],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.0,0.5,0.0],[0.125,-0.125,0.125],[-0.25,0.25,-0.25]],
  [[0.0,0.5,0.0],[0.0,-0.5,0.0]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,-0.375],[-0.25,0.0,0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.125,0.125,0.125],[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[-0.125,0.125,0.125],[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.375,0.375,-0.375],[-0.25,-0.25,0.0],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,-0.25,-0.25],[-0.125,0.125,-0.125],[0.25,0.25,0.0]],
  [[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.25,-0.25],[-0.25,0.25,-0.25],[-0.125,0.125,-0.125],[-0.125,0.125,-0.125]],
  [[-0.25,0.0,-0.25],[0.375,-0.375,-0.375],[0.0,0.25,-0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.0,0.0,0.5],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.25,-0.0,-0.25],[-0.375,0.375,0.375],[-0.25,-0.25,0.0],[-0.125,0.125,0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25],[-0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[0.125,-0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.25,0.0,0.25],[-0.375,-0.375,0.375],[-0.25,0.25,0.0],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25],[0.125,-0.125,-0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.5,0.0,-0.0],[0.25,-0.25,-0.25],[0.125,-0.125,-0.125]],
  [[-0.25,0.25,0.25],[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.375,-0.375,0.375],[0.0,0.25,0.25],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,-0.5,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.375,-0.375,0.375],[0.25,-0.25,0.0],[0.0,0.25,0.25],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.125,0.125,0.125],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.375,-0.375,0.375],[0.25,-0.25,0.0],[0.0,0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,0.25,0.25],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[-0.25,0.25,0.25],[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.5,0.0,-0.0],[0.25,-0.25,-0.25],[0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.25,0.0,0.25],[-0.375,-0.375,0.375],[-0.25,0.25,0.0],[-0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25],[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.0,0.0,0.5],[0.0,0.0,0.5]],
  [[0.0,0.0,-0.5],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[-0.375,0.375,0.375],[-0.25,-0.25,0.0],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.0,0.0,0.5],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[0.5,0.0,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.0,-0.25],[0.375,-0.375,-0.375],[0.0,0.25,-0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.25,-0.25],[-0.25,0.25,-0.25],[-0.125,0.125,-0.125],[-0.125,0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,-0.25,-0.25],[-0.125,0.125,-0.125],[0.25,0.25,0.0]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25]],
  [[-0.125,-0.125,0.125],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[0.125,0.125,0.125],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.375,0.375,-0.375],[-0.25,-0.25,0.0],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[-0.375,-0.375,-0.375],[-0.25,0.0,0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.125,-0.125,0.125]],
  [[0.0,0.5,0.0],[0.0,-0.5,0.0]],
  [[0.0,0.5,0.0],[0.125,-0.125,0.125],[-0.25,0.25,-0.25]],
  [[0.0,0.5,0.0],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,0.25,-0.25],[0.375,-0.375,-0.375],[-0.125,0.125,0.125],[0.25,0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.25,0.25,-0.25],[0.25,0.25,-0.25],[0.125,0.125,-0.125],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,0.375],[-0.0,0.25,0.25],[0.125,0.125,-0.125],[-0.25,-0.0,-0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.125,0.125,0.125]],
  [[0.375,0.375,0.375],[0.0,0.25,-0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.125,-0.125,-0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.125,0.125,0.125],[0.375,0.375,0.375],[0.0,-0.25,0.25],[-0.25,0.0,0.25]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,0.25],[-0.125,-0.125,-0.125]],
  [[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25]],
  [[0.125,-0.125,-0.125]],
  [[0.5,0.0,0.0],[0.5,0.0,0.0]],
  [[-0.5,0.0,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125]],
  [[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0,0,0]]]

def resize_longest_side(image, target_length=1024):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=1024):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

def resize_box_to_1024(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 1024 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou

def get_bbox1024(mask_1024, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_1024 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_1024.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes1024 = np.array([x_min, y_min, x_max, y_max])

    return bboxes1024

def resize_box_to_1024(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 1024 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

# MedSAM_infer_npz_2D
def MedSAM_infer_npz_2D(img_npz_file, medsam_model, device):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W, _ = img_3c.shape
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    img_1024 = resize_longest_side(img_3c, 1024)
    newh, neww = img_1024.shape[:2]
    img_1024_norm = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_padded = pad_image(img_1024_norm, 1024)
    img_1024_tensor = torch.tensor(img_1024_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)

    for idx, box in enumerate(boxes, start=1):
        box1024 = resize_box_to_1024(box, original_size=(H, W))
        box1024 = box1024[None, ...] # (1, 4)
        medsam_mask, iou_pred = medsam_inference(medsam_model, image_embedding, box1024, (newh, neww), (H, W))
        segs[medsam_mask>0] = idx
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')

    np.savez_compressed(
        join(args.testdata_path, 'segs', npz_name),
        segs=segs,
    )

# MedSAM_infer_npz_3D
def MedSAM_infer_npz_3D(img_npz_file, medsam_model, device):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8)
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_1024 = resize_longest_side(img_3c, 1024)
            new_H, new_W = img_1024.shape[:2]

            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 1024*1024
            img_1024 = pad_image(img_1024)

            # convert the shape to (3, H, W)
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_1024 = resize_box_to_1024(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs[z-1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg1024 = resize_longest_side(pre_seg)
                    pre_seg1024 = pad_image(pre_seg1024)
                    box_1024 = get_bbox1024(pre_seg1024)
                else:
                    box_1024 = resize_box_to_1024(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(medsam_model, image_embedding, box_1024, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_1024 = resize_longest_side(img_3c)
            new_H, new_W = img_1024.shape[:2]

            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_1024 = pad_image(img_1024)

            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)

            pre_seg = segs[z+1, :, :]
            if np.max(pre_seg) > 0:
                pre_seg1024 = resize_longest_side(pre_seg)
                pre_seg1024 = pad_image(pre_seg1024)
                box_1024 = get_bbox1024(pre_seg1024)
            else:
                scale_1024 = 1024 / max(H, W)
                box_1024 = mid_slice_bbox_2d * scale_1024
            img_2d_seg, iou_pred = medsam_inference(medsam_model, image_embedding, box_1024, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        segs[segs_3d_temp>0] = idx
    np.savez_compressed(
        join(args.testdata_path, 'segs', npz_name),
        segs=segs)

def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
  """Compute closest distances from all surface points to the other surface.

  Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
  the predicted mask `mask_pred`, computes their area in mm^2 and the distance
  to the closest point on the other surface. It returns two sorted lists of
  distances together with the corresponding surfel areas. If one of the masks
  is empty, the corresponding lists are empty and all distances in the other
  list are `inf`

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
        direction

  Returns:
    A dict with
    "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
        from all ground truth surface elements to the predicted surface,
        sorted from smallest to largest
    "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
        from all predicted surface elements to the ground truth surface,
        sorted from smallest to largest
    "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of
        the ground truth surface elements in the same order as
        distances_gt_to_pred
    "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of
        the predicted surface elements in the same order as
        distances_pred_to_gt

  """

  # compute the area for all 256 possible surface elements
  # (given a 2x2x2 neighbourhood) according to the spacing_mm
  neighbour_code_to_surface_area = np.zeros([256])
  for code in range(256):
    normals = np.array(neighbour_code_to_normals[code])
    sum_area = 0
    for normal_idx in range(normals.shape[0]):
      # normal vector
      n = np.zeros([3])
      n[0] = normals[normal_idx,0] * spacing_mm[1] * spacing_mm[2]
      n[1] = normals[normal_idx,1] * spacing_mm[0] * spacing_mm[2]
      n[2] = normals[normal_idx,2] * spacing_mm[0] * spacing_mm[1]
      area = np.linalg.norm(n)
      sum_area += area
    neighbour_code_to_surface_area[code] = sum_area

  # compute the bounding box of the masks to trim
  # the volume to the smallest possible processing subvolume
  mask_all = mask_gt | mask_pred
  bbox_min = np.zeros(3, np.int64)
  bbox_max = np.zeros(3, np.int64)

  # max projection to the x0-axis
  proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
  idx_nonzero_0 = np.nonzero(proj_0)[0]
  if len(idx_nonzero_0) == 0:
    return {"distances_gt_to_pred":  np.array([]),
            "distances_pred_to_gt":  np.array([]),
            "surfel_areas_gt":       np.array([]),
            "surfel_areas_pred":     np.array([])}

  bbox_min[0] = np.min(idx_nonzero_0)
  bbox_max[0] = np.max(idx_nonzero_0)

  # max projection to the x1-axis
  proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
  idx_nonzero_1 = np.nonzero(proj_1)[0]
  bbox_min[1] = np.min(idx_nonzero_1)
  bbox_max[1] = np.max(idx_nonzero_1)

  # max projection to the x2-axis
  proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
  idx_nonzero_2 = np.nonzero(proj_2)[0]
  bbox_min[2] = np.min(idx_nonzero_2)
  bbox_max[2] = np.max(idx_nonzero_2)

#  print("bounding box min = {}".format(bbox_min))
#  print("bounding box max = {}".format(bbox_max))

  # crop the processing subvolume.
  # we need to zeropad the cropped region with 1 voxel at the lower,
  # the right and the back side. This is required to obtain the "full"
  # convolution result with the 2x2x2 kernel
  cropmask_gt = np.zeros((bbox_max - bbox_min)+2, np.uint8)
  cropmask_pred = np.zeros((bbox_max - bbox_min)+2, np.uint8)

  cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0]+1,
                                          bbox_min[1]:bbox_max[1]+1,
                                          bbox_min[2]:bbox_max[2]+1]

  cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0]+1,
                                              bbox_min[1]:bbox_max[1]+1,
                                              bbox_min[2]:bbox_max[2]+1]

  # compute the neighbour code (local binary pattern) for each voxel
  # the resultsing arrays are spacially shifted by minus half a voxel in each axis.
  # i.e. the points are located at the corners of the original voxels
  kernel = np.array([[[128,64],
                      [32,16]],
                     [[8,4],
                      [2,1]]])
  neighbour_code_map_gt = scipy.ndimage.filters.correlate(cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
  neighbour_code_map_pred = scipy.ndimage.filters.correlate(cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

  # create masks with the surface voxels
  borders_gt   = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
  borders_pred = ((neighbour_code_map_pred != 0) & (neighbour_code_map_pred != 255))

  # compute the distance transform (closest distance of each voxel to the surface voxels)
  if borders_gt.any():
    distmap_gt = scipy.ndimage.morphology.distance_transform_edt(~borders_gt, sampling=spacing_mm)
  else:
    distmap_gt = np.Inf * np.ones(borders_gt.shape)

  if borders_pred.any():
    distmap_pred = scipy.ndimage.morphology.distance_transform_edt(~borders_pred, sampling=spacing_mm)
  else:
    distmap_pred = np.Inf * np.ones(borders_pred.shape)

  # compute the area of each surface element
  surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
  surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

  # create a list of all surface elements with distance and area
  distances_gt_to_pred = distmap_pred[borders_gt]
  distances_pred_to_gt = distmap_gt[borders_pred]
  surfel_areas_gt   = surface_area_map_gt[borders_gt]
  surfel_areas_pred = surface_area_map_pred[borders_pred]

  # sort them by distance
  if distances_gt_to_pred.shape != (0,):
    sorted_surfels_gt = np.array(sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
    distances_gt_to_pred = sorted_surfels_gt[:,0]
    surfel_areas_gt      = sorted_surfels_gt[:,1]

  if distances_pred_to_gt.shape != (0,):
    sorted_surfels_pred = np.array(sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
    distances_pred_to_gt = sorted_surfels_pred[:,0]
    surfel_areas_pred    = sorted_surfels_pred[:,1]


  return {"distances_gt_to_pred":  distances_gt_to_pred,
          "distances_pred_to_gt":  distances_pred_to_gt,
          "surfel_areas_gt":       surfel_areas_gt,
          "surfel_areas_pred":     surfel_areas_pred}


def compute_average_surface_distance(surface_distances):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  average_distance_gt_to_pred = np.sum( distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
  average_distance_pred_to_gt = np.sum( distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
  return (average_distance_gt_to_pred, average_distance_pred_to_gt)

def compute_robust_hausdorff(surface_distances, percent):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  if len(distances_gt_to_pred) > 0:
    surfel_areas_cum_gt   = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
    idx = np.searchsorted(surfel_areas_cum_gt, percent/100.0)
    perc_distance_gt_to_pred = distances_gt_to_pred[min(idx, len(distances_gt_to_pred)-1)]
  else:
    perc_distance_gt_to_pred = np.Inf

  if len(distances_pred_to_gt) > 0:
    surfel_areas_cum_pred = np.cumsum(surfel_areas_pred) / np.sum(surfel_areas_pred)
    idx = np.searchsorted(surfel_areas_cum_pred, percent/100.0)
    perc_distance_pred_to_gt = distances_pred_to_gt[min(idx, len(distances_pred_to_gt)-1)]
  else:
    perc_distance_pred_to_gt = np.Inf

  return max( perc_distance_gt_to_pred, perc_distance_pred_to_gt)

def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  rel_overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) / np.sum(surfel_areas_gt)
  rel_overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) / np.sum(surfel_areas_pred)
  return (rel_overlap_gt, rel_overlap_pred)

def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
  overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
  surface_dice = (overlap_gt + overlap_pred) / (
      np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
  return surface_dice


def compute_dice_coefficient(mask_gt, mask_pred):
  """Compute soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum

def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)

def compute_metrics(npz_name):
    metric_dict = {'dsc': -1.}
    if True:
        metric_dict['nsd'] = -1.
    npz_seg = np.load(join(args.testdata_path, 'segs', npz_name), allow_pickle=True, mmap_mode='r')
    npz_gt = np.load(join(args.testdata_path, 'gts', npz_name), allow_pickle=True, mmap_mode='r')
    gts = npz_gt['gts']
    segs = npz_seg['segs']
    if npz_name.startswith('3D'):
        spacing = npz_gt['spacing']

    dsc = compute_multi_class_dsc(gts, segs)
    # comupute nsd
    if True:
        if dsc > 0.2:
        # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
            if npz_name.startswith('3D'):
                nsd = compute_multi_class_nsd(gts, segs, spacing)
            else:
                spacing = [1.0, 1.0, 1.0]
                nsd = compute_multi_class_nsd(np.expand_dims(gts, -1), np.expand_dims(segs, -1), spacing)
        else:
            nsd = 0.0
    return npz_name, dsc, nsd

def main():
    img_npz_files = sorted(glob(join(args.testdata_path, 'imgs', '*.npz'), recursive=True))
    SlimSAM_model_fine_tuned = torch.load(args.model_path)
    SlimSAM_model = torch.load(args.reference_model_path)
    original_state_dict = SlimSAM_model_fine_tuned['model']
    modified_state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith('image_encoder.'):
            modified_key = 'image_encoder.module.' + key[len('image_encoder.'):]
        else:
            modified_key = key
        modified_state_dict[modified_key] = value

    # SlimSAM_model_pretrained is now the new fine-tuned SlimSAM global model
    SlimSAM_model.load_state_dict(modified_state_dict)

    SlimSAM_model.image_encoder = SlimSAM_model.image_encoder.module
    modelname_with_extension = os.path.basename(args.model_path)
    model_name, _ = os.path.splitext(modelname_with_extension)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x,qkv_emb,mid_emb,x_emb = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x

    import types
    funcType = types.MethodType
    SlimSAM_model.image_encoder.forward = funcType(forward, SlimSAM_model.image_encoder)

    # device = "cuda:0"
    device = "cpu"
    medsam_model = SlimSAM_model.to(device)
    medsam_model.eval()

    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if basename(img_npz_file).startswith('3D'):
            MedSAM_infer_npz_3D(img_npz_file, medsam_model, device)
        else:
            MedSAM_infer_npz_2D(img_npz_file, medsam_model, device)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(args.output_save_path, f"efficiency_{model_name}.csv"), index=False)

    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    seg_metrics['dsc'] = []
    seg_metrics['nsd'] = []
    npz_names = listdir(join(args.testdata_path, 'gts'))
    npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
    with mp.Pool(4) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            for i, (npz_name, dsc, nsd) in enumerate(pool.imap_unordered(compute_metrics, npz_names)):
                seg_metrics['case'].append(npz_name)
                seg_metrics['dsc'].append(np.round(dsc, 4))
                if True:
                    seg_metrics['nsd'].append(np.round(nsd, 4))
                pbar.update()
    df = pd.DataFrame(seg_metrics)
    # rank based on case column
    df = df.sort_values(by=['case'])
    df.to_csv(join(args.output_save_path, f"metrics_{model_name}.csv"), index=False)

if __name__ == '__main__':
    main()
