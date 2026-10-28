"""The train tests never use the GPU (training may run on it): hide CUDA before torch loads."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
