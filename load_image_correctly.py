from PIL import Image
from pathlib import Path
import numpy as np


def load_image_correctly(src:Path):
	return np.array(Image.open(src))