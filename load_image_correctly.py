from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose as fix_orientation

def load_image_correctly(src:Path):
	'''
	charge les images en prenant 
	en compte leur orientation
	'''
	img = fix_orientation(Image.open(src))
	return img

