# -*- coding: utf-8 -*-

import cv2
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose as fix_orientation
import numpy as np


def resize_whole_database(srcDir: Path, destDir: Path, size=(400, 600)):
    for imgFile in srcDir.glob("*.jpg"):
    	# j'utilise PIl pour ouvrir les images parce que ça accepte
    	# les objets de Pathlib en entrée (plus simple)
        img = fix_orientation(Image.open(imgFile))
        imgArr = np.array(img)
        resArr = cv2.resize(imgArr, size, interpolation=cv2.INTER_AREA)
        res = Image.fromarray(resArr)
        destPath = destDir / imgFile.name
        res.save(destPath)


srcDir = Path(
    'C:/Users/lenovo/Desktop/Programmes extremmement bien codésIRONIE/vomi matlab/image/Pozu/DATABASE_final_nontraité/')
destDir = Path('./DB_RESIZED/')
resize_whole_database(srcDir, destDir)
