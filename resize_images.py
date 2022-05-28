# -*- coding: utf-8 -*-

import cv2
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose as fix_orientation
import numpy as np
from tkinter.filedialog import askdirectory, Tk
import pickle


def resize_image(src: Path, dest: Path, size=(400, 600)):
    # j'utilise PIl pour ouvrir les images parce que ça accepte
    # les objets de Pathlib en entrée (plus simple)
    img = fix_orientation(Image.open(src))
    imgArr = np.array(img)
    resArr = cv2.resize(imgArr, size, interpolation=cv2.INTER_AREA)
    res = Image.fromarray(resArr)
    res.save(dest)


def resize_whole_database(destDir: Path, size=(400, 600)):
    '''
    Va chercher toutes les images dans un répertoire externe srcDir,
    les redimensionne et les met dans destDir. 
    Si srcDir n'existe pas, le demande à l'utilisateur.
    ne fait par ailleurs le calcul que si srcDir a été modifié
    depuis la dernière fois que le calcul a été fait
    '''

    srcDirSave = Path('./dbSrcDirSave')
    lastUpdateDateSave = Path('./dbLastUpdateDateSave')

    try:
        assert srcDirSave.exists()
        with srcDirSave.open("rb") as f:
            srcDir = pickle.load(f)
        assert srcDir.exists()
    except:
        Tk().withdraw()  # cache la petite fenêtre de Tkinker
        srcDir = Path(askdirectory(
            title='Selectionner le dossier des images :'))
        with srcDirSave.open("wb") as f:
            pickle.dump(obj=srcDir, file=f)
        srcDirChanged = True
    else:
        srcDirChanged = False

    newUpdateDate = srcDir.stat().st_mtime

    try:
        assert lastUpdateDateSave.exists()
        with lastUpdateDateSave.open("rb") as f:
            lastUpdateDate = pickle.load(f)
    except:
        lastUpdateDate = newUpdateDate

    with lastUpdateDateSave.open("wb") as f:
        pickle.dump(file=f, obj=newUpdateDate)

    if srcDirChanged or (lastUpdateDate < newUpdateDate):
        for imgFile in srcDir.glob("*.jpg"):
            destPath = destDir / imgFile.name
            resize_image(imgFile, destPath)
    else :
        print("base de donnée prétraitée à jour, rien à faire")


destDir = Path('./DB_RESIZED/')
resize_whole_database(destDir)
