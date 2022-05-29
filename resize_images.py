# -*- coding: utf-8 -*-

import cv2
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose as fix_orientation
import numpy as np
from tkinter.filedialog import askdirectory, Tk
import pickle
import os


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
    dont l'adresse est enregistre dans ./dbSrcDirSave,
    les redimensionne et les met dans destDir. 
    Si srcDir n'existe pas, le demande à l'utilisateur.
    ne fait par ailleurs le calcul que si srcDir a été modifié
    depuis la dernière fois que le calcul a été fait
    '''

    srcDirSave = Path('./dbSrcDirSave')
    lastUpdateDateSave = Path('./dbLastUpdateDateSave')

    # vérification que srcDir est acessible, 
    # sinon demande à l'utilisateur de le màj
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
        srcDirUpdated = True
    else:
        srcDirUpdated = False

    # détection d'une màj du contenu du rep. source
    newUpdateDate = srcDir.stat().st_mtime
    try:
        assert lastUpdateDateSave.exists()
        with lastUpdateDateSave.open("rb") as f:
            lastUpdateDate = pickle.load(f)
    except:
        lastUpdateDate = newUpdateDate

    with lastUpdateDateSave.open("wb") as f:
        pickle.dump(file=f, obj=newUpdateDate)

    if srcDirUpdated or (lastUpdateDate < newUpdateDate):
        
        #nettoyage de destDir
        for imgFile in destDir.glob("*.jpg"):
            os.remove(imgFile)

        for imgFile in srcDir.glob("*.jpg"):
            destPath = destDir / imgFile.name
            resize_image(imgFile, destPath)
    else :
        print("base de donnée prétraitée à jour, rien à faire")


if __name__ == "__main__":
    destDir = Path('./DB_RESIZED/')
    resize_whole_database(destDir)
