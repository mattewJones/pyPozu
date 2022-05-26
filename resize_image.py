import cv2
from pathlib import Path


def resize_whole_database(srcDir: Path, destDir: Path, size=(600, 400)):
    srcDir = srcDir.absolute()
    destDir = destDir.absolute()
    # merci pour le support de pathlib opencv
    for imgFile in srcDir.glob("*.jpg"):
        img = cv2.imread(str(imgFile))
        resizedImg = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        destPath = destDir / imgFile.name
        cv2.imwrite(str(destPath), resizedImg)


srcDir = Path(
    'C:/Users/lenovo/Desktop/Programmes extremmement bien codésIRONIE/vomi matlab/image/Pozu')
destDir = Path('./DB_RESIZED/')
resize_whole_database(srcDir, destDir)
