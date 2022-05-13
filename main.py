from collections import defaultdict
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt



class SampleData:
    """
    une classe pour représenter toutes
    les informations d'un échantillon
    """

    def __init__(self, adress: Path):
        """
        extraction des infos à partir 
        de l'adresse du fichier, de format :
        [nom_classe]_[nom_pose]_[bordel]_[numéro_échantillon].jpg
        """
        self.adress = adress
        self.className, self.pose, otherData = adress.name.split("_", 2)


def parseImages(rep: Path):
	# un dictionnaire qui contiendra la liste des
	# échantillons de chaque classe, indexée par son nom
    classes = defaultdict(list)
    for imgFile in rep.glob("*.jpg"):
        plt.figure()
        plt.imshow(Image.open(imgFile))
        sample = SampleData(imgFile)
        classes[sample.className].append(sample)
    return classes


db_dir = Path("./DB_RESIZED")

classes = parseImages(db_dir)


for c in classes.values():
    for s in c:
        print(s.className)
        print(s.pose)
        print(s.adress)
