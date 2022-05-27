from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import math


class SampleData:
    """
    une classe pour représenter toutes
    les informations de base d'un échantillon
    """

    def __init__(self, adress: Path):
        """
        extraction des infos à partir 
        de l'adresse du fichier, de format :
        [nom_classe]_[nom_pose]_[bordel]_[numéro_échantillon].jpg
        """
        self.adress = adress
        self.className, self.pose, otherData = adress.name.split("_", 2)

    def imData(self):
        """
        charge l'image en mémoire et retourne un objet 
        qui la représente (utilise la librairie de chargement 
        d'images PIL)
        """
        return Image.open(self.adress)


def read_database(rep: Path):
    """
    lit les données (des noms de fichiers)
    de la base de données avec un dictionnaire classes
    qui contient la liste des échantillons 
    de chaque classe, indexée par son nom
    """
    classes = defaultdict(list)
    for imgFile in rep.glob("*.jpg"):
        sample = SampleData(imgFile)
        classes[sample.className].append(sample)
    return classes


def split_database(db_data, p: float):
    """
    sépare la bdd en une base d'apprentissage
    et une base de validation

    p est la proportion d'échantillon qui vont
    dans la base d'apprentissage
    """

    learn_data = defaultdict(list)
    eval_data = defaultdict(list)

    for cName, c in db_data.items():
        nmax = math.ceil(len(c)*p)
        # modifie l'ordre des éléments de la classe mais pas grave
        np.random.shuffle(c)
        learn_data[cName] = c[:nmax-1]
        eval_data[cName] = c[nmax:]

    return learn_data, eval_data


db_dir = Path("./DB_RESIZED")

db_data = read_database(db_dir)

learn_data, eval_data = split_database(db_data, 0.5)


if __name__ == "__main__":
    # test : affichage des données

    classes = db_data.values()

    for c in classes:
        for sample in c:
            print(sample.className)
            print(sample.pose)
            print(sample.adress)

    for cName in db_data.keys():
        learn_c = learn_data[cName]
        validation_c = eval_data[cName]

        print("\nclasse : ", cName)

        print("apprentissage --------------")
        for sample in learn_c:
            print(sample.adress)

        print("validation ------------")
        for sample in validation_c:
            print(sample.adress)
