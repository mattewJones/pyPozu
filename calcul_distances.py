import numpy as np
import cv2
import mediapipe as mp
import os
from coord_corps_Mediapipe import *



def calcul_distances(Coord):
    '''Il faut rentrer en paramètre les coordonnées des différents joints du corps obtenue à l'aide 
    de la fonction coord_corps( arg = image)'''
    
    Mil_Hanche = [(Coord[24][1]+Coord[23][1])/2,(Coord[24][2]+Coord[23][2])/2]
    
    # distance de normalisation : nez-milieu des hanches
    Dist_Norm = np.sqrt((Coord[0][1]-Mil_Hanche[0])**2+(Coord[0][2]-Mil_Hanche[1])**2)
    
    Mil_Cheville = [(Coord[27][1]+Coord[28][1])/2,(Coord[27][2]+Coord[28][2])/2]
    
    # Distance nez-cheville
    D_Nez_Cheville = np.sqrt((Coord[0][1]-Mil_Cheville[0])**2+(Coord[0][1]-Mil_Cheville[1])**2)
     
    # Distance Nez-main gauche/droite
    D_Nez_MainG = np.sqrt((Coord[0][1]-Coord[15][1])**2+(Coord[0][2]-Coord[15][2])**2)
    D_Nez_MainD = np.sqrt((Coord[0][1]-Coord[16][1])**2+(Coord[0][2]-Coord[16][2])**2)
    
    # Distance Hanche-main gauche/droite
    D_Hanche_MainG = np.sqrt((Mil_Hanche[0]-Coord[15][1])**2+(Mil_Hanche[1]-Coord[15][2])**2)
    D_Hanche_MainD = np.sqrt((Mil_Hanche[0]-Coord[16][1])**2+(Mil_Hanche[1]-Coord[16][2])**2)
    
    # Distance épaules 
    D_Epaules = np.sqrt((Coord[11][1]-Coord[12][1])**2+(Coord[11][2]-Coord[12][2])**2)
    
    # Distances Coudes Hanches
    D_Hanche_CoudeG = np.sqrt((Mil_Hanche[0]-Coord[13][1])**2+(Mil_Hanche[1]-Coord[13][2])**2)
    D_Hanche_CoudeD = np.sqrt((Mil_Hanche[0]-Coord[14][1])**2+(Mil_Hanche[1]-Coord[14][2])**2)
    
    Dnc_norm = D_Nez_Cheville/Dist_Norm
    Dnmg_norm = D_Nez_MainG/Dist_Norm
    Dnmd_norm = D_Nez_MainD/Dist_Norm
    Dhmg_norm =  D_Hanche_MainG/Dist_Norm
    Dhmd_norm = D_Hanche_MainD/Dist_Norm
    De_norm = D_Epaules/Dist_Norm
    Dhcg_norm =  D_Hanche_CoudeG/Dist_Norm
    Dhcd_norm =  D_Hanche_CoudeD/Dist_Norm
    
    
    '''(Nez_Cheville, Nez_Main Gauche, Nez_Main Droite, Hanche_Main Gauche, Hanche_Main Droite, 
   Epaules, Coude Gauche_Hanche, Coude Droit_Hanche)'''
    
    return (Dnc_norm,Dnmg_norm,Dnmd_norm,Dhmg_norm,Dhmd_norm,De_norm, Dhcg_norm,Dhcd_norm )

    

if __name__=="__main__":
    """
    test
    """
    photo = 'DB_RESIZED/jolyne_mainshanches_2.jpg'
    img = cv2.imread(photo)
    coord_photo = coord_corps(img)
    print("#### retour ####")
    print(calcul_distances(coord_photo))
    