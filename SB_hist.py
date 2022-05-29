import cv2
import numpy as np
from coord_corps_Mediapipe import body_mask

def SB_hist(image):
	chans = cv2.split(image)

	mask=body_mask(image)
	
	hist=cv2.calcHist(
		images=chans, 
		channels=[0,1,2], 
		mask=mask,
		histSize=[4,4,4], #taille de l'histogramme S&B 
		ranges=[0,256,0,256,0,256] #syntaxe très zarb
	)
	return np.reshape(hist,-1) #conversion en vecteur




if __name__=="__main__" :
	img=cv2.imread('./DB_RESIZED/abbacchio_mainshanches_4.jpg')
	print(SB_hist(img))