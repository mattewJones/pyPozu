import cv2
import numpy as np
from coord_corps_Mediapipe import body_mask

def SB_hist(image):
	chans = cv2.split(image)

	mask=np.array(body_mask(image),dtype=int)
	
	hist=cv2.calcHist(
		images=chans, 
		channels=[0,1,2], 
		mask=body_mask(image), 
		histSize=[4,4,4], #taille de l'histogramme S&B 
		ranges=[0,256,0,256,0,256] #syntaxe très zarb
	)
	return np.reshape(hist,-1) #conversion en vecteur




if __name__=="__main__" :
	img=cv2.imread('./DB_RESIZED/abbacchio_mainshanches_4.jpg')
	print(SB_hist(img))