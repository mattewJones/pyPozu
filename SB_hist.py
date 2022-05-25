import cv2


def SB_hist(image):
	chans = cv2.split(image)
	return cv2.calcHist(
		images=chans, 
		channels=[0,1,2], 
		mask=None, 
		histSize=[4,4,4], #taille de l'histogramme S&B 
		ranges=[0,256,0,256,0,256] #très zarb
	)



if __name__=="__main__" :
	img=cv2.imread('./DB_RESIZED/joseph_mainTete_1.jpg')
	print(SB_hist(img))