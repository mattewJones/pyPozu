import cv2

def resize_image(img,scale):
    # Parametres de reduction de l'image
    scale_percent = scale  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
      
    # resize image
    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return image