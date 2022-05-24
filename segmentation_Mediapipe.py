import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


def segmentation_image(sample_img):
    change_background_mp = mp.solutions.selfie_segmentation
    
    change_bg_segment = change_background_mp.SelfieSegmentation()
    

    
    RGB_sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    
    result = change_bg_segment.process(RGB_sample_img)
    
    
    binary_mask = result.segmentation_mask > 0.9
    
    binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))
    
    output_image = np.where(binary_mask_3, sample_img, 255)    
    
    plt.figure(figsize=[22,22])
    plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    cv2.imwrite('Resultats_segmentation/image_segmentee.jpg',output_image)
    return output_image


sample_img = cv2.imread('DatabasePP/josuke_assis_1.jpg')
segmentation_image(sample_img)