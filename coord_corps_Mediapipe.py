import numpy as np
import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



def coords_full_pose(image):
    with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
        
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return results


def coord_corps(image):
    """
    retourne uniquement les coordonéees intéressantes pour nous
    """
    # Liste des poses 
    pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
                  'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
                  'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    results=coords_full_pose(image)    

        
    # Ecriture des coordonnées dans une liste
    Coord=[]
    if results.pose_landmarks: 
        for i in range(len(pose_tubuh)):
            results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x * image.shape[0]
            results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[i].y * image.shape[1]
            
            pc_coord = [pose_tubuh[i] , results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y]
            
            Coord.append(pc_coord)
                

    return Coord


def body_mask(image):
    """
    utilise la magine (noire) de Mediapipe
    pour renvoyer un masque binaire
    issu de la segmentation du corps de l'utilisateur
    détecté sur la photo
    """
    results=coords_full_pose(image)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    conditionBin=condition.astype(np.uint8)
    res=conditionBin[:,:,1]
    return res


def display_coords(image):

    # For static images:
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192) # gray

    results=coords_full_pose(image)
    annotated_image = image.copy()


    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    
            
    
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('mp_im/image_annotee.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)



    # # Enregistrement de ces coordonnées dans un fichier txt
    # with open('Resultats/Coord.txt', 'w+') as f: 
    #     for items in Coord: 
    #         f.write('%s\n' %items) 
            
        
    #     print("File written successfully") 
    # f.close()

    cv2.imshow('MediaPipe Pose', cv2.flip(annotated_image, 1))
    cv2.waitKey(0)
    


# if __name__=="__main__" :
#     # TEST 
#     file_name = os.path.join(os.path.dirname(__file__), 'DB_RESIZED/jolyne_mainshanches_2.jpg')
#     assert os.path.exists(file_name)
#     img = cv2.imread(file_name, -1)

#     display_coords(img)

