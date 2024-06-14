import cv2
import numpy as np
import os
import dlib
import sys
from imutils import face_utils
from scipy.spatial import Delaunay
import time
from moviepy.editor import VideoFileClip



def main(): # ex: python ~./faceChange.py man.jpg play.mp4
    start = time.time()
    args = sys.argv
    image_path = ""
    video_path = ""
    if len(args)==1:
        print("No image file and video file provided!")
        return
    elif len(args)==2:
        if(args[1][-3:]=="mp4"):
            video_path = os.path.join(os.getcwd(),args[1])
            if not os.path.exists(video_path):
                print("Video path doesn't exist!")
            print("No image file provided!")
        elif(args[1][-3:]==("jpg" or "png" or "jpeg")):
            image_path = os.path.join(os.getcwd(),args[1])
            if not os.path.exists(image_path):
                print("Image path doesn't exist!")
            print("No video file provided!")
        else:
            print("No image file and video file provided!")
        return
    elif len(args)==3:
        if ((args[1][-3:] in ("jpg","png")) or (args[1][-4:]=="jpeg")) and args[2][-3:]=="mp4":
            image_path = os.path.join(os.getcwd(),args[1])
            video_path = os.path.join(os.getcwd(),args[2])

            if not os.path.exists(image_path):
                print("Image path doesn't exist!")
                return
            if not os.path.exists(video_path):
                print("Video path doesn't exist!")
                return


    #-----------------------image-----------------------
    detector = dlib.get_frontal_face_detector()
    #detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    src_face = face_cascade.detectMultiScale(gray_img, 1.1, 4, minSize=(30,30), flags=cv2.CASCADE_FIND_BIGGEST_OBJECT)
    x_1, y_1, w_1, h_1 = src_face[0]
    image = image[y_1:y_1+h_1, x_1:x_1+w_1]

    if len(src_face) == 0:
        print("No face detected in provide image!")
        return
    elif len(src_face) > 1:
        print("Only need 1 face!") 
        #return


    output = []
    #-----------------------video-----------------------
    cap = cv2.VideoCapture(video_path)
    num=1
    while cap.isOpened():
        frame_start = time.time()
        ret, frame = cap.read() #type(frame)=numpy.ndarray
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        output_frame = frame.copy()
        gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for face in faces:
            x_2, y_2, w_2, h_2 = face
            tar_face = output_frame[y_2:y_2+h_2, x_2:x_2+w_2]

            r, g, b = cv2.split(tar_face)
            tar_brightness = np.mean(0.299 * r + 0.587 * g + 0.114 * b)

            gray_tar_face = cv2.cvtColor(tar_face, cv2.COLOR_BGR2GRAY)
            detected = detector(gray_tar_face, 2)
            if len(detected)==0:
                continue
            sh = predictor(gray_tar_face, detected[0])
            tar_dots = face_utils.shape_to_np(sh) #(68, 2)
            x2, y2, w2, h2 = cv2.boundingRect(tar_dots)


            #1. src_img
            warped_tri = []
            warped = cv2.resize(image, (w_2, h_2))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            detected = detector(warped_gray, 2)#2 for smaller or blurred faces
            # print(f"num={num},(w_2, h_2)= {(w_2, h_2)} ")
            if len(detected)==0:continue
            
            sh = predictor(warped_gray, detected[0])
            src_dots = face_utils.shape_to_np(sh) #(68, 2)
            x1, y1, w1, h1 = cv2.boundingRect(src_dots)

            src_indices = Delaunay(src_dots).simplices
            for idx in src_indices:
                warped_tri.append([src_dots[idx[0]],src_dots[idx[1]],src_dots[idx[2]]])
            warped_tri = np.array(warped_tri)


            #2. video_frame
            tar_tri = []
            for idx in src_indices:
                tar_tri.append([tar_dots[idx[0]],tar_dots[idx[1]],tar_dots[idx[2]]])
            tar_tri = np.array(tar_tri)

            min_len = min(len(warped_tri), len(tar_tri))
            sorted_indices = sorted(range(min_len), key=lambda i: cv2.boundingRect(warped_tri[i])[1])
            warped_tri = np.float32([warped_tri[i] for i in sorted_indices]) #.astype(warped_tri.dtype)
            tar_tri =  np.float32([tar_tri[i] for i in sorted_indices]) #.astype(warped_tri.dtype)

            
            for i in range(min_len):
                points = np.array([[tar_tri[i][0][0], tar_tri[i][0][1]],
                                  [tar_tri[i][1][0], tar_tri[i][1][1]],
                                  [tar_tri[i][2][0], tar_tri[i][2][1]]], np.int32) #np.int32

                src_mask = np.zeros((warped.shape[0], warped.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(src_mask, points, 255)

                tar_mask = np.ones((warped.shape[0], warped.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(tar_mask, points, 0)

                M = cv2.getAffineTransform(warped_tri[i], tar_tri[i])
                src_ = cv2.warpAffine(warped, M, (warped.shape[1],warped.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                
                #TODO:
                r, g, b = cv2.split(src_)
                src_brightness = np.mean(0.299 * r + 0.587 * g + 0.114 * b)
                src_ = cv2.convertScaleAbs(src_, alpha=0.8, beta=abs((tar_brightness-src_brightness)*(7/6)))
                src_ = cv2.bitwise_and(src_, src_, mask=src_mask)
                tar_face = cv2.bitwise_and(output_frame[y_2:y_2+h_2, x_2:x_2+w_2], output_frame[y_2:y_2+h_2, x_2:x_2+w_2], mask=tar_mask)
                final = cv2.bitwise_or(src_, tar_face)
                output_frame[y_2:y_2+h_2, x_2:x_2+w_2] = final

        output.append(output_frame)
        end = time.time()
        current_used_time = round(end-frame_start,2)
        print(f"frame {num} finished! frame processing time:{current_used_time}")
        num+=1



    output = np.array(output)
    height, width, layers = output[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_name = "temp.mp4"
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    for img in output:
        out.write(img)
    out.release()
    cv2.destroyAllWindows()


    # take sound from video
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio

    output_path = os.path.join(os.getcwd(),output_name)
    final_video = VideoFileClip(output_path)
    final_video = final_video.set_audio(audio_clip)
    output_name = args[2].split(".")[0]+"_output.mp4"
    final_video.write_videofile(output_name, codec="libx264", audio_codec="aac")
    os.remove("temp.mp4")

    end = time.time()
    print("\ntotal processing time : ",round(end-start,2)," sec")



if __name__ == "__main__":
    main()
