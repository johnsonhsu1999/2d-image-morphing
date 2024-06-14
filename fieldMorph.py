import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw
import cv2
import numpy as np
import dlib
import sys
import os

class App:
    def __init__(self, root):
        self.left_image = None
        self.middle_image = None
        self.right_image = None
        self.left_start_point = None
        self.right_start_point = None
        self.left_sketch_mode = False
        self.right_sketch_mode = False
        self.PiQi_left = []
        self.PiQi_right = []

        self.left_label = tk.Label(root, text="image1", font=("Georgia", 25))
        self.left_label.place(x=167, y=30)

        self.right_label = tk.Label(root, text="image2", font=("Georgia", 25))
        self.right_label.place(x=860, y=30)

        self.middle_label = tk.Label(root, text="morphed", font=("Georgia", 25))
        self.middle_label.place(x=509, y=30)

        self.left_frame = tk.Frame(root, width=310, height=310, bd=2, relief=tk.GROOVE)
        self.left_frame.place(x=50, y=70)

        self.middle_frame = tk.Frame(root, width=310, height=310, bd=2, relief=tk.RIDGE)
        self.middle_frame.place(x=400, y=70)

        self.right_frame = tk.Frame(root, width=310, height=310, bd=2, relief=tk.GROOVE)
        self.right_frame.place(x=750, y=70)

        # load img
        self.left_button = tk.Button(root, text="load img1", command=lambda: self.load_image("left"), height=2, width=10, fg="#63caf9",cursor="heart")
        self.left_button.place(x=78, y=390)

        self.right_button = tk.Button(root, text="load img2", command=lambda: self.load_image("right"), height=2, width=10, fg="#63caf9",cursor="heart")
        self.right_button.place(x=779, y=390)

        # sketch buttons
        self.left_sketch_button = tk.Button(root, text="sketch", command=lambda: self.toggle_sketch_mode("left"), height=2, width=10, fg="#34b744",cursor="circle")
        self.left_sketch_button.place(x=205, y=390)

        self.right_sketch_button = tk.Button(root, text="sketch", command=lambda: self.toggle_sketch_mode("right"), height=2, width=10, fg="#34b744",cursor="circle")
        self.right_sketch_button.place(x=905, y=390)

        # undo buttons
        self.left_undo_button = tk.Button(root, text="undo", command=lambda: self.undo("left"), height=2, width=10)
        self.left_undo_button.place(x=78, y=450)

        self.right_undo_button = tk.Button(root, text="undo", command=lambda: self.undo("right"), height=2, width=10)
        self.right_undo_button.place(x=779, y=450)

        self.warp_button = tk.Button(root, text="warp", command=self.warping, height=2, width=10, fg="#242624")
        self.warp_button.place(x=490, y=390)

        self.saveimg = tk.Button(root, text="save img", command=self.save_img, height=2, width=10, fg="#242624")
        self.saveimg.place(x=490, y=440)


        # exit button
        self.exit_button = tk.Button(root, text="exit", command=root.quit, height=2, width=10, fg="#f43924")
        self.exit_button.place(x=490, y=490)

        self.animating = False
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.get_resource_path("shape_predictor_68_face_landmarks.dat"))

    def get_resource_path(self, relative_path):
        """ Get the absolute path to a resource, works for dev and for PyInstaller """
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def load_image(self, position):
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return

        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = ImageOps.fit(image, (300, 300), Image.LANCZOS)
            image = ImageTk.PhotoImage(image)

            if position == "left":
                if self.left_image:
                    self.left_image.destroy()
                self.left_image = tk.Label(self.left_frame, image=image)
                self.left_image.image = image
                self.left_warp = image #TODO: 
                self.left_image.place(x=0, y=0)
                self.left_image_original = image
                self.left_image.bind("<Button-1>", lambda event: self.start_draw(event, "left"))
                self.left_image.bind("<B1-Motion>", lambda event: self.draw(event, "left"))
                self.left_image.bind("<ButtonRelease-1>", lambda event: self.end_draw(event, "left"))
            elif position == "right":
                if self.right_image:
                    self.right_image.destroy()
                self.right_image = tk.Label(self.right_frame, image=image)
                self.right_image.image = image
                self.right_warp = image #TODO: 
                self.right_image.place(x=0, y=0)
                self.right_image_original = image
                self.right_image.bind("<Button-1>", lambda event: self.start_draw(event, "right"))
                self.right_image.bind("<B1-Motion>", lambda event: self.draw(event, "right"))
                self.right_image.bind("<ButtonRelease-1>", lambda event: self.end_draw(event, "right"))
            self.PiQi_left = []
            self.PiQi_right = []
            
            self.right_sketch_mode = False
            self.left_sketch_mode = False

    def toggle_sketch_mode(self, position):
        if position == "left":
            if not self.left_image:
                return
            self.left_sketch_mode = not self.left_sketch_mode
            if self.left_sketch_mode:
                self.left_sketch_button.config(relief=tk.SUNKEN)
            else:
                self.left_sketch_button.config(relief=tk.RAISED)
        elif position == "right":
            if not self.right_image:
                return
            self.right_sketch_mode = not self.right_sketch_mode
            if self.right_sketch_mode:
                self.right_sketch_button.config(relief=tk.SUNKEN)
            else:
                self.right_sketch_button.config(relief=tk.RAISED)

    def start_draw(self, event, position):
        if (position == "left" and self.left_sketch_mode) or (position == "right" and self.right_sketch_mode):
            if position == "left":
                self.left_start_point = (event.x, event.y)
            elif position == "right":
                self.right_start_point = (event.x, event.y)

    def draw(self, event, position):
        if position == "left" and self.left_start_point:
            self.update_image("left", self.left_start_point, (event.x, event.y))
        elif position == "right" and self.right_start_point:
            self.update_image("right", self.right_start_point, (event.x, event.y))

    def end_draw(self, event, position):
        if position == "left" and self.left_start_point:
            self.PiQi_left.append([self.left_start_point, (event.x, event.y)])
            self.left_image_original = self.finalize_image(self.left_image_original, self.left_start_point, (event.x, event.y))
            self.left_start_point = None
        elif position == "right" and self.right_start_point:
            self.PiQi_right.append([self.right_start_point, (event.x, event.y)])
            self.right_image_original = self.finalize_image(self.right_image_original, self.right_start_point, (event.x, event.y))
            self.right_start_point = None
        if self.left_sketch_mode:
            print("left: ",len(self.PiQi_left))
        if self.right_sketch_mode:
            print("right: ",len(self.PiQi_right))


    def update_image(self, position, start_point, end_point):
        if position == "left" and self.left_image:
            image = ImageTk.getimage(self.left_image_original).copy()
            draw = ImageDraw.Draw(image)
            draw.line([start_point, end_point], fill="red", width=3)
            draw.polygon(self.arrow_head(start_point, end_point), fill="red")
            self.left_image.image = ImageTk.PhotoImage(image)
            self.left_image.config(image=self.left_image.image)
        elif position == "right" and self.right_image:
            image = ImageTk.getimage(self.right_image_original).copy()
            draw = ImageDraw.Draw(image)
            draw.line([start_point, end_point], fill="red", width=3)
            draw.polygon(self.arrow_head(start_point, end_point), fill="red")
            self.right_image.image = ImageTk.PhotoImage(image)
            self.right_image.config(image=self.right_image.image)

    def finalize_image(self, original_image, start_point, end_point):
        image = ImageTk.getimage(original_image).copy()
        draw = ImageDraw.Draw(image)
        draw.line([start_point, end_point], fill="red", width=3)
        draw.polygon(self.arrow_head(start_point, end_point), fill="red")
        return ImageTk.PhotoImage(image)

    def arrow_head(self, start, end, size=10):
        """ Helper function to create an arrow head polygon """
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        return [(end[0], end[1]),
                (end[0] - size * np.cos(angle - np.pi / 6), end[1] - size * np.sin(angle - np.pi / 6)),
                (end[0] - size * np.cos(angle + np.pi / 6), end[1] - size * np.sin(angle + np.pi / 6))]

    def undo(self, position):
        if position == "left" and self.PiQi_left:
            self.PiQi_left.pop()
            self.redraw_all_arrows("left")
        elif position == "right" and self.PiQi_right:
            self.PiQi_right.pop()
            self.redraw_all_arrows("right")

    def redraw_all_arrows(self, position):
        if position == "left" and self.left_image:
            image = ImageTk.getimage(self.left_image_original).copy()
            draw = ImageDraw.Draw(image)
            for line in self.PiQi_left:
                draw.line(line, fill="red", width=3)
                draw.polygon(self.arrow_head(line[0], line[1]), fill="red")
            self.left_image.image = ImageTk.PhotoImage(image)
            self.left_image.config(image=self.left_image.image)
        elif position == "right" and self.right_image:
            image = ImageTk.getimage(self.right_image_original).copy()
            draw = ImageDraw.Draw(image)
            for line in self.PiQi_right:
                draw.line(line, fill="red", width=3)
                draw.polygon(self.arrow_head(line[0], line[1]), fill="red")
            self.right_image.image = ImageTk.PhotoImage(image)
            self.right_image.config(image=self.right_image.image)

    def save_img(self):
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return
        if not self.animating and self.middle_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                image = ImageTk.getimage(self.middle_image.image)
                image.save(file_path)
        return

    def warping(self):

        if len(self.PiQi_left)!=len(self.PiQi_right) or len(self.PiQi_left)==0 or len(self.PiQi_right)==0:
            messagebox.showinfo("Error", "arrow pairs number not match!")
            print("left:",len(self.PiQi_left))
            print("right:",len(self.PiQi_right))
            return

        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return
        
        # TODO: ImagePhoto -> numpy
        if self.right_image and self.left_image:
            image_right = ImageTk.getimage(self.right_warp)
            image_right = np.array(image_right)[:, :, :3]
            image_left = ImageTk.getimage(self.left_warp)
            image_left = np.array(image_left)[:, :, :3]

            lis = [[(0,0),(0,300)],[(0,0),(300,0)],[(0,300),(300,300)],[(300,0),(300,300)]] #border
            piqi_left = self.PiQi_left.copy()
            piqi_right = self.PiQi_right.copy()
            for i in lis:
                piqi_left.append(i)
                piqi_right.append(i)
            piqi_left = np.array(piqi_left)
            piqi_right = np.array(piqi_right)

        else:
            return       

        def distance(P, Q, X):
            d = np.dot(X - P, np.array([-(Q - P)[1], (Q - P)[0]]))/np.linalg.norm(Q - P)
            return d  #abs(d)
        

        img1_warped = np.zeros((300,300,3),dtype=np.uint8)
        img2_warped = np.zeros((300,300,3),dtype=np.uint8)

        #middle lines
        warped = (piqi_left + piqi_right)/2



        #img1 : warp
        a = 0.5 #a: [0,infinity)
        b = 1.5 #b: [0.5, 2]
        p = 0.3 #p: [0, 1]
        for i in range(300):
            for j in range(300):
                X = np.float64([i,j])
                DSUM = np.float64([0,0])
                weightsum = 0.0
                for k in range(len(piqi_left)):
                    Q, P = np.array(piqi_left[k][0]),np.array(piqi_left[k][1]) #dst
                    Q_, P_ = np.array(warped[k][0]),np.array(warped[k][1]) #src

                    P = np.array([P[1],P[0]])
                    Q = np.array([Q[1],Q[0]])
                    P_ = np.array([P_[1],P_[0]])
                    Q_ = np.array([Q_[1],Q_[0]])

                    u = np.dot(X-P, Q-P)/np.power(np.linalg.norm(P - Q),2) #dst's u
                    v = distance(P, Q, X) #dst's v

                    #shortDist
                    if 0<=u and u<=1:
                        shortDist=v
                    elif u<0:
                        shortDist=np.linalg.norm(X-P)
                    else:
                        shortDist=np.linalg.norm(X-Q)
                    
                    
                    Xi = P_ + u*(Q_-P_) + v*(np.array([-(Q_ - P_)[1], (Q_ - P_)[0]]))/np.linalg.norm(Q_ - P_)
                    Di = X - Xi
                    
                    
                    weight = pow(pow(np.linalg.norm(P - Q),p)/(a+abs(shortDist)),b)
                    DSUM  += Di * weight
                    weightsum += weight

                X_ = X + DSUM / weightsum
                X_ = np.round(X_).astype(np.int64)

                if X_[0]>=0 and X_[0]<300 and X_[1]>=0 and X_[1]<300:
                    img1_warped[i][j] = image_left[X_[0]][X_[1]]
                else:
                    img1_warped[i][j] = 0

 
        #img2 : warp
        for i in range(300):
            for j in range(300):
                X = np.float64([i,j])
                DSUM = np.float64([0,0])
                weightsum = 0.0
                for k in range(len(piqi_left)):
                    Q, P = np.array(piqi_right[k][0]),np.array(piqi_right[k][1]) #dst
                    Q_, P_ = np.array(warped[k][0]),np.array(warped[k][1]) #src

                    P = np.array([P[1],P[0]])
                    Q = np.array([Q[1],Q[0]])
                    P_ = np.array([P_[1],P_[0]])
                    Q_ = np.array([Q_[1],Q_[0]])

                    u = np.dot(X-P, Q-P)/np.power(np.linalg.norm(P - Q),2) #dst's u
                    v = distance(P, Q, X) #dst's v

                    #shortDist
                    if 0<=u and u<=1:
                        shortDist=v
                    elif u<0:
                        shortDist=np.linalg.norm(X-P)
                    else: #u>1
                        shortDist=np.linalg.norm(X-Q)
                    

                    Xi = P_ + u*(Q_-P_) + v*(np.array([-(Q_ - P_)[1], (Q_ - P_)[0]]))/np.linalg.norm(Q_ - P_)
                    Di = X - Xi
                    weight = pow(pow(np.linalg.norm(P - Q),p)/(a+abs(shortDist)),b)
                    DSUM  += Di * weight
                    weightsum += weight
                X_ = X + DSUM / weightsum
                X_ = np.round(X_).astype(np.int64)

                if X_[0]>=0 and X_[0]<300 and X_[1]>=0 and X_[1]<300:
                    img2_warped[i][j] = image_right[X_[0]][X_[1]]
                else:
                    img2_warped[i][j] = 0


        alp = 0.5
        warped = cv2.addWeighted(img1_warped, alp, img2_warped, (1-alp),0.0)
        warped = Image.fromarray(warped)
        warped = ImageTk.PhotoImage(warped)

        if self.middle_image:
            self.middle_image.destroy()

        self.middle_image = tk.Label(self.middle_frame, image=warped)
        self.middle_image.image = warped
        self.middle_image.place(x=0, y=0)


        
        self.PiQi_left = []
        self.PiQi_right = []
        self.right_sketch_mode = False
        self.left_sketch_mode = False
        print("finished")
        return



if __name__=="__main__":
    root = tk.Tk()
    app = App(root)
    root.title("field morphing")
    root.geometry("1100x550")
    root.mainloop()
