
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
import dlib
from imutils import face_utils
import numpy as np
from scipy.spatial import Delaunay
import sys
import os

class ImageApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Mesh Morphing")

        self.left_image = None
        self.middle_image = None
        self.right_image = None

        self.left_label = tk.Label(master, text="image1", font=("Georgia", 25))
        self.left_label.place(x=167, y=30)

        self.right_label = tk.Label(master, text="image2", font=("Georgia", 25))
        self.right_label.place(x=860, y=30)

        self.middle_label = tk.Label(master, text="morphed", font=("Georgia", 25))
        self.middle_label.place(x=509, y=30)

        self.left_frame = tk.Frame(master, width=310, height=310, bd=2, relief=tk.GROOVE)
        self.left_frame.place(x=50, y=70)

        self.middle_frame = tk.Frame(master, width=310, height=310, bd=2, relief=tk.RIDGE)
        self.middle_frame.place(x=400, y=70)

        self.right_frame = tk.Frame(master, width=310, height=310, bd=2, relief=tk.GROOVE)
        self.right_frame.place(x=750, y=70)

        # load img
        self.left_button = tk.Button(master, text="load img1", command=lambda: self.load_image("left"), height=2, width=10)
        self.left_button.place(x=78, y=390)

        self.right_button = tk.Button(master, text="load img2", command=lambda: self.load_image("right"), height=2, width=10)
        self.right_button.place(x=779, y=390)

        # show mesh
        self.show_img1_mesh = tk.Button(master, text="show mesh", command=lambda: self.toggle_mesh("left"), height=2, width=10)
        self.show_img1_mesh.place(x=210, y=390)

        self.show_img2_mesh = tk.Button(master, text="show mesh", command=lambda: self.toggle_mesh("right"), height=2, width=10)
        self.show_img2_mesh.place(x=911, y=390)

        # alpha slider
        self.alpha_slider = tk.Scale(master, from_=0.01, to=0.99, resolution=0.01, orient=tk.HORIZONTAL, command=self.set_alpha)
        self.alpha_slider.set(0.5)
        self.alpha_slider.place(x=505, y=490)

        # alpha slider label
        self.alpha_label = tk.Label(master, text="α")
        self.alpha_label.place(x=492, y=505)

        # warping
        self.warp_button = tk.Button(master, text="animate", command=self.start_warp, height=2, width=10)
        self.warp_button.place(x=430, y=390)

        # show result
        self.show_result_button = tk.Button(master, text="result", command=self.show_warping_result, height=2, width=10) 
        self.show_result_button.place(x=562, y=390)


        # img1 to img2 morphing
        self.img1_to_img2_button = tk.Button(master, text="img1 to img2", command=lambda: self.start_morph(0, 1), height=2, width=10)
        self.img1_to_img2_button.place(x=78, y=440)

        # faceChange 1->2
        self.show_result_button = tk.Button(master, text="face 1 to 2", command=lambda: self.changeFace("left"), height=2, width=10)
        self.show_result_button.place(x=210, y=440)


        # img2 to img1 morphing
        self.img2_to_img1_button = tk.Button(master, text="img2 to img1", command=lambda: self.start_morph(1, 0), height=2, width=10)
        self.img2_to_img1_button.place(x=779, y=440)
    
        # faceChange 2->1
        self.show_result_button = tk.Button(master, text="face 2 to 1", command=lambda: self.changeFace("right"), height=2, width=10)
        self.show_result_button.place(x=911, y=440)

        # save img
        self.save_button = tk.Button(master, text="save img", command=self.save_img, height=2, width=5)
        self.save_button.place(x=515, y=445)

        self.animating = False  # 用於追蹤動畫是否正在進行
        self.current_triangle = 0
        self.img1_tri_idx = None
        self.img1_tri = None
        self.img2_tri = None
        self.alpha = 0.5
        self.beta = 0.5
        self.show_left_mesh = False
        self.show_right_mesh = False

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.get_resource_path("shape_predictor_68_face_landmarks.dat"))

    def get_resource_path(self, relative_path):
        """ Get the absolute path to a resource, works for dev and for PyInstaller """
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def set_alpha(self, value):
        self.alpha = float(value)

    def load_image(self, position):
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return  # 動畫進行中，禁止加載圖片

        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = ImageOps.fit(image, (300, 300), Image.LANCZOS)
            # Convert image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Detect faces
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) == 0:
                messagebox.showinfo("Error", "No face detected!\nPlease select another image")
                return  # 沒有檢測到人臉，要求重新加載圖片

            image = ImageTk.PhotoImage(image)
            if position == "left":
                if self.left_image:
                    self.left_image.destroy()
                self.left_image = tk.Label(self.left_frame, image=image)
                self.left_image.image = image
                self.left_image.place(x=0, y=0)
            elif position == "right":
                if self.right_image:
                    self.right_image.destroy()
                self.right_image = tk.Label(self.right_frame, image=image)
                self.right_image.image = image
                self.right_image.place(x=0, y=0)

    def toggle_mesh(self, position):
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return

        if position == "left":
            self.show_left_mesh = not self.show_left_mesh
            self.show_mesh(position, self.show_left_mesh)
        elif position == "right":
            self.show_right_mesh = not self.show_right_mesh
            self.show_mesh(position, self.show_right_mesh)

    def show_mesh(self, position, show):
        if position == "left":
            if self.left_image:
                left_mesh = np.array(ImageTk.getimage(self.left_image.image))[:, :, :3]
                left_mesh = cv2.cvtColor(left_mesh, cv2.COLOR_RGB2BGR)
                if show:
                    img1_cp = self.get_landmarks(left_mesh)
                    if len(img1_cp) > 0:
                        self.compute_delaunay_triangles(img1_cp, img1_cp)
                        for tri in self.img1_tri:
                            pts = np.int32(tri)
                            cv2.polylines(left_mesh, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
                left_mesh = cv2.cvtColor(left_mesh, cv2.COLOR_BGR2RGB)
                left_mesh = Image.fromarray(left_mesh)
                left_mesh = ImageTk.PhotoImage(left_mesh)
                if self.show_left_mesh:
                    self.left_overlay = tk.Label(self.left_frame, image=left_mesh)
                    self.left_overlay.image = left_mesh
                    self.left_overlay.place(x=0, y=0)
                else:
                    self.left_overlay.destroy()
                    self.left_overlay = None
        elif position == "right":
            if self.right_image:
                right_mesh = np.array(ImageTk.getimage(self.right_image.image))[:, :, :3]
                right_mesh = cv2.cvtColor(right_mesh, cv2.COLOR_RGB2BGR)
                if show:
                    img2_cp = self.get_landmarks(right_mesh)
                    if len(img2_cp) > 0:
                        self.compute_delaunay_triangles(img2_cp, img2_cp)
                        for tri in self.img1_tri:
                            pts = np.int32(tri)
                            cv2.polylines(right_mesh, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
                right_mesh = cv2.cvtColor(right_mesh, cv2.COLOR_BGR2RGB)
                right_mesh = Image.fromarray(right_mesh)
                right_mesh = ImageTk.PhotoImage(right_mesh)
                if self.show_right_mesh:
                    self.right_overlay = tk.Label(self.right_frame, image=right_mesh)
                    self.right_overlay.image = right_mesh
                    self.right_overlay.place(x=0, y=0)
                else:
                    self.right_overlay.destroy()
                    self.right_overlay = None

    def save_img(self):
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return
        if not self.animating and self.middle_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                image = ImageTk.getimage(self.middle_image.image)
                image.save(file_path)

    def get_landmarks(self, image):
        img_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray_img, 1)
        for face in faces:
            shape = self.predictor(gray_img, face)
            shape = face_utils.shape_to_np(shape)
            for s in shape:
                img_cp.append(s)
            if len(shape) != 68:
                return []
        img_cp = np.array(img_cp)
        return img_cp

    def compute_delaunay_triangles(self, img1_cp, img2_cp):
        self.img1_tri_idx = Delaunay(img1_cp).simplices
        self.img1_tri = []
        for tri in self.img1_tri_idx:
            self.img1_tri.append([img1_cp[tri[0]], img1_cp[tri[1]], img1_cp[tri[2]]])
        self.img1_tri = np.array(self.img1_tri).astype(np.float32)

        self.img2_tri = []
        for tri in self.img1_tri_idx:
            self.img2_tri.append([img2_cp[tri[0]], img2_cp[tri[1]], img2_cp[tri[2]]])
        self.img2_tri = np.array(self.img2_tri).astype(np.float32)

        def get_bbox(triangle):
            x, y, w, h = cv2.boundingRect(triangle)
            return (x, y)

        #sorted_indices = sorted(range(len(self.img1_tri)), key=lambda i: get_bbox(self.img1_tri[i])[0])
        sorted_indices = sorted(range(len(self.img1_tri)), key=lambda i: get_bbox(self.img1_tri[i])[1])
        self.img1_tri = np.float32([self.img1_tri[i] for i in sorted_indices])
        self.img2_tri = np.float32([self.img2_tri[i] for i in sorted_indices])

    def start_warp(self):
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return

        self.alpha_slider.config(state=tk.DISABLED)

        if self.left_image and self.right_image:
            self.animating = True
            self.left_original = np.array(ImageTk.getimage(self.left_image.image))[:, :, :3]
            self.right_original = np.array(ImageTk.getimage(self.right_image.image))[:, :, :3]

            self.img1 = self.left_original.copy()
            self.img2 = self.right_original.copy()

            img1_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]
            img2_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]

            gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            faces_img1 = self.detector(gray_img1, 1)
            for face in faces_img1:
                shape = self.predictor(gray_img1, face)
                shape = face_utils.shape_to_np(shape)
                for s in shape:
                    img1_cp.append(s)
                if len(shape) != 68:
                    return

            gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            faces_img2 = self.detector(gray_img2, 1)
            for face in faces_img2:
                shape = self.predictor(gray_img2, face)
                shape = face_utils.shape_to_np(shape)
                for s in shape:
                    img2_cp.append(s)
                if len(shape) != 68:
                    return

            img1_cp = np.array(img1_cp)
            img2_cp = np.array(img2_cp)

            self.compute_delaunay_triangles(img1_cp, img2_cp)

            self.img1_warp = self.img1.copy()
            self.img2_warp = self.img2.copy()
            self.morphed_image = np.zeros_like(self.img1)
            self.morphed_image = np.ones(self.img1.shape) * 236
            self.mid_tri = self.img1_tri * (1 - self.alpha) + self.img2_tri * self.alpha

            self.current_triangle = 0

            self.show_initial_grid()
            self.master.after(1000, self.warp_step)  # Add delay to show initial grid

    def show_initial_grid(self):
        img1_copy = self.left_original.copy()
        img2_copy = self.right_original.copy()

        for tri in self.img1_tri:
            pts = np.int32(tri)
            cv2.polylines(img1_copy, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
        for tri in self.img2_tri:
            pts = np.int32(tri)
            cv2.polylines(img2_copy, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

        img1_copy = Image.fromarray(img1_copy)
        img1_copy = ImageTk.PhotoImage(img1_copy)

        img2_copy = Image.fromarray(img2_copy)
        img2_copy = ImageTk.PhotoImage(img2_copy)

        if self.left_image:
            self.left_image.destroy()
        self.left_image = tk.Label(self.left_frame, image=img1_copy)
        self.left_image.image = img1_copy
        self.left_image.place(x=0, y=0)

        if self.right_image:
            self.right_image.destroy()
        self.right_image = tk.Label(self.right_frame, image=img2_copy)
        self.right_image.image = img2_copy
        self.right_image.place(x=0, y=0)

    def warp_step(self):
        if self.current_triangle < len(self.img1_tri_idx):
            i = self.current_triangle

            # ----------for img1----------
            M = cv2.getAffineTransform(self.img1_tri[i], self.mid_tri[i])
            (x_1, y_1, w_1, h_1) = cv2.boundingRect(self.img1_tri[i])
            img1_rect = self.img1[y_1:y_1 + h_1, x_1:x_1 + w_1]

            (x_2, y_2, w_2, h_2) = cv2.boundingRect(self.mid_tri[i])
            mid_rect = self.img1_warp[y_2:y_2 + h_2, x_2:x_2 + w_2]

            points = np.array([[self.mid_tri[i][0][0] - x_2, self.mid_tri[i][0][1] - y_2],
                               [self.mid_tri[i][1][0] - x_2, self.mid_tri[i][1][1] - y_2],
                               [self.mid_tri[i][2][0] - x_2, self.mid_tri[i][2][1] - y_2]], np.int32)

            img1_mask = np.zeros((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img1_mask, points, 255)

            img2_mask = np.ones((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img2_mask, points, 0)

            warp = cv2.warpAffine(self.img1, M, (300, 300), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warp = warp[y_2:y_2 + h_2, x_2:x_2 + w_2]
            warp = cv2.bitwise_and(warp, warp, mask=img1_mask)

            mid_rect = cv2.bitwise_and(mid_rect, mid_rect, mask=img2_mask)
            final = cv2.bitwise_or(warp, mid_rect)
            self.img1_warp[y_2:y_2 + h_2, x_2:x_2 + w_2] = final

            # ----------for img2----------
            M = cv2.getAffineTransform(self.img2_tri[i], self.mid_tri[i])
            (x_1, y_1, w_1, h_1) = cv2.boundingRect(self.img2_tri[i])
            img2_rect = self.img2[y_1:y_1 + h_1, x_1:x_1 + w_1]

            (x_2, y_2, w_2, h_2) = cv2.boundingRect(self.mid_tri[i])
            mid_rect = self.img2_warp[y_2:y_2 + h_2, x_2:x_2 + w_2]

            points = np.array([[self.mid_tri[i][0][0] - x_2, self.mid_tri[i][0][1] - y_2],
                               [self.mid_tri[i][1][0] - x_2, self.mid_tri[i][1][1] - y_2],
                               [self.mid_tri[i][2][0] - x_2, self.mid_tri[i][2][1] - y_2]], np.int32)

            img1_mask = np.zeros((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img1_mask, points, 255)

            img2_mask = np.ones((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img2_mask, points, 0)

            warp = cv2.warpAffine(self.img2, M, (300, 300), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warp = warp[y_2:y_2 + h_2, x_2:x_2 + w_2]
            warp = cv2.bitwise_and(warp, warp, mask=img1_mask)

            mid_rect = cv2.bitwise_and(mid_rect, mid_rect, mask=img2_mask)
            final = cv2.bitwise_or(warp, mid_rect)
            self.img2_warp[y_2:y_2 + h_2, x_2:x_2 + w_2] = final

            # Update morphed image using triangle mask
            mask = np.zeros_like(self.morphed_image)
            cv2.fillConvexPoly(mask, np.int32(self.mid_tri[i]), (1, 1, 1))
            morphed_tri = cv2.addWeighted(self.img1_warp, (1 - self.alpha), self.img2_warp, self.alpha, 0.0)
            self.morphed_image = cv2.add(self.morphed_image * (1 - mask), morphed_tri * mask)

            # 將當前的三角形畫紅色邊框
            morphed_image_copy = self.morphed_image.copy()
            pts = np.int32(self.mid_tri[i])
            cv2.polylines(morphed_image_copy, [pts], isClosed=True, color=(255, 204, 0), thickness=2)  # 紅色邊框

            final_morphed = morphed_image_copy.astype(np.uint8)
            final_morphed = Image.fromarray(final_morphed)
            final_morphed = ImageTk.PhotoImage(final_morphed)

            if self.middle_image:
                self.middle_image.destroy()
            self.middle_image = tk.Label(self.middle_frame, image=final_morphed)
            self.middle_image.image = final_morphed
            self.middle_image.place(x=0, y=0)

            # corres. img1&img2 triangle
            img1_copy = self.left_original.copy()
            img2_copy = self.right_original.copy()

            for j, tri in enumerate(self.img1_tri):
                pts = np.int32(tri)
                if j != i:
                    cv2.polylines(img1_copy, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
                else:
                    cv2.polylines(img1_copy, [pts], isClosed=True, color=(51, 204, 255), thickness=2)  # 紅色邊框

            for j, tri in enumerate(self.img2_tri):
                pts = np.int32(tri)
                if j != i:
                    cv2.polylines(img2_copy, [pts], isClosed=True, color=(255, 255, 255), thickness=1)
                else:
                    cv2.polylines(img2_copy, [pts], isClosed=True, color=(51, 204, 255), thickness=2)  # 紅色邊框

            img1_copy = Image.fromarray(img1_copy)
            img1_copy = ImageTk.PhotoImage(img1_copy)

            img2_copy = Image.fromarray(img2_copy)
            img2_copy = ImageTk.PhotoImage(img2_copy)

            if self.left_image:
                self.left_image.destroy()
            self.left_image = tk.Label(self.left_frame, image=img1_copy)
            self.left_image.image = img1_copy
            self.left_image.place(x=0, y=0)

            if self.right_image:
                self.right_image.destroy()
            self.right_image = tk.Label(self.right_frame, image=img2_copy)
            self.right_image.image = img2_copy
            self.right_image.place(x=0, y=0)

            self.current_triangle += 1
            self.master.after(150, self.warp_step)  # speed
        else:
            # clear all triangles
            img1_copy = self.left_original.copy()
            img2_copy = self.right_original.copy()

            img1_copy = Image.fromarray(img1_copy)
            img1_copy = ImageTk.PhotoImage(img1_copy)

            img2_copy = Image.fromarray(img2_copy)
            img2_copy = ImageTk.PhotoImage(img2_copy)

            if self.left_image:
                self.left_image.destroy()
            self.left_image = tk.Label(self.left_frame, image=img1_copy)
            self.left_image.image = img1_copy
            self.left_image.place(x=0, y=0)

            if self.right_image:
                self.right_image.destroy()
            self.right_image = tk.Label(self.right_frame, image=img2_copy)
            self.right_image.image = img2_copy
            self.right_image.place(x=0, y=0)

            # 最後清除黃色邊框
            final_morphed = self.morphed_image.astype(np.uint8)
            final_morphed = Image.fromarray(final_morphed)
            final_morphed = ImageTk.PhotoImage(final_morphed)

            if self.middle_image:
                self.middle_image.destroy()
            self.middle_image = tk.Label(self.middle_frame, image=final_morphed)
            self.middle_image.image = final_morphed
            self.middle_image.place(x=0, y=0)

            self.animating = False  # 動畫結束
            self.alpha_slider.config(state=tk.NORMAL)

    def start_morph(self, start_beta, end_beta):
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return

        self.alpha_slider.config(state=tk.DISABLED)

        if self.left_image and self.right_image:
            self.animating = True
            self.left_original = np.array(ImageTk.getimage(self.left_image.image))[:, :, :3]
            self.right_original = np.array(ImageTk.getimage(self.right_image.image))[:, :, :3]

            self.img1 = self.left_original.copy()
            self.img2 = self.right_original.copy()

            img1_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]
            img2_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]
            
            gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            faces_img1 = self.detector(gray_img1, 1)
            for face in faces_img1:
                shape = self.predictor(gray_img1, face)
                shape = face_utils.shape_to_np(shape)
                for s in shape:
                    img1_cp.append(s)
                if len(shape) != 68:
                    return

            gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            faces_img2 = self.detector(gray_img2, 1)
            for face in faces_img2:
                shape = self.predictor(gray_img2, face)
                shape = face_utils.shape_to_np(shape)
                for s in shape:
                    img2_cp.append(s)
                if len(shape) != 68:
                    return

            img1_cp = np.array(img1_cp)
            img2_cp = np.array(img2_cp)

            self.compute_delaunay_triangles(img1_cp, img2_cp)

            self.img1_warp = self.img1.copy()
            self.img2_warp = self.img2.copy()
            self.morphed_image = np.zeros_like(self.img1)
            self.morphed_image = np.ones(self.img1.shape) * 236
            self.current_beta = start_beta
            self.end_beta = end_beta
            self.beta_increment = (end_beta - start_beta) / 10.0

            self.morph_step()

    def morph_step(self):
        if (self.beta_increment > 0 and self.current_beta <= self.end_beta) or (self.beta_increment < 0 and self.current_beta >= self.end_beta):
            self.mid_tri = self.img1_tri * (1 - self.current_beta) + self.img2_tri * self.current_beta
            self.warp_images()
            self.current_beta += self.beta_increment
            self.master.after(100, self.morph_step)  # speed
        else:
            self.animating = False
            self.alpha_slider.config(state=tk.NORMAL)

    def warp_images(self):
        self.morphed_image.fill(236)

        for i in range(len(self.img1_tri_idx)):
            M = cv2.getAffineTransform(self.img1_tri[i], self.mid_tri[i])
            (x_1, y_1, w_1, h_1) = cv2.boundingRect(self.img1_tri[i])
            img1_rect = self.img1[y_1:y_1 + h_1, x_1:x_1 + w_1]

            (x_2, y_2, w_2, h_2) = cv2.boundingRect(self.mid_tri[i])
            mid_rect = self.img1_warp[y_2:y_2 + h_2, x_2:x_2 + w_2]

            points = np.array([[self.mid_tri[i][0][0] - x_2, self.mid_tri[i][0][1] - y_2],
                               [self.mid_tri[i][1][0] - x_2, self.mid_tri[i][1][1] - y_2],
                               [self.mid_tri[i][2][0] - x_2, self.mid_tri[i][2][1] - y_2]], np.int32)

            img1_mask = np.zeros((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img1_mask, points, 255)

            img2_mask = np.ones((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img2_mask, points, 0)

            warp = cv2.warpAffine(self.img1, M, (300, 300), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warp = warp[y_2:y_2 + h_2, x_2:x_2 + w_2]
            warp = cv2.bitwise_and(warp, warp, mask=img1_mask)

            mid_rect = cv2.bitwise_and(mid_rect, mid_rect, mask=img2_mask)
            final = cv2.bitwise_or(warp, mid_rect)
            self.img1_warp[y_2:y_2 + h_2, x_2:x_2 + w_2] = final

            # ----------for img2----------
            M = cv2.getAffineTransform(self.img2_tri[i], self.mid_tri[i])
            (x_1, y_1, w_1, h_1) = cv2.boundingRect(self.img2_tri[i])
            img2_rect = self.img2[y_1:y_1 + h_1, x_1:x_1 + w_1]

            (x_2, y_2, w_2, h_2) = cv2.boundingRect(self.mid_tri[i])
            mid_rect = self.img2_warp[y_2:y_2 + h_2, x_2:x_2 + w_2]

            points = np.array([[self.mid_tri[i][0][0] - x_2, self.mid_tri[i][0][1] - y_2],
                               [self.mid_tri[i][1][0] - x_2, self.mid_tri[i][1][1] - y_2],
                               [self.mid_tri[i][2][0] - x_2, self.mid_tri[i][2][1] - y_2]], np.int32)

            img1_mask = np.zeros((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img1_mask, points, 255)

            img2_mask = np.ones((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img2_mask, points, 0)

            warp = cv2.warpAffine(self.img2, M, (300, 300), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warp = warp[y_2:y_2 + h_2, x_2:x_2 + w_2]
            warp = cv2.bitwise_and(warp, warp, mask=img1_mask)

            mid_rect = cv2.bitwise_and(mid_rect, mid_rect, mask=img2_mask)
            final = cv2.bitwise_or(warp, mid_rect)
            self.img2_warp[y_2:y_2 + h_2, x_2:x_2 + w_2] = final

            # Update morphed image using triangle mask
            mask = np.zeros_like(self.morphed_image)
            cv2.fillConvexPoly(mask, np.int32(self.mid_tri[i]), (1, 1, 1))
            morphed_tri = cv2.addWeighted(self.img1_warp, (1 - self.beta), self.img2_warp, self.beta, 0.0)
            self.morphed_image = cv2.add(self.morphed_image * (1 - mask), morphed_tri * mask)

        final_morphed = self.morphed_image.astype(np.uint8)
        final_morphed = Image.fromarray(final_morphed)
        final_morphed = ImageTk.PhotoImage(final_morphed)

        if self.middle_image:
            self.middle_image.destroy()
        self.middle_image = tk.Label(self.middle_frame, image=final_morphed)
        self.middle_image.image = final_morphed
        self.middle_image.place(x=0, y=0)

    def show_warping_result(self):
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return

        if self.left_image and self.right_image:
            self.left_original = np.array(ImageTk.getimage(self.left_image.image))[:, :, :3]
            self.right_original = np.array(ImageTk.getimage(self.right_image.image))[:, :, :3]

            self.img1 = self.left_original.copy()
            self.img2 = self.right_original.copy()

            img1_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]
            img2_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]

            gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            faces_img1 = self.detector(gray_img1, 1)
            for face in faces_img1:
                shape = self.predictor(gray_img1, face)
                shape = face_utils.shape_to_np(shape)
                for s in shape:
                    img1_cp.append(s)
                if len(shape) != 68:
                    return

            gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            faces_img2 = self.detector(gray_img2, 1)
            for face in faces_img2:
                shape = self.predictor(gray_img2, face)
                shape = face_utils.shape_to_np(shape)
                for s in shape:
                    img2_cp.append(s)
                if len(shape) != 68:
                    return

            img1_cp = np.array(img1_cp)
            img2_cp = np.array(img2_cp)

            self.compute_delaunay_triangles(img1_cp, img2_cp)

            self.img1_warp = self.img1.copy()
            self.img2_warp = self.img2.copy()
            self.morphed_image = np.zeros_like(self.img1)
            self.morphed_image = np.ones(self.img1.shape) * 236
            self.mid_tri = self.img1_tri * (1 - self.alpha) + self.img2_tri * self.alpha
            self.beta = self.alpha  # 確保在 result 按鈕時能正確顯示

            self.warp_images()
    
    def changeFace(self, pos): #1->2
        if self.animating:
            messagebox.showinfo("Error", "Animation in progress!\nPlease try again later.")
            return

        if self.left_image and self.right_image:
            self.left_original = np.array(ImageTk.getimage(self.left_image.image))[:, :, :3]
            self.right_original = np.array(ImageTk.getimage(self.right_image.image))[:, :, :3]

            self.img1 = self.left_original.copy()
            self.img2 = self.right_original.copy()

            img1_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]
            img2_cp = [[0, 0], [150, 0], [300, 0], [0, 150], [300, 150], [0, 300], [150, 300], [300, 300]]

            gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            faces_img1 = self.detector(gray_img1, 1)
            for face in faces_img1:
                shape = self.predictor(gray_img1, face)
                shape = face_utils.shape_to_np(shape)
                if pos=="right":
                    hull = cv2.convexHull(shape)
                for s in shape:
                    img1_cp.append(s)
                if len(shape) != 68:
                    return

            gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
            faces_img2 = self.detector(gray_img2, 1)
            for face in faces_img2:
                shape = self.predictor(gray_img2, face)
                shape = face_utils.shape_to_np(shape)
                if pos=="left":
                    hull = cv2.convexHull(shape)                    #TODO: 
                for s in shape:
                    img2_cp.append(s)
                if len(shape) != 68:
                    return

            img1_cp = np.array(img1_cp)
            img2_cp = np.array(img2_cp)

            self.compute_delaunay_triangles(img1_cp, img2_cp)

            self.img1_warp = self.img1.copy()
            self.img2_warp = self.img2.copy()
            self.morphed_image = np.zeros_like(self.img1)
            self.morphed_image = np.ones(self.img1.shape) * 236
            if pos=="left":
                self.mid_tri = self.img2_tri
            elif pos=="right":
                self.mid_tri = self.img1_tri
            self.beta = self.alpha  # 確保在 result 按鈕時能正確顯示
        self.morphed_image.fill(236)




        for i in range(len(self.img1_tri_idx)):
            M = cv2.getAffineTransform(self.img1_tri[i], self.mid_tri[i])
            (x_1, y_1, w_1, h_1) = cv2.boundingRect(self.img1_tri[i])
            img1_rect = self.img1[y_1:y_1 + h_1, x_1:x_1 + w_1]

            (x_2, y_2, w_2, h_2) = cv2.boundingRect(self.mid_tri[i])
            mid_rect = self.img1_warp[y_2:y_2 + h_2, x_2:x_2 + w_2]

            points = np.array([[self.mid_tri[i][0][0] - x_2, self.mid_tri[i][0][1] - y_2],
                               [self.mid_tri[i][1][0] - x_2, self.mid_tri[i][1][1] - y_2],
                               [self.mid_tri[i][2][0] - x_2, self.mid_tri[i][2][1] - y_2]], np.int32)

            img1_mask = np.zeros((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img1_mask, points, 255)

            img2_mask = np.ones((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img2_mask, points, 0)

            warp = cv2.warpAffine(self.img1, M, (300, 300), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warp = warp[y_2:y_2 + h_2, x_2:x_2 + w_2]
            warp = cv2.bitwise_and(warp, warp, mask=img1_mask)

            mid_rect = cv2.bitwise_and(mid_rect, mid_rect, mask=img2_mask)
            final = cv2.bitwise_or(warp, mid_rect)
            self.img1_warp[y_2:y_2 + h_2, x_2:x_2 + w_2] = final

            # ----------for img2----------
            M = cv2.getAffineTransform(self.img2_tri[i], self.mid_tri[i])
            (x_1, y_1, w_1, h_1) = cv2.boundingRect(self.img2_tri[i])
            img2_rect = self.img2[y_1:y_1 + h_1, x_1:x_1 + w_1]

            (x_2, y_2, w_2, h_2) = cv2.boundingRect(self.mid_tri[i])
            mid_rect = self.img2_warp[y_2:y_2 + h_2, x_2:x_2 + w_2]

            points = np.array([[self.mid_tri[i][0][0] - x_2, self.mid_tri[i][0][1] - y_2],
                               [self.mid_tri[i][1][0] - x_2, self.mid_tri[i][1][1] - y_2],
                               [self.mid_tri[i][2][0] - x_2, self.mid_tri[i][2][1] - y_2]], np.int32)

            img1_mask = np.zeros((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img1_mask, points, 255)

            img2_mask = np.ones((mid_rect.shape[0], mid_rect.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(img2_mask, points, 0)

            warp = cv2.warpAffine(self.img2, M, (300, 300), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warp = warp[y_2:y_2 + h_2, x_2:x_2 + w_2]
            warp = cv2.bitwise_and(warp, warp, mask=img1_mask)

            mid_rect = cv2.bitwise_and(mid_rect, mid_rect, mask=img2_mask)
            final = cv2.bitwise_or(warp, mid_rect)
            self.img2_warp[y_2:y_2 + h_2, x_2:x_2 + w_2] = final






            # Update morphed image using triangle mask
            mask = np.zeros_like(self.morphed_image)
            cv2.fillConvexPoly(mask, np.int32(self.mid_tri[i]), (1, 1, 1))

            if pos=="left":
                if cv2.pointPolygonTest(hull, self.mid_tri[i][0], measureDist=False)>=0 and cv2.pointPolygonTest(hull, self.mid_tri[i][1], measureDist=False)>=0 and cv2.pointPolygonTest(hull, self.mid_tri[i][2], measureDist=False)>=0:
                    morphed_tri = self.img1_warp
                else:
                    morphed_tri = self.img2_warp
            elif pos=="right":
                if cv2.pointPolygonTest(hull, self.mid_tri[i][0], measureDist=False)>=0 and cv2.pointPolygonTest(hull, self.mid_tri[i][1], measureDist=False)>=0 and cv2.pointPolygonTest(hull, self.mid_tri[i][2], measureDist=False)>=0:
                    morphed_tri = self.img2_warp
                else:
                    morphed_tri = self.img1_warp

            #morphed_tri = cv2.addWeighted(self.img1_warp, (1 - self.beta), self.img2_warp, self.beta, 0.0)
            self.morphed_image = cv2.add(self.morphed_image * (1 - mask), morphed_tri * mask)

        # for i in hull:
        #     print("i=",i)
        #     cv2.circle(self.morphed_image,i[0],2,color=(255,50,0))
        final_morphed = self.morphed_image.astype(np.uint8)
        final_morphed = Image.fromarray(final_morphed)
        final_morphed = ImageTk.PhotoImage(final_morphed)

        if self.middle_image:
            self.middle_image.destroy()
        self.middle_image = tk.Label(self.middle_frame, image=final_morphed)
        self.middle_image.image = final_morphed
        self.middle_image.place(x=0, y=0)
  
        

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.geometry("1100x550")
    root.mainloop()
