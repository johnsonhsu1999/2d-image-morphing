import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import sys
import os

class MeshWarpingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Mesh Warping App")

        self.canvas_width = 300
        self.canvas_height = 300

        self.canvas_left = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height)
        self.canvas_left.grid(row=0, column=0)

        self.canvas_center = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas_center.grid(row=0, column=1)

        self.canvas_right = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height)
        self.canvas_right.grid(row=0, column=2)

        self.load_image_button_left = tk.Button(self.master, text="load image1", command=self.load_left_image)
        self.load_image_button_left.grid(row=1, column=0)

        self.load_image_button_right = tk.Button(self.master, text="load image2", command=self.load_right_image)
        self.load_image_button_right.grid(row=1, column=2)

        self.reset_button_left = tk.Button(self.master, text="reset", command=self.reset_left_mesh)
        self.reset_button_left.grid(row=2, column=0)

        self.reset_button_right = tk.Button(self.master, text="reset", command=self.reset_right_mesh)
        self.reset_button_right.grid(row=2, column=2)

        self.undo_button_left = tk.Button(self.master, text="undo", command=self.undo_left_last_action)
        self.undo_button_left.grid(row=3, column=0)

        self.undo_button_right = tk.Button(self.master, text="undo", command=self.undo_right_last_action)
        self.undo_button_right.grid(row=3, column=2)

        self.warp_button = tk.Button(self.master, text="WARP!", command=self.warp_and_blend)
        self.warp_button.grid(row=4, column=1)

        self.left_image = None
        self.left_photo_image = None
        self.left_original_mesh_points = []
        self.left_mesh_points = []
        self.left_mesh_lines = []
        self.left_point_ids = []
        self.left_point_labels = []
        self.left_history = []

        self.right_image = None
        self.right_photo_image = None
        self.right_original_mesh_points = []
        self.right_mesh_points = []
        self.right_mesh_lines = []
        self.right_point_ids = []
        self.right_point_labels = []
        self.right_history = []

        self.canvas_left.bind("<Button-1>", lambda event: self.select_mesh_point(event, "left"))
        self.canvas_left.bind("<B1-Motion>", lambda event: self.drag_mesh_point(event, "left"))
        self.canvas_left.bind("<ButtonRelease-1>", lambda event: self.record_history(event, "left"))

        self.canvas_right.bind("<Button-1>", lambda event: self.select_mesh_point(event, "right"))
        self.canvas_right.bind("<B1-Motion>", lambda event: self.drag_mesh_point(event, "right"))
        self.canvas_right.bind("<ButtonRelease-1>", lambda event: self.record_history(event, "right"))

    def get_resource_path(self, relative_path):
        """ Get the absolute path to a resource, works for dev and for PyInstaller """
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def load_left_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.resize(image, (self.canvas_width, self.canvas_height))
            self.left_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.left_photo_image = ImageTk.PhotoImage(image=Image.fromarray(self.left_image))
            self.canvas_left.create_image(0, 0, anchor=tk.NW, image=self.left_photo_image)
            self.create_mesh_grid("left")

    def load_right_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.resize(image, (self.canvas_width, self.canvas_height))
            self.right_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.right_photo_image = ImageTk.PhotoImage(image=Image.fromarray(self.right_image))
            self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.right_photo_image)
            self.create_mesh_grid("right")

    def create_mesh_grid(self, side):
        rows, cols = 10, 10  # Increase grid size for more density
        step_x = self.canvas_width // (cols - 1)
        step_y = self.canvas_height // (rows - 1)

        if side == "left":
            self.left_mesh_points = []
            self.left_point_ids = []
            self.left_point_labels = []
            self.left_mesh_lines = []

            for y in range(rows):
                for x in range(cols):
                    point = (x * step_x, y * step_y)
                    self.left_mesh_points.append(point)
                    point_id = self.canvas_left.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill="red")
                    self.left_point_ids.append(point_id)
                    label_id = self.canvas_left.create_text(point[0], point[1] - 10, text=f"({point[0]},{point[1]})", font=("Arial", 8), fill="black")
                    self.left_point_labels.append(label_id)

            self.left_original_mesh_points = self.left_mesh_points.copy()
            self.left_history = [self.left_mesh_points.copy()]

            for y in range(rows):
                for x in range(cols - 1):
                    idx = y * cols + x
                    line_id = self.canvas_left.create_line(self.left_mesh_points[idx], self.left_mesh_points[idx + 1], fill="blue")
                    self.left_mesh_lines.append(line_id)

            for x in range(cols):
                for y in range(rows - 1):
                    idx = y * cols + x
                    line_id = self.canvas_left.create_line(self.left_mesh_points[idx], self.left_mesh_points[idx + cols], fill="blue")
                    self.left_mesh_lines.append(line_id)
        
        elif side == "right":
            self.right_mesh_points = []
            self.right_point_ids = []
            self.right_point_labels = []
            self.right_mesh_lines = []

            for y in range(rows):
                for x in range(cols):
                    point = (x * step_x, y * step_y)
                    self.right_mesh_points.append(point)
                    point_id = self.canvas_right.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill="red")
                    self.right_point_ids.append(point_id)
                    label_id = self.canvas_right.create_text(point[0], point[1] - 10, text=f"({point[0]},{point[1]})", font=("Arial", 8), fill="black")
                    self.right_point_labels.append(label_id)

            self.right_original_mesh_points = self.right_mesh_points.copy()
            self.right_history = [self.right_mesh_points.copy()]

            for y in range(rows):
                for x in range(cols - 1):
                    idx = y * cols + x
                    line_id = self.canvas_right.create_line(self.right_mesh_points[idx], self.right_mesh_points[idx + 1], fill="blue")
                    self.right_mesh_lines.append(line_id)

            for x in range(cols):
                for y in range(rows - 1):
                    idx = y * cols + x
                    line_id = self.canvas_right.create_line(self.right_mesh_points[idx], self.right_mesh_points[idx + cols], fill="blue")
                    self.right_mesh_lines.append(line_id)

    def select_mesh_point(self, event, side):
        x, y = event.x, event.y
        if side == "left":
            closest_point_idx = min(range(len(self.left_mesh_points)), key=lambda i: ((self.left_mesh_points[i][0]-x)**2 + (self.left_mesh_points[i][1]-y)**2)**0.5)
            self.selected_point_idx = closest_point_idx
            self.selected_side = "left"
        elif side == "right":
            closest_point_idx = min(range(len(self.right_mesh_points)), key=lambda i: ((self.right_mesh_points[i][0]-x)**2 + (self.right_mesh_points[i][1]-y)**2)**0.5)
            self.selected_point_idx = closest_point_idx
            self.selected_side = "right"

    def drag_mesh_point(self, event, side):
        x, y = event.x, event.y
        if side == "left":
            self.left_mesh_points[self.selected_point_idx] = (x, y)
            self.canvas_left.coords(self.left_point_ids[self.selected_point_idx], x - 3, y - 3, x + 3, y + 3)
            self.canvas_left.coords(self.left_point_labels[self.selected_point_idx], x, y - 10)
            self.canvas_left.itemconfig(self.left_point_labels[self.selected_point_idx], text=f"({x},{y})")
            self.update_lines("left")
        elif side == "right":
            self.right_mesh_points[self.selected_point_idx] = (x, y)
            self.canvas_right.coords(self.right_point_ids[self.selected_point_idx], x - 3, y - 3, x + 3, y + 3)
            self.canvas_right.coords(self.right_point_labels[self.selected_point_idx], x, y - 10)
            self.canvas_right.itemconfig(self.right_point_labels[self.selected_point_idx], text=f"({x},{y})")
            self.update_lines("right")

    def update_lines(self, side):
        if side == "left":
            for line_id in self.left_mesh_lines:
                self.canvas_left.delete(line_id)
            self.left_mesh_lines = []

            rows, cols = 10, 10  # Must match grid size in create_mesh_grid
            for y in range(rows):
                for x in range(cols - 1):
                    idx = y * cols + x
                    line_id = self.canvas_left.create_line(self.left_mesh_points[idx], self.left_mesh_points[idx + 1], fill="blue")
                    self.left_mesh_lines.append(line_id)

            for x in range(cols):
                for y in range(rows - 1):
                    idx = y * cols + x
                    line_id = self.canvas_left.create_line(self.left_mesh_points[idx], self.left_mesh_points[idx + cols], fill="blue")
                    self.left_mesh_lines.append(line_id)
        
        elif side == "right":
            for line_id in self.right_mesh_lines:
                self.canvas_right.delete(line_id)
            self.right_mesh_lines = []

            rows, cols = 10, 10  # Must match grid size in create_mesh_grid
            for y in range(rows):
                for x in range(cols - 1):
                    idx = y * cols + x
                    line_id = self.canvas_right.create_line(self.right_mesh_points[idx], self.right_mesh_points[idx + 1], fill="blue")
                    self.right_mesh_lines.append(line_id)

            for x in range(cols):
                for y in range(rows - 1):
                    idx = y * cols + x
                    line_id = self.canvas_right.create_line(self.right_mesh_points[idx], self.right_mesh_points[idx + cols], fill="blue")
                    self.right_mesh_lines.append(line_id)

    def reset_left_mesh(self):
        self.left_mesh_points = self.left_original_mesh_points.copy()
        for idx, point in enumerate(self.left_mesh_points):
            self.canvas_left.coords(self.left_point_ids[idx], point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3)
            self.canvas_left.coords(self.left_point_labels[idx], point[0], point[1] - 10)
            self.canvas_left.itemconfig(self.left_point_labels[idx], text=f"({point[0]},{point[1]})")
        self.update_lines("left")
        self.left_history = [self.left_mesh_points.copy()]

    def reset_right_mesh(self):
        self.right_mesh_points = self.right_original_mesh_points.copy()
        for idx, point in enumerate(self.right_mesh_points):
            self.canvas_right.coords(self.right_point_ids[idx], point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3)
            self.canvas_right.coords(self.right_point_labels[idx], point[0], point[1] - 10)
            self.canvas_right.itemconfig(self.right_point_labels[idx], text=f"({point[0]},{point[1]})")
        self.update_lines("right")
        self.right_history = [self.right_mesh_points.copy()]

    def record_history(self, event, side):
        if side == "left":
            self.left_history.append(self.left_mesh_points.copy())
            if len(self.left_history) > 10:  # limit the history size to prevent memory issues
                self.left_history.pop(0)
        elif side == "right":
            self.right_history.append(self.right_mesh_points.copy())
            if len(self.right_history) > 10:  # limit the history size to prevent memory issues
                self.right_history.pop(0)

    def undo_left_last_action(self):
        if len(self.left_history) > 1:
            self.left_history.pop()  # Remove current state
            self.left_mesh_points = self.left_history[-1].copy()
            for idx, point in enumerate(self.left_mesh_points):
                self.canvas_left.coords(self.left_point_ids[idx], point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3)
                self.canvas_left.coords(self.left_point_labels[idx], point[0], point[1] - 10)
                self.canvas_left.itemconfig(self.left_point_labels[idx], text=f"({point[0]},{point[1]})")
            self.update_lines("left")

    def undo_right_last_action(self):
        if len(self.right_history) > 1:
            self.right_history.pop()  # Remove current state
            self.right_mesh_points = self.right_history[-1].copy()
            for idx, point in enumerate(self.right_mesh_points):
                self.canvas_right.coords(self.right_point_ids[idx], point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3)
                self.canvas_right.coords(self.right_point_labels[idx], point[0], point[1] - 10)
                self.canvas_right.itemconfig(self.right_point_labels[idx], text=f"({point[0]},{point[1]})")
            self.update_lines("right")

    def warp_and_blend(self):
        if self.right_mesh_points and self.left_mesh_points:
            left_quadrangle = []
            right_quadrangle = []
            for i in range(89):
                if i % 10 == 9:
                    continue
                left_quadrangle.append([self.left_mesh_points[i], self.left_mesh_points[i + 1], self.left_mesh_points[i + 10], self.left_mesh_points[i + 11]])
                right_quadrangle.append([self.right_mesh_points[i], self.right_mesh_points[i + 1], self.right_mesh_points[i + 10], self.right_mesh_points[i + 11]])

            left_quadrangle = np.array(left_quadrangle)  # (81, 4, 2)
            right_quadrangle = np.array(right_quadrangle)

            if left_quadrangle.shape != right_quadrangle.shape:
                messagebox.showerror("Error", "The number of quadrangles is not consistent between the two images.")
                return

            warped_left = np.zeros_like(self.left_image)  # (300, 300, 3)
            warped_right = np.zeros_like(self.right_image)  # (300, 300, 3)

            for i in range(left_quadrangle.shape[0]):
                M1, _ = cv2.estimateAffine2D(left_quadrangle[i], right_quadrangle[i])
                warped_left_part = cv2.warpAffine(self.left_image, M1, (self.left_image.shape[1], self.left_image.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)  # 0.0 black

                M2, _ = cv2.estimateAffine2D(right_quadrangle[i], left_quadrangle[i])
                warped_right_part = cv2.warpAffine(self.right_image, M2, (self.right_image.shape[1], self.right_image.shape[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)  # 0.0 black

                points = np.array([right_quadrangle[i][0], right_quadrangle[i][2], right_quadrangle[i][3], right_quadrangle[i][1]], dtype=np.int32)
                mask = np.zeros(self.left_image.shape[:2], dtype=np.uint8)  # for source
                mask2 = np.ones(self.left_image.shape[:2], dtype=np.uint8)  # for target

                cv2.fillPoly(mask, [points], 255)
                cv2.fillPoly(mask2, [points], 0)

                res1 = cv2.bitwise_and(warped_left_part, warped_left_part, mask=mask)
                res2 = cv2.bitwise_and(warped_right_part, warped_right_part, mask=mask)
                warped_left = cv2.bitwise_or(res1, warped_left)
                warped_right = cv2.bitwise_or(res2, warped_right)

            # Cross dissolve
            alpha = 0.5
            blended_image = cv2.addWeighted(warped_left, alpha, warped_right, 1 - alpha, 0)

            self.show_blended_image(blended_image)

    def show_blended_image(self, image):
        blended_photo_image = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.canvas_center.create_image(0, 0, anchor=tk.NW, image=blended_photo_image)
        self.canvas_center.image = blended_photo_image

def main():
    root = tk.Tk()
    app = MeshWarpingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
