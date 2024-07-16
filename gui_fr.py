import tkinter as tk
from tkinter import filedialog, messagebox
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

#took a lil help from chatgpt to understand tkinter and its use
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("500x200")
        
        self.known_face_encodings = []
        self.known_face_names = []

        self.frame = tk.Frame(self.root, padx=10, pady=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.label = tk.Label(self.frame, text="Enter your name:")
        self.label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.name_entry = tk.Entry(self.frame)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)

        self.load_button = tk.Button(self.frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        self.start_button = tk.Button(self.frame, text="Start Recognition", command=self.start_recognition)
        self.start_button.grid(row=1, column=0, columnspan=3, pady=10)

        self.status_label = tk.Label(self.frame, text="Status: Waiting to load images", anchor="w")
        self.status_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W+tk.E)

        self.video_capture = cv2.VideoCapture(1)
        
    
    def load_image(self):
        name = self.name_entry.get()
        image_path = filedialog.askopenfilename()
        if name and image_path:
            try:
                image_ = face_recognition.load_image_file(image_path)
                image_encoding = face_recognition.face_encodings(image_)[0]
                self.known_face_encodings.append(image_encoding)
                self.known_face_names.append(name)
                self.name_entry.delete(0, tk.END)
                messagebox.showinfo("Info", f"Loaded {name}'s image successfully!")
                self.status_label.config(text=f"Status: Loaded image for {name}")
                
            except IndexError:
                messagebox.showerror("Error", "No face found in the image. Please load a different image.")
                self.status_label.config(text="Status: Error loading image. No face detected.")
        else:
            messagebox.showwarning("Warning", "Please enter a name and select an image.")
            self.status_label.config(text="Status: Failed to load image. Missing name or image.")

    def start_recognition(self):
        if not self.known_face_encodings:
            messagebox.showwarning("Warning", "No images loaded. Please load images before starting recognition.")
            return

        students = self.known_face_names.copy()
        face_locations = []
        face_encodings = []

        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")

        with open(f"{current_date}.csv", "a+", newline="") as f:
            lnwriter = csv.writer(f)
            self.status_label.config(text="Status: Recognition started. Press 'q' to quit.")
            
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    self.status_label.config(text="Status: Error accessing webcam.")
                    break

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distance = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distance)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        
                        if name in self.known_face_names:
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4
                            
                            # Draw a box around the face (Green)
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            # Draw a box around the text (Yellow)
                            text = f"{name} Checked In at {now.strftime("%H:%M")}"
                            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            cv2.rectangle(frame, (10, 70), (10 + text_width, 70 + text_height + baseline), (0, 255, 255), 2)

                            # Add the text (Red)
                            cv2.putText(frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            
                            if name in students:
                                students.remove(name)
                                current_time = now.strftime("%H:%M:%S")
                                lnwriter.writerow([name, current_time])
                
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.video_capture.release()
        cv2.destroyAllWindows()
        self.status_label.config(text="Status: Recognition stopped. Video feed closed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
