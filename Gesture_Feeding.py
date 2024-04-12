import os
import cv2


def add_new_data(data_size,data_dir,gesture_name):
    dictionary = {}
    cap = cv2.VideoCapture(0)
    
    
    
    file_path = 'my_dictionary.pkl'
    if os.path.exists(file_path):
        print("f exist")
    else:
        with open(file_path, 'wb') as file:
            pickle.dump(dictionary, file)
            
            
    with open(file_path, 'rb') as file:
        loaded_dict = pickle.load(file)
    
    
    
    
    dir_name = 0 if os.listdir(data_dir) == [] else int(os.listdir(data_dir)[-1])+1
    dictionary = loaded_dict
    dictionary[dir_name]= gesture_name
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)
    print(dictionary)
    if not os.path.exists(os.path.join(data_dir, str(dir_name))):
            os.makedirs(os.path.join(data_dir, str(dir_name)))

    print('Collecting data for class {}'.format(dir_name))

    done = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < data_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(dir_name), '{}.jpg'.format(counter)), frame)

        counter += 1

    cap.release()
    cv2.destroyAllWindows()

def delete_directory_contents(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error: {file_path} : {e.strerror}")

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
                print(f"Deleted directory: {dir_path}")
            except OSError as e:
                print(f"Error: {dir_path} : {e.strerror}")

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tkinter import messagebox

class GestureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("900x600")
        self.minsize(900, 600)
        self.maxsize(900, 600)
        self.configure(background='white')
        self.title("Gesture App")
        
        self.pages = {}
        self.create_pages()
        self.show_page("StartPage")

    def create_pages(self):
        for Page in [StartPage, AddGesturePage, DeleteGesturePage]:
            page_name = Page.__name__
            self.pages[page_name] = Page(self)
            self.pages[page_name].pack(expand=True, fill="both")

    def show_page(self, page_name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[page_name].pack(expand=True, fill="both")

class StartPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(background='white')

        # Load image
        self.image = Image.open("./src/8773751.jpg")
        self.image = self.image.resize((352, 352))
        self.photo = ImageTk.PhotoImage(self.image)

        # Create and place the image label and progress bar on the left side
        left_frame = tk.Frame(self, bg='white')
        left_frame.pack(side=tk.LEFT, padx=50)

        text_label = tk.Label(left_frame, text="Cloud Motion", bg='white', fg='black', font=('Verdana', 25))
        text_label.pack()

        self.img_label = tk.Label(left_frame, image=self.photo, bg='white')
        self.img_label.pack(pady=(20, 20))

        # Create and place the input box and button on the right side
        right_frame = tk.Frame(self, bg='white')
        right_frame.pack(side=tk.RIGHT, padx=50)

        add_gesture_btn = tk.Button(right_frame, text="Add Gesture", bg='#92E3A9', fg='black', width=28, height=2,
                                    command=lambda: parent.show_page("AddGesturePage"))
        add_gesture_btn.pack(pady=(20, 20))
        add_gesture_btn.config(font=('vardana', 14))

        remove_gesture_btn = tk.Button(right_frame, text="Delete Gesture", bg='#92E3A9', fg='black', width=28, height=2,
                                       command=lambda: parent.show_page("DeleteGesturePage"))
        remove_gesture_btn.pack(pady=(20, 20))
        remove_gesture_btn.config(font=('vardana', 14))

        upload_btn = tk.Button(right_frame, text="Upload Model", bg='#92E3A9', fg='black', width=28, height=2)
        upload_btn.pack(pady=(20, 20))
        upload_btn.config(font=('vardana', 14))
        


class AddGesturePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(background='white')

        # Load image
        self.image = Image.open("./src/Add files-amico.png")
        self.image = self.image.resize((352, 352))
        self.photo = ImageTk.PhotoImage(self.image)

        # Create and place the image label and progress bar on the left side
        left_frame = tk.Frame(self, bg='white')
        left_frame.pack(side=tk.LEFT, padx=50)

        text_label = tk.Label(left_frame, text="Add New Gesture", bg='white', fg='black', font=('Verdana', 25))
        text_label.pack()

        self.img_label = tk.Label(left_frame, image=self.photo, bg='white')
        self.img_label.pack(pady=(20, 20))

        text_label = tk.Label(left_frame, text="Training %", bg='white', fg='black', font=('Verdana', 15))
        text_label.pack()

        self.progress = ttk.Progressbar(left_frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=20)

        # Create and place the input box and button on the right side
        right_frame = tk.Frame(self, bg='white')
        right_frame.pack(side=tk.RIGHT, padx=50)

        text_label = tk.Label(right_frame, text="Enter Gesture Name", bg='white', fg='black', font=('Verdana', 9), pady=10)
        text_label.pack()

        self.gesture_name_input = tk.Entry(right_frame,width=30, font=('Verdana', 12), bd=1, bg='#f2f2f2', highlightbackground='#92E3A9')
        self.gesture_name_input.pack(ipady=6, pady=(2, 20))

        train_gesture_btn = tk.Button(right_frame, text="Train", bg='#92E3A9', fg='black', width=20, height=2,command=self.train)
        train_gesture_btn.pack(pady=(20, 20))
        train_gesture_btn.config(font=('vardana', 14))
        
    def train(self):
        gesture_name = self.gesture_name_input.get()
        if gesture_name == "":
            messagebox.showerror("Error","Please Define Gesture Name")

        else:
            DATA_DIR = "./data"
            dataset_size = 200
            add_new_data(dataset_size,DATA_DIR,gesture_name)
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

            data = []
            labels = []
            num_of_classes = len(os.listdir(DATA_DIR))
            print(num_of_classes)
            def update_progress(self):
                self.progress['value'] += 100/num_of_classes
                self.update_idletasks()

            for dir_ in os.listdir(DATA_DIR):
                print("training "+ dir_)
                update_progress(self)
                for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
                    data_aux = []

                    x_ = []
                    y_ = []

                    img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    results = hands.process(img_rgb)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y

                                x_.append(x)
                                y_.append(y)

                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                data_aux.append(x - min(x_))
                                data_aux.append(y - min(y_))

                        data.append(data_aux)
                        labels.append(dir_)

            f = open("data.pickle", 'wb')
            pickle.dump({'data': data, 'labels': labels}, f)
            f.close()
            messagebox.showinfo("Success","Model Trained")

            data_dict = pickle.load(open("data.pickle", 'rb'))

            data = np.asarray(data_dict['data'])
            labels = np.asarray(data_dict['labels'])

            x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

            model = LogisticRegression()

            model.fit(x_train, y_train)

            y_predict = model.predict(x_test)

            score = accuracy_score(y_predict, y_test)

            print('{}% of samples were classified correctly !'.format(score * 100))

            f = open('model.p', 'wb')
            pickle.dump({'model': model}, f)
            f.close()
            self.master.show_page("StartPage")

            pass



class DeleteGesturePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.configure(background='white')

        # Load image
        self.image = Image.open("./src/Inbox cleanup-rafiki.png")
        self.image = self.image.resize((352, 352))
        self.photo = ImageTk.PhotoImage(self.image)

        # Create and place the image label and progress bar on the left side
        left_frame = tk.Frame(self, bg='white')
        left_frame.pack(side=tk.LEFT, padx=50)

        text_label = tk.Label(left_frame, text="Delete Gesture", bg='white', fg='black', font=('Verdana', 25))
        text_label.pack()

        self.img_label = tk.Label(left_frame, image=self.photo, bg='white')
        self.img_label.pack(pady=(20, 20))

        text_label = tk.Label(left_frame, text="Deleting %", bg='white', fg='black', font=('Verdana', 15))
        text_label.pack()

        self.progress = ttk.Progressbar(left_frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=20)

        # Create and place the input box and button on the right side
        right_frame = tk.Frame(self, bg='white')
        right_frame.pack(side=tk.RIGHT, padx=50)

        text_label = tk.Label(right_frame, text="Enter Gesture Name", bg='white', fg='black', font=('Verdana', 9), pady=10)
        text_label.pack()

        self.gesture_name_input = tk.Entry(right_frame, width=30, font=('Verdana', 12), bd=1, bg='#f2f2f2', highlightbackground='#92E3A9')
        self.gesture_name_input.pack(ipady=6, pady=(2, 20))

        delete_gesture_btn = tk.Button(right_frame, text="Delete", bg='#92E3A9', fg='black', width=20, height=2,command=self.delete)
        delete_gesture_btn.pack(pady=(20, 20))
        delete_gesture_btn.config(font=('vardana', 14))
        
    def delete(self):
        delete_gesture = self.gesture_name_input.get()
        file_path = 'my_dictionary.pkl'

        with open(file_path, 'rb') as file:
            loaded_dict = pickle.load(file)
        dictionary = loaded_dict
        inverted_dict = {v: k for k, v in dictionary.items()}
        result_key = inverted_dict.get(delete_gesture)
        directory_path = "./data/" + str(result_key)

        

        dictionary.pop(result_key)
        with open(file_path, 'wb') as file:
            pickle.dump(dictionary, file)
        print(dictionary)
        delete_directory_contents(directory_path)

        try:
            os.rmdir(directory_path)
            print(f"Directory '{directory_path}' successfully deleted.")
        except OSError as e:
            print(f"Error: {directory_path} : {e.strerror}")
        
        
        DATA_DIR = "./data"
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        data = []
        labels = []
        num_of_classes = len(os.listdir(DATA_DIR))
        print(num_of_classes)
        def update_progress(self):
            self.progress['value'] += 100/num_of_classes
            self.update_idletasks()

        for dir_ in os.listdir(DATA_DIR):
            print("training "+ dir_)
            update_progress(self)
            for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
                data_aux = []

                x_ = []
                y_ = []

                img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(dir_)

        f = open("data.pickle", 'wb')
        pickle.dump({'data': data, 'labels': labels}, f)
        f.close()
        messagebox.showinfo("Success","Gesture Deleted")

        data_dict = pickle.load(open("data.pickle", 'rb'))

        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        model = LogisticRegression()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)

        score = accuracy_score(y_predict, y_test)

        print('{}% of samples were classified correctly !'.format(score * 100))

        
        f = open('model.p', 'wb')
        pickle.dump({'model': model}, f)
        f.close()
    
        self.master.show_page("StartPage")

    


        pass

if __name__ == "__main__":
    app = GestureApp()
    app.mainloop()
