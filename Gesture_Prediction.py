import pyttsx3

from tkinter import *
from PIL import Image, ImageTk
import cv2 
import mediapipe as mp
import numpy as np
import pickle



def speak(text):
    engine = pyttsx3.init() # object creation


    rate = engine.getProperty('rate')   # getting details of current speaking rate
    engine.setProperty('rate', 160)     # setting up new voice rate




    voices = engine.getProperty('voices')      
    engine.setProperty('voice', voices[1].id)  

    engine.say(text[5:])
    engine.runAndWait()
    engine.stop()



 

root = Tk()

sentence_text = ""

def update_sentence_label():
    global sentence_label, sentence_text
    sentence_label.config(text=sentence_text)
def camStart():
    global sentence_text
    model_dict = pickle.load(open("model.p", 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)
    empty_array = np.array([])
    sentence = np.array(['TEXT : '])

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    file_path = "my_dictionary.pkl"
    with open(file_path, 'rb') as file:
        loaded_dict = pickle.load(file)

    labels_dict = loaded_dict

    while True:

        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]

            empty_array = np.append(empty_array, predicted_character)
            if(len(empty_array)>30):
                if(max(empty_array)=='close'):
                    break
                if(sentence[-1]==max(empty_array)):
                    empty_array = np.array([])
                    continue
                else:
                    sentence = np.append(sentence,max(empty_array))
                    empty_array = np.array([])
                    words = ""
                    for i in sentence:
                        words+=i+" "
                    print(words)
                    global sentence_text
                    sentence_text = words
                    update_sentence_label()
                    

                    
                
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            #if(predicted_character=='close'):
             #   cap.release()
              #  cv2.destroyAllWindows()
              #  break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    #speak(sentence_text)
    cap.release()
    cv2.destroyAllWindows()

root.geometry("900x600")
root.minsize(900, 600)
root.maxsize(900, 600)
root.configure(background='white')

file_path = "my_dictionary.pkl"
with open(file_path, 'rb') as file:
    loaded_dict = pickle.load(file)

gestures = ""
for i in loaded_dict:
    gestures+=loaded_dict[i]+'\n'

image = Image.open("./src/Usability testing-bro.png")
image = image.resize((352, 352))
photo = ImageTk.PhotoImage(image)

left_frame = Frame(root, bg='white')
left_frame.pack(side=LEFT, padx=50)

text_label = Label(left_frame, text="Cloud Motion", bg='white', fg='black', font=('Verdana', 25))
text_label.pack()

img_label = Label(left_frame, image=photo, bg='white')
img_label.pack(pady=(20, 20))

right_frame = Frame(root, bg='white')
right_frame.pack(side=RIGHT, padx=50)

start_btn = Button(right_frame, text="Start", bg='#92E3A9', fg='black', width=30, height=2,command=camStart)
start_btn.pack(pady=(20, 20))
start_btn.config(font=('Verdana', 14))

sentence_label = Label(right_frame, text="", bg='white', fg='black', font=('Verdana', 15))
sentence_label.pack()

root.mainloop()
