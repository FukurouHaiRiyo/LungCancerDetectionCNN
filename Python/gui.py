import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy as np

from keras.models import load_model
# model = load_model('modelGAN-C.h5')
model = load_model('/home/andrei/Desktop/ProiectLicenta/Python/VGG19-fruits-99.00.h5')

classes = {
    0: 'Lung adenocarcinoma',
    1: 'Lung squamous cell carcinoma',
    2: 'Lung benign tissue'
}

print(classes)


# initialise GUI
top = tk.Tk()
top.geometry('900x800')
top.title('Detectarea celuleor anormale la nivelul plamanilor')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((224, 224))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image/255.0
    pred = model.predict([image])
    print(pred[0][0])
    print(pred[0])

    lung_cancer_type = ['Lung aca', 'Lung scc', 'Lung n']

    dictionary = dict(zip(lung_cancer_type, pred[0]))

    print(dictionary)

    max_key = max(dictionary, key=dictionary.get)
    label.configure(text=f'{max_key} detected')
     

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


if __name__ == '__main__':
    upload = Button(top, text="Upload an image", command=upload_image, padx=10,  pady=5)
    upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    upload.pack(side=BOTTOM, pady=50)
    sign_image.pack(side=BOTTOM, expand=True)
    label.pack(side=BOTTOM, expand=True)
    heading = Label(top, text="Detectarea celuleor anormale la nivelul plamanilor",pady=20, font=('arial', 20, 'bold'))
    heading.configure(background='#CDCDCD', foreground='#364156')
    heading.pack()
    top.mainloop()

