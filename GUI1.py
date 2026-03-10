from tkinter import *
from PIL import ImageGrab
from main import predict
import numpy as np


def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()


def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y 

   
def MyProject():
    global l1

    widget = cv
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
    img = img.convert('L')

    x = np.asarray(img)
    vec = np.zeros((1, 784))
    k = 0
    for i in range(28):
        for j in range(28):
            vec[0][k] = x[i][j]
            k += 1

    Theta1 = np.loadtxt('Theta1.txt')
    Theta2 = np.loadtxt('Theta2.txt')

    pred = predict(Theta1, Theta2, vec / 255)

    l1 = Label(window, text="Digit = " + str(pred[0]), font=('Times New Roman', 25))
    l1.place(x=230, y=420)


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=15, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y
    
def change_font_size():
    new_font_size = 25  
    L1.config(font=('Times New Roman', new_font_size))
    l1.config(font=('Times New Roman', new_font_size))
    b1.config(font=('Times New Roman', new_font_size - 5))
    b2.config(font=('Times New Roman', new_font_size - 5))

window = Tk()
window.title("Handwritten digit recognition")
l1 = Label()


L1 = Label(window, text="Handwritten Digit Recoginition", font=('Times New Roman', 30), fg="black")
L1.place(x=35, y=10)

b1 = Button(window, text="Clear", font=('Times New Roman', 20), bg="white", fg="black", command=clear_widget)
b1.place(x=120, y=370)

b2 = Button(window, text="Predict", font=('Times New Roman', 20), bg="white", fg="red", command=MyProject)
b2.place(x=320, y=370)

cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()
