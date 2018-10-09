from tkinter import *
from PIL import ImageTk, Image

response = None
def set_true():
    response=True

def set_false():
    response=False
def ask_question(image_no,question_prompt):
    response = None
    root=Tk()

    image_frame = Frame(root)
    image_frame.pack()
    question_frame = Frame(root)
    question_frame.pack()
    response_frame = Frame(root)
    response_frame.pack(side=BOTTOM)

    yes_button = Button(response_frame, text="Yes", fg="green", command=set_true)
    yes_button.pack(side=LEFT)

    no_button = Button(response_frame, text="No", fg="red")
    no_button.pack()
    root.mainloop()

    return response

ask_question(1,"dope")