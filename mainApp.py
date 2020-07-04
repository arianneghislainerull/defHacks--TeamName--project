import tkinter as tk
import subprocess, os
from tkinter import font  as tkfont # python 3

from PIL import ImageTk, Image
from facedetection import *


class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.winfo_toplevel().title("TherAssist")


        # the container is where we'll stack a bunch of frames on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            frame.configure(bg='white')
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
        global tImg
        tImg = ImageTk.PhotoImage(Image.open('logo3.png'))

        # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
        panel = tk.Label(self, image=tImg)

        # The Pack geometry manager packs widgets in rows or columns.
        panel.pack(side="top", fill="both", expand="no")



        global img,img2,img3
        img = ImageTk.PhotoImage(Image.open("buttonManual.png"))  # make sure to add "/" not "\"
        img2 = ImageTk.PhotoImage(Image.open("btnHW.png"))  # make sure to add "/" not "\"
        img3 = ImageTk.PhotoImage(Image.open("btnEmo.png"))  # make sure to add "/" not "\"
        button1 = tk.Button(self, text="Manual",
                            command=lambda: controller.show_frame("PageOne"), image = img)
        button1.image = img

        def openHw():
            controller.show_frame("PageTwo")
            os.system('emotionSelector2.py')

        button2 = tk.Button(self, text="Homework",
                            command=lambda: openHw(), image = img2)
        button2.image = img2

        def openEmo():
            controller.show_frame("PageTwo")
            os.system('emotionSelector2.py')

        button3 = tk.Button(self, text="Emotion Selector", command=lambda: openEmo(),image = img3)
        button3.image = img3
        button1.pack()
        button2.pack()
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Manual Face Detection", font=controller.title_font)
        label.configure(bg ='white')
        label.pack(side="top", fill="x", pady=10)

        self.input = 0
        self.tries = 0
        #keeps track of user progress
        def changeInput(a):

            self.tries +=1
            if(a):
                self.input+=1
            else:
                self.input-=1
            return self.input, self.tries

        def generateReport():
            result = tk.Label(self, text = "Result: " + str(self.input/self.tries * 100) + "%")
            result.pack()
            return

        global imgS, imgN, imgF, imgH
        imgS = ImageTk.PhotoImage(Image.open("button_satisfactory.png"))
        imgN = ImageTk.PhotoImage(Image.open("button_needs-improvement.png"))
        imgF = ImageTk.PhotoImage(Image.open("button_finish-session.png"))
        imgH = ImageTk.PhotoImage(Image.open("button_home.png"))
        good = tk.Button(self, text = "Satisfactory", command = lambda: changeInput(True),image = imgS)
        improve = tk.Button(self, text = "Needs Improvement", command = lambda: changeInput(False), image = imgN)
        finish = tk.Button(self, text = "Finish Session", command = lambda: generateReport(), image = imgF)
        button = tk.Button(self, text="Home",
                           command=lambda: controller.show_frame("StartPage"), image = imgH)
        good.pack()
        improve.pack()
        finish.pack()
        button.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Homework Exercises", font=controller.title_font, bg = 'white')
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Home",
                           command=lambda: controller.show_frame("StartPage"), image = imgH)
        button.pack()


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
