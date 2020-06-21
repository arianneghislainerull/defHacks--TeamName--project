import tkinter as tk
import subprocess, os
from tkinter import font  as tkfont # python 3


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
        label = tk.Label(self, text="Face Detection", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="Manual",
                            command=lambda: controller.show_frame("PageOne"))

        def openHw():
            controller.show_frame("PageTwo")
            os.system('emotionSelector.py')

        button2 = tk.Button(self, text="Homework",
                            command=lambda: openHw())

        def openEmo():
            controller.show_frame("PageTwo")
            os.system('emotionSelecter.py')

        button3 = tk.Button(self, text="Emotion Selector", command=lambda: openEmo())
        button1.pack()
        button2.pack()
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Manual Face Detection", font=controller.title_font)
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

        good = tk.Button(self, text = "Satisfactory", command = lambda: changeInput(True))
        improve = tk.Button(self, text = "Needs Improvement", command = lambda: changeInput(False))
        finish = tk.Button(self, text = "Finish Session", command = lambda: generateReport())
        button = tk.Button(self, text="Home",
                           command=lambda: controller.show_frame("StartPage"))
        good.pack()
        improve.pack()
        finish.pack()
        button.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Homework Exercises", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Home",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
