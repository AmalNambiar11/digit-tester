import os, importlib.util
package_name = 'tkinter'
is_tkinter = True
if importlib.util.find_spec(package_name) is None:
    is_tkinter = False
else:
    import tkinter as tk
import testbench
from random import randint

train_X = testbench.train_X
train_Y = testbench.train_Y
TRAINING_SET_SIZE = testbench.TRAINING_SET_SIZE
TESTING_SET_SIZE = testbench.TESTING_SET_SIZE

bot = testbench.ClassifierBot()
bot.load_file("model3")

class App(tk.Tk):
    def __init__(self, *args, **kwargs): 
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)  
        container.pack(side = "top", fill = "both", expand = True) 
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        self.frames = {}  
        for F in (StartPage, SimpleTestPage):
            frame = F(container, self)
            self.frames[F] = frame 
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(tk.Frame):
    def __init__(self, parent, controller): 
        tk.Frame.__init__(self, parent)
 
        label = tk.Label(self, text ="Startpage")   #, font = LARGEFONT)
         
        label.grid(row = 0, column = 4, padx = 10, pady = 10) 
  
        button1 = tk.Button(self, text ="Test Single Image",
        command = lambda : controller.show_frame(SimpleTestPage))
     
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)
  
        button2 = tk.Button(self, text ="Models",
        command = lambda : controller.show_frame(Page2))

        button2.grid(row = 2, column = 1, padx = 10, pady = 10)

class SimpleTestPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.image_num = tk.IntVar()
        self.top_label_text = tk.StringVar()

        top_label = tk.Label(self, textvariable=self.top_label_text)

        image_num_entry = tk.Entry(self, textvariable=self.image_num)
        image_num_entry.bind('<Return>', lambda event: self.load(C, self.image_num.get()-1))

        C = tk.Canvas(self, height=300, width=400, bg="white")
        C.create_rectangle(10, 10, 100, 100, fill="yellow")

        load_btn = tk.Button(self, text="Load", command=lambda: self.load(C, self.image_num.get()-1))
        random_btn = tk.Button(self, text="Get Random Image", command=lambda: self.load(C, -1))
        test_btn = tk.Button(self, text="Test", command=lambda: self.test(self.image_num.get()-1))

        self.top_label_text.set("HELLO WORLD!")
        self.draw_image(C, 58247)
        top_label.pack()
        C.pack()
        image_num_entry.pack()
        load_btn.pack()
        random_btn.pack()
        test_btn.pack()
    
    def load(self, canvas, image_num):
        self.focus()
        if image_num == -1:
            image_num = randint(0, TRAINING_SET_SIZE-1)
            self.image_num.set(image_num+1)
        self.draw_image(canvas, image_num)

    def draw_image(self, canvas, image_num):
        image_size = (28, 28)
        h = 8
        label = train_Y[image_num]
        for j in range(image_size[0]):
            for i in range(image_size[1]):
                content = train_X[image_num][j][i]
                code = "#" + 3*hex(content)[2:]
                canvas.create_rectangle(i*h, j*h, (i+1)*h, (j+1)*h, width=0, fill=code)
        self.top_label_text.set("Image: " + str(image_num+1) + "\t\tLabel: " + str(label))

    def test(self, image_num):
        testbench.test_single_image(bot, image_num, False)

class ModelsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

def new_model_page():
    pass

def load_model_page():
    dir_list = os.listdir("models")
    for i in dir_list:
        print(i[0:-4:1], sep=" ")
    while True:
        choice = input("Choose a model> ")
        if choice+".pkl" in dir_list:
            bot.load_file(choice)
            break
        else:
            print("INVALID MODEL")
    print("Bot loaded.")
    print("1) Train 2) Test")   

def bots_page():
    print("1) Load model 2) New Model")
    while True:
        choice = input("> ")
        if choice.isdigit(): 
            break
    choice = int(choice)
    if choice == 1:
        load_model_page()
    elif choice == 2:
        new_model_page()

def simple_test_page():
    print("The MNIST testing set contains 10000 images to try your bot on.")
    print("Using model1 ..")
    print("Type a number from 1 to", TESTING_SET_SIZE, "to pick an image or type an invalid input to pick randomly -")
    choice = input("> ")
    if choice.isdigit():
        choice = int(choice)
        if not (choice>0 and choice<=TESTING_SET_SIZE):
            choice = randint(1, TESTING_SET_SIZE)
    else:
        choice = randint(1, TESTING_SET_SIZE)
    testbench.test_single_image(bot, choice-1, False)

def start3():
    root = App()
    root.title("Digit Tester")
    root.mainloop()

def start_gui():
    print("GUI NOT READY!")
    start3()
    return
    root = Tk()
    root.geometry("800x800")
    button1 = Button(text="Test Single Image")
    button2 = Button(text="Test Batch")
    button1.pack()
    button2.pack()
    
    root.mainloop()

def start_shell():
    print("MENU\n1) Test Single Image\n2) Bots\n3) Quit")
    choice = ""
    while True:
        choice = input("> ")
        if choice.isdigit(): 
            break
    choice = int(choice)
    if choice == 1:
        simple_test_page()
    elif choice == 2:
        bots_page()

def start():
    if is_tkinter:
        print("DIGIT TESTER\n")
        print("Tkinter is installed on your system. Press 1 to use GUI, 2 to continue with shell, any other key to quit.")
        choice = input("> ")
        print("")
        if not choice.isdigit():
            print("BYE!")
            return
        choice = int(choice)
        if choice == 1:
            start_gui()
        elif choice == 2:
            start_shell()
        else:
            print("BYE!")
            return
    else:
        print("DIGIT TESTER\n")
        print("Tkinter is NOT installed on your system. Continuing with shell ..\n")
        start_shell()

#load_model_page()
start()