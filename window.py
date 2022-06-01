import tkinter as tk
from tkinter import filedialog
from tkinter import*
from tkinter.ttk import*
from PIL import ImageTk, Image
from Programme_pozu import*
import pygame

global left_path 
left_path = "aucune_image.png"
global right_path 
right_path = "aucune_image.png"

class Window(tk.Frame):

    def __init__(self, master = None):
        
        tk.Frame.__init__(self, master)
        self.master = master
        self.path = ""
        self.database_here=0
        self.s, self.clf, self.PCA = 0,0,0
        self.label = 0
        self.feature = 0
        self.image_left_bool = 1
        self.filepath = tk.StringVar()
        self.master.title("Pozu")
        self.pack(fill=BOTH, expand=True)
        self['background'] = '#8AAAE5'
       
        self.title_label = tk.Label(self, text="POZU", font=('Lato',20,'bold'), fg='#FEF0E2', bg='#8AAAE5')
        self.title_label.place(x=500, y=35, anchor=CENTER)

        self.buttons_frame = tk.Frame(self, relief=FLAT, bg='#2F3C7E', borderwidth=1, height=20)
        self.buttons_frame.pack(fill=BOTH, side=BOTTOM, padx = 5, pady = 5)
        self.operation_button = tk.Button(self.buttons_frame, text='Start', font=('Lato',12,'bold'), width="30", command = self.start_operation)
        self.operation_button.pack(side=LEFT,padx=5,pady=5)
        self.load_db_button = tk.Button(self.buttons_frame, text='Load Database', font=('Lato',12,'bold'), width="30", command = self.learn_database)
        self.load_db_button.pack(side=LEFT,padx=5,pady=5)
        self.quit_button = tk.Button(self.buttons_frame, text='Quit', font=('Lato',12,'bold'), width="30", command = self.close_window)
        self.quit_button.pack(side=RIGHT,padx=5,pady=5)
        
        self.path_frame = tk.Frame(self, relief=FLAT, bg='#2F3C7E', borderwidth=1, height=20)
        self.path_frame.pack(fill=BOTH, side=BOTTOM, padx = 5, pady = 5)
        self.filepathText = tk.Entry(self.path_frame, textvariable = self.filepath, state=DISABLED, width="65")
        self.filepathText.pack(padx=40,pady=5)
        self.open_button = tk.Button(self.path_frame, text='Open', width="50", font=('Lato',11,'bold'), command=self.first_browser)
        self.open_button.pack(padx=40,pady=5)

        self.images_frame = tk.Frame(self, relief=FLAT, bg='#2F3C7E', borderwidth=1, height=200)
        self.images_frame.pack(fill=BOTH, side=BOTTOM, padx = 5, pady = 5) 

        self.left_image_before = Image.open(left_path)
        self.left_image_resize = self.left_image_before.resize((300,450), Image.ANTIALIAS)
        self.left_image = ImageTk.PhotoImage(self.left_image_resize)
        self.left_zone = Label(self.images_frame, image=self.left_image)
        self.left_zone.image = self.left_image
        self.left_zone.pack(side=LEFT,padx=50,pady=5)

        self.right_image_before = Image.open("aucune_image.png")
        self.right_image_resize = self.right_image_before.resize((250,450), Image.ANTIALIAS)
        self.right_image = ImageTk.PhotoImage(self.right_image_resize)
        self.right_zone = Label(self.images_frame, image=self.right_image)
        self.right_zone.image = self.right_image
        self.right_zone.pack(side=RIGHT,padx=50,pady=5)
        
    def show_file_browser(self):
        self.filename = filedialog.askopenfilename()
        return self.filename

    def first_browser(self):
        file = self.show_file_browser()
        self.filepath.set(file)
        self.path = file
        left_path = self.path
        print(self.path)
        print("left : ",left_path)
        self.change_image(left_path)
        
    def start_operation(self):
        '''Mettez le son'''
        pygame.mixer.init()
        pygame.mixer.music.load('pillar.mp3')
        pygame.mixer.music.set_volume(0.008)
        pygame.mixer.music.play()
        
        print("Я люблю вас")
        print(self.path)
        
        if self.database_here==0:
            print("NO DATABASE LOADED")
        if self.path == "":
            print("NO IMAGE TO COMPUTE")
        else:
            self.feature = association_pozu(self.path, self.s, self.clf, self.PCA)
            self.label = labellisation( self.feature,self.clf)
            print("YOU ARE !")
            print(self.label[0])
            self.change_image_result("DB_Anime/"+self.label[0]+".jpg")
            
    def close_window(self):
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        interface.destroy()
        self.quit
        
    def learn_database(self):
        if self.database_here == 1:
            print("Features already extracted")
        else:
            self.database_here = 1      
            print("Features Loading...")
            self.s, self.clf, self.PCA = learn_from_all_data()
            
        print(self.path)


    def change_image(self,path):
        self.left_zone.destroy()
        self.left_image_before = Image.open(path)
        self.left_image_resize = self.left_image_before.resize((300,450), Image.ANTIALIAS)
        self.left_image = ImageTk.PhotoImage(self.left_image_resize)
        self.left_zone = Label(self.images_frame, image=self.left_image)
        self.left_zone.image = self.left_image
        self.left_zone.pack(side=LEFT,padx=50,pady=5)
        
    def change_image_result(self,path):
        self.right_zone.destroy()
        self.right_image_before = Image.open(path)
        self.right_image_resize = self.right_image_before.resize((250,450), Image.ANTIALIAS)
        self.right_image = ImageTk.PhotoImage(self.right_image_resize)
        self.right_zone = Label(self.images_frame, image=self.right_image)
        self.right_zone.image = self.right_image
        self.right_zone.pack(side=RIGHT,padx=50,pady=5)

    
    
    
    
interface = tk.Tk()
interface.geometry("1000x700")
interface.resizable(0, 0)

app = Window(interface)

interface.mainloop()
