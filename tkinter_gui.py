import tkinter as tk
root = tk.Tk()
root.title("Simple tkinter Application")
root.geometry("400x300")

def say_hello():
    print("Hello, World!")  
    print("good bye world")
    
hello_buton = tk.Button(root, text="click me", command=say_hello)
hello_buton.pack(pady=20)

root.mainloop()
