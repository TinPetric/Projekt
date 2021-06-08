import tkinter as tk
from PIL import ImageTk, Image

path = "ucenje\\"

i = 358
for i in range(2500,2633):
    print(str(i + 1))
    ime = path + str(i + 1) + ".jpg"
    root = tk.Tk()

    img = ImageTk.PhotoImage(Image.open(ime))
    label = tk.Label(root, image = img).pack()

    root.after(1000, lambda: root.destroy())
    root.mainloop()

