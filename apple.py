import tkinter as tk
import webbrowser as wb
def instagram(event=None):
    username=e.get() or "arus._.h"
    
    url = f"https://www.instagram.com/{username}/"
    wb.open(url)
r= tk.Tk()
r.title("instagram user searcher")
r.geometry("400x400")
r.configure(bg="black")
r.resizable(False, False)    
r.minsize(200, 200)
l=tk.Label(r, text="Find anyone user from here", bg="#0a85a3", fg="white", font=("arial", 20), width=23,  wraplength=200)
l.pack(pady=20)
b=tk.Button(r, text="CLICK TO GO TO THE OWNER", bg="yellow", fg="black", font=("calibiri", 18, "bold"), command=instagram, activeforeground="red" )
b.pack(pady=20)
e=tk.Entry(r, bg="black", fg="white", font=("calibiri", 18))
e.pack(pady=20)
e.bind("<Return>", instagram)
r.mainloop()