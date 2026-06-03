# main.py

import tkinter as tk
from mural_app import SAM2MuralApp


def main():
    root = tk.Tk()
    app = SAM2MuralApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
