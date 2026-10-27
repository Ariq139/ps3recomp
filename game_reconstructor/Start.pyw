from pathlib import Path
import traceback

from reconstructor.cli import main


try:
    main(["gui"])
except BaseException as error:
    if not isinstance(error, SystemExit) or error.code:
        details = traceback.format_exc()
        try:
            Path(__file__).with_name("startup_error.log").write_text(details, encoding="utf-8")
        except OSError:
            pass
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Cannot start Offline Game Reconstructor", str(error) + "\nSee startup_error.log for details.")
        root.destroy()
