import json
import os
import queue
import sqlite3
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import Config, ROOT
from .controller import Controller


class Application:
    def __init__(self, root, profile):
        self.root, self.profile = root, Path(profile)
        self.events = queue.SimpleQueue()
        self.thread, self.stop, self.closing = None, threading.Event(), False
        self.config = Config.load(profile) if self.profile.is_file() else Config(Path.cwd(), Path.cwd(), ROOT / "workspaces/default")
        self.root.title("Offline Game Reconstructor")
        self.root.geometry(f"{min(1040, root.winfo_screenwidth() - 60)}x{min(760, root.winfo_screenheight() - 100)}")
        self.root.minsize(800, 540)
        self.project = tk.StringVar(value=str(self.config.project))
        self.ghidra = tk.StringVar(value=str(self.config.ghidra))
        self.workspace = tk.StringVar(value=str(self.config.workspace))
        self.batch = tk.IntVar(value=self.config.batch_size)
        self.apply = tk.BooleanVar(value=self.config.apply_verified)
        self.status = tk.StringVar(value="Ready. Local Ghidra and deterministic rewrite rules; no online services.")
        self.counts = tk.StringVar(value="Analysis and readable drafts are tracked separately from verified implementations.")
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)
        setup, functions, log = ttk.Frame(notebook), ttk.Frame(notebook), ttk.Frame(notebook)
        notebook.add(setup, text="Project and progress")
        notebook.add(functions, text="Functions")
        notebook.add(log, text="Activity")
        setup.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate((("Game-only project", self.project), ("Ghidra installation", self.ghidra), ("Work folder", self.workspace))):
            ttk.Label(setup, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=10)
            ttk.Entry(setup, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6)
            ttk.Button(setup, text="Browse", command=lambda v=variable: self.choose_directory(v)).grid(row=row, column=2, padx=10)
        ttk.Label(setup, text="Local reference programs").grid(row=3, column=0, sticky="nw", padx=10, pady=10)
        self.references = tk.Listbox(setup, height=5, exportselection=False)
        self.references.grid(row=3, column=1, sticky="nsew", padx=6, pady=6)
        for reference in self.config.references:
            self.references.insert("end", str(reference))
        buttons = ttk.Frame(setup)
        buttons.grid(row=3, column=2, sticky="n", padx=10)
        ttk.Button(buttons, text="Add .gzf", command=self.add_references).pack(fill="x", pady=4)
        ttk.Button(buttons, text="Remove", command=self.remove_reference).pack(fill="x", pady=4)
        settings = ttk.Frame(setup)
        settings.grid(row=4, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        ttk.Label(settings, text="Functions per batch").pack(side="left")
        ttk.Spinbox(settings, from_=1, to=64, textvariable=self.batch, width=5).pack(side="left", padx=10)
        ttk.Checkbutton(settings, text="Apply verified changes automatically", variable=self.apply).pack(side="left", padx=15)
        ttk.Label(setup, text="Readable drafts are review files. Only checked replacements enter the buildable project.", wraplength=850).grid(row=5, column=0, columnspan=3, sticky="w", padx=10, pady=10)
        actions = ttk.Frame(setup)
        actions.grid(row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        self.start_button = ttk.Button(actions, text="Start / resume", command=self.start)
        self.start_button.pack(side="left", padx=(0, 8))
        self.pause_button = ttk.Button(actions, text="Pause after batch", command=self.pause, state="disabled")
        self.pause_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Check setup", command=self.check).pack(side="left", padx=8)
        ttk.Button(actions, text="Save profile", command=self.save).pack(side="left", padx=8)
        ttk.Button(actions, text="Open work folder", command=self.open_workspace).pack(side="left", padx=8)
        ttk.Label(setup, textvariable=self.status, wraplength=900).grid(row=7, column=0, columnspan=3, sticky="w", padx=10, pady=16)
        ttk.Label(setup, textvariable=self.counts, wraplength=900).grid(row=8, column=0, columnspan=3, sticky="w", padx=10, pady=10)
        self.progress = ttk.Progressbar(setup, maximum=1, value=0)
        self.progress.grid(row=9, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        ttk.Label(setup, text="Game builds are disabled. No Codex, API, or remote model is used.").grid(row=10, column=0, columnspan=3, sticky="w", padx=10, pady=10)
        self.functions = ttk.Treeview(functions, columns=("address", "name", "role", "status"), show="headings")
        for column, width in (("address", 110), ("name", 480), ("role", 100), ("status", 130)):
            self.functions.heading(column, text=column.capitalize())
            self.functions.column(column, width=width)
        self.functions.pack(fill="both", expand=True, padx=10, pady=10)
        bar = ttk.Frame(functions)
        bar.pack(fill="x", padx=10, pady=10)
        ttk.Button(bar, text="Refresh functions", command=self.refresh_functions).pack(side="left", padx=4)
        ttk.Button(bar, text="Open readable draft", command=self.open_draft).pack(side="left", padx=4)
        ttk.Button(bar, text="Show details", command=self.show_details).pack(side="left", padx=4)
        self.search = tk.StringVar()
        ttk.Entry(bar, textvariable=self.search, width=22).pack(side="left", padx=8)
        ttk.Button(bar, text="Find name / address", command=self.refresh_functions).pack(side="left", padx=4)
        ttk.Label(functions, text="Shows up to 1,000 matching entries. Search to find entries anywhere in the project.").pack(anchor="w", padx=12, pady=4)
        self.log = tk.Text(log, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, padx=10, pady=10)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(200, self.poll)
        self.load_progress()

    def read_config(self):
        return Config(Path(self.project.get()).resolve(), Path(self.ghidra.get()).resolve(), Path(self.workspace.get()).resolve(),
                      self.config.language, [Path(v).resolve() for v in self.references.get(0, "end")],
                      int(self.batch.get()), self.config.worker_timeout, bool(self.apply.get()))

    def choose_directory(self, variable):
        selected = filedialog.askdirectory(initialdir=variable.get())
        if selected:
            variable.set(selected)

    def add_references(self):
        for path in filedialog.askopenfilenames(filetypes=[("Ghidra archives", "*.gzf")]):
            if path not in self.references.get(0, "end"):
                self.references.insert("end", path)

    def remove_reference(self):
        for selected in reversed(self.references.curselection()):
            self.references.delete(selected)

    def check(self):
        try:
            result = self.read_config().validate()
            messagebox.showinfo("Setup", f"Local setup is ready. {result['references']} reference program(s).")
        except Exception as error:
            messagebox.showerror("Setup needs attention", str(error))

    def save(self):
        try:
            self.config = self.read_config()
            self.config.save(self.profile)
            self.status.set("Profile saved.")
        except Exception as error:
            messagebox.showerror("Cannot save profile", str(error))

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            self.config = self.read_config()
            self.config.validate()
            self.config.save(self.profile)
        except Exception as error:
            messagebox.showerror("Cannot start", str(error))
            return
        self.stop = threading.Event()
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        def run():
            try:
                Controller(self.config, self.events.put, self.stop).run()
            except Exception as error:
                self.events.put({"stage": "error", "reason": str(error)})
            finally:
                self.events.put({"stage": "thread_finished"})
        self.thread = threading.Thread(target=run, daemon=False)
        self.thread.start()

    def pause(self):
        self.stop.set()
        self.status.set("Pausing after the current local batch. Completed work will be retained.")
        self.pause_button.configure(state="disabled")

    def load_progress(self):
        if self.workspace.get().startswith(("\\\\", "//")):
            return {}
        try:
            value = json.loads((Path(self.workspace.get()) / "progress.json").read_text(encoding="utf-8"))
            counts = value.get("tasks", {})
            total = sum(counts.values())
            analyzed, blocked = counts.get("analyzed", 0), counts.get("blocked", 0)
            self.progress.configure(maximum=max(total, 1), value=analyzed + blocked)
            self.counts.set(f"Analyzed: {analyzed:,} / {total:,}   Needs attention: {blocked:,}\n"
                            f"Readable drafts: {value.get('readable_drafts', 0):,}   Validated rewrites: {value.get('validated_rewrites', 0):,}")
            return value
        except (OSError, ValueError):
            return {}

    def poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event["stage"] == "thread_finished":
                    self.start_button.configure(state="normal")
                    self.pause_button.configure(state="disabled")
                    self.refresh_functions()
                else:
                    self.status.set(event["stage"].replace("_", " ").capitalize() + (": " + str(event["reason"]) if event.get("reason") else ""))
                    self.log.configure(state="normal")
                    self.log.insert("end", json.dumps(event, ensure_ascii=False) + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
        except queue.Empty:
            pass
        self.load_progress()
        if self.closing and (not self.thread or not self.thread.is_alive()):
            self.root.destroy()
            return
        self.root.after(300, self.poll)

    def task_rows(self):
        if self.workspace.get().startswith(("\\\\", "//")):
            return []
        path = (Path(self.workspace.get()) / "state.sqlite").resolve()
        if not path.is_file():
            return []
        profile = self.load_progress().get("profile", "")
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            query, parameters = "profile=? AND enabled=1", [profile]
            search = self.search.get().strip()
            if search:
                try:
                    address = int(search, 16)
                except ValueError:
                    address = -1
                query += " AND (name LIKE ? OR address=?)"
                parameters.extend(["%" + search + "%", address])
            return [dict(row) for row in db.execute(
                "SELECT * FROM tasks WHERE " + query + " ORDER BY CASE status WHEN 'blocked' THEN 0 WHEN 'running' THEN 1 WHEN 'analyzed' THEN 2 ELSE 3 END,priority,address LIMIT 1000", parameters)]

    def refresh_functions(self):
        try:
            self.rows = self.task_rows()
            self.functions.delete(*self.functions.get_children())
            for index, row in enumerate(self.rows):
                self.functions.insert("", "end", iid=str(index), values=(f"0x{row['address']:08X}", row["name"], row["role"], row["status"]))
        except sqlite3.Error:
            pass

    def selected(self):
        selected = self.functions.selection()
        return self.rows[int(selected[0])] if selected else None

    def open_draft(self):
        row = self.selected()
        if row:
            paths = list((Path(self.workspace.get()) / "readable_drafts").rglob(f"*_{row['address']:08X}.c"))
            paths.sort(key=lambda path: ("incoming" in path.relative_to(Path(self.workspace.get()) / "readable_drafts").parts, str(path)))
            if paths:
                self.open_path(paths[0])
            else:
                messagebox.showinfo("Readable draft", "No complete readable draft is available for this entry yet.")

    def show_details(self):
        row = self.selected()
        if row:
            messagebox.showinfo(row["name"], row["detail"] or row["export"] or "Waiting for local analysis.")

    def open_workspace(self):
        path = Path(self.workspace.get())
        path.mkdir(parents=True, exist_ok=True)
        self.open_path(path)

    @staticmethod
    def open_path(path):
        if os.name == "nt":
            os.startfile(path)
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])

    def close(self):
        if self.thread and self.thread.is_alive():
            self.closing = True
            self.pause()
            self.status.set("Finishing the current batch before closing safely.")
        else:
            self.root.destroy()


def run_gui(profile, check=False):
    root = tk.Tk()
    if check:
        root.withdraw()
    app = Application(root, profile)
    if check:
        root.update_idletasks()
        result = {"window_constructed": True, "title": root.title(), "offline": True}
        root.destroy()
        print(json.dumps(result))
        return 0
    root.mainloop()
    return 0
