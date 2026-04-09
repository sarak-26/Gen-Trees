import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from .plot_metrics import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_METRICS,
    available_datasets,
    available_metrics,
    available_models,
    build_plot_figure,
    load_runs,
    save_figure,
)

PREVIEW_WIDTH = 1000
PREVIEW_HEIGHT = 700


class PlotMetricsUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Synthetic Data Metric Comparison")
        self.root.geometry("1050x760")

        self.runs = load_runs()
        if not self.runs:
            raise SystemExit("No MLflow evaluation artifacts were found under mlruns/.")

        self.datasets = ["ALL"] + available_datasets(self.runs)
        self.selected_dataset = tk.StringVar(value=self.datasets[0])
        self.status_var = tk.StringVar(value="Choose a dataset, one or more models, and one or more metrics.")
        self.last_output_path: Path | None = None
        self.preview_window: tk.Toplevel | None = None
        self.preview_label: ttk.Label | None = None
        self.preview_image = None

        self._build_layout()
        self._refresh_lists()

    def _build_layout(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Dataset").grid(row=0, column=0, sticky="w")
        dataset_box = ttk.Combobox(
            controls,
            textvariable=self.selected_dataset,
            values=self.datasets,
            state="readonly",
            width=30,
        )
        dataset_box.grid(row=1, column=0, padx=(0, 12), sticky="we")
        dataset_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_lists())

        button_frame = ttk.Frame(controls)
        button_frame.grid(row=1, column=1, padx=(0, 12), sticky="w")
        ttk.Button(button_frame, text="Refresh", command=self._refresh_lists).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_frame, text="Generate Plot", command=self._generate_plot).pack(side=tk.LEFT)

        ttk.Label(controls, text="Models").grid(row=2, column=0, pady=(12, 4), sticky="w")
        ttk.Label(controls, text="Metrics").grid(row=2, column=1, pady=(12, 4), sticky="w")

        self.model_list = tk.Listbox(controls, selectmode=tk.MULTIPLE, exportselection=False, height=10, width=30)
        self.model_list.grid(row=3, column=0, padx=(0, 12), sticky="nsew")

        self.metric_list = tk.Listbox(controls, selectmode=tk.MULTIPLE, exportselection=False, height=16, width=60)
        self.metric_list.grid(row=3, column=1, sticky="nsew")

        action_frame = ttk.Frame(controls)
        action_frame.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="w")
        ttk.Button(action_frame, text="Select All Models", command=lambda: self._select_all(self.model_list)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(action_frame, text="Select Default Metrics", command=self._select_default_metrics).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(action_frame, text="Clear Metrics", command=lambda: self.metric_list.selection_clear(0, tk.END)).pack(side=tk.LEFT)

        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=2)
        controls.rowconfigure(3, weight=1)

        status = ttk.Label(self.root, textvariable=self.status_var, padding=(12, 0))
        status.pack(side=tk.TOP, fill=tk.X)

        help_text = (
            "The plot opens in a fixed-size preview window after you click Generate Plot. "
            "A PNG is also saved in results/."
        )
        self.preview_message = ttk.Label(self.root, text=help_text, padding=12, anchor="center")
        self.preview_message.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _selected_dataset_value(self) -> str | None:
        value = self.selected_dataset.get()
        return None if value == "ALL" else value

    def _refresh_lists(self) -> None:
        dataset = self._selected_dataset_value()
        models = available_models(self.runs, dataset=dataset)
        metrics = available_metrics(self.runs, dataset=dataset)

        self._fill_listbox(self.model_list, models)
        self._fill_listbox(self.metric_list, metrics)
        self._select_default_metrics()
        self.status_var.set(f"Loaded {len(models)} models and {len(metrics)} metrics.")

    def _fill_listbox(self, listbox: tk.Listbox, values: list[str]) -> None:
        listbox.delete(0, tk.END)
        for value in values:
            listbox.insert(tk.END, value)

    def _select_all(self, listbox: tk.Listbox) -> None:
        listbox.selection_set(0, tk.END)

    def _select_default_metrics(self) -> None:
        self.metric_list.selection_clear(0, tk.END)
        metric_values = list(self.metric_list.get(0, tk.END))
        for metric in DEFAULT_METRICS:
            if metric in metric_values:
                index = metric_values.index(metric)
                self.metric_list.selection_set(index)

    def _selected_values(self, listbox: tk.Listbox) -> list[str]:
        return [listbox.get(index) for index in listbox.curselection()]

    def _generate_plot(self) -> None:
        dataset = self._selected_dataset_value()
        models = self._selected_values(self.model_list)
        metrics = self._selected_values(self.metric_list)

        if not models:
            messagebox.showerror("Missing Models", "Select at least one model.")
            return
        if not metrics:
            messagebox.showerror("Missing Metrics", "Select at least one metric.")
            return

        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_name = f"ui_plot_{dataset or 'all'}_{len(models)}models_{len(metrics)}metrics.png"
        output_path = DEFAULT_OUTPUT_DIR / output_name

        try:
            figure = build_plot_figure(
                runs=self.runs,
                dataset=dataset,
                models=models,
                metrics=metrics,
            )
            saved_path = save_figure(figure, output=output_path)
        except Exception as exc:
            messagebox.showerror("Plot Failed", str(exc))
            return

        self.last_output_path = saved_path
        self.status_var.set(f"Saved plot to {saved_path}")
        figure.clf()
        self._show_preview(saved_path)

    def _show_preview(self, image_path: Path) -> None:
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = tk.Toplevel(self.root)
            self.preview_window.title("Plot Preview")
            self.preview_window.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
            self.preview_window.minsize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
            self.preview_window.configure(bg="white")

            self.preview_label = ttk.Label(self.preview_window, anchor="center")
            self.preview_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        image = Image.open(image_path)
        image.thumbnail((PREVIEW_WIDTH - 24, PREVIEW_HEIGHT - 24))
        self.preview_image = ImageTk.PhotoImage(image)

        assert self.preview_label is not None
        self.preview_label.configure(image=self.preview_image, text="")
        self.preview_window.lift()
        self.preview_window.focus_force()


def main() -> None:
    root = tk.Tk()
    PlotMetricsUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
