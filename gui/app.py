"""Tkinter GUI for PyPatchMatch: paint the holes, then inpaint."""

from __future__ import annotations

import contextlib
import math
import secrets
import tkinter as tk
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import TracebackType

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import patchmatch

TITLE = "PyPatchMatch GUI"
VIEW = 640  # initial width and height of both view areas, in screen pixels
MAX_ZOOM = 32  # screen pixels per image pixel
HOLE = (255, 0, 255)
GLOBAL = (0, 255, 255)
MASKS = ("mask", "global_mask")  # keywords of patchmatch.inpaint
Pixel = tuple[int, int]


SHAPES = {
    "round": "\N{BLACK CIRCLE}",
    "square": "\N{BLACK SQUARE}",
    "diamond": "\N{BLACK DIAMOND}",
    "flat": "\N{BLACK RECTANGLE}",
    "upright": "\N{BLACK VERTICAL RECTANGLE}",
}


def extent(shape: str, radius: int) -> Pixel:
    """Half width and half height of the brush, in pixels."""
    thin = radius // 4
    return {"flat": (radius, thin), "upright": (thin, radius)}.get(
        shape, (radius, radius)
    )


def corners(shape: str, box: Sequence[float]) -> list[tuple[float, float]]:
    """The polygon of a brush that is not round, within its bounding box."""
    x0, y0, x1, y1 = box
    if shape == "diamond":
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def oval(shape: str, radius: int) -> bool:
    return shape == "round" and radius > 0  # ellipse draws nothing for radius 0


def stroke(
    draw: ImageDraw.ImageDraw,
    a: Pixel,
    b: Pixel,
    radius: int,
    shape: str,
    fill: int = 255,
) -> None:
    """Stamp the brush on every pixel from a to b; fast drags skip pixels."""
    hx, hy = extent(shape, radius)
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
    for i in range(n + 1):
        x = round(a[0] + (b[0] - a[0]) * i / n)
        y = round(a[1] + (b[1] - a[1]) * i / n)
        box = [x - hx, y - hy, x + hx, y + hy]
        if oval(shape, radius):
            draw.ellipse(box, fill=fill)
        else:
            draw.polygon(corners(shape, box), fill=fill)


def clamp(
    zoom: float, x: float, y: float, image: Pixel, view: Pixel
) -> tuple[float, float, float]:
    """Limit zoom and top left corner so that the view shows only the image."""
    (w, h), (vw, vh) = image, view
    zoom = max(min(zoom, MAX_ZOOM), vw / w, vh / h)
    return zoom, max(min(x, w - vw / zoom), 0), max(min(y, h - vh / zoom), 0)


def show_error(
    _type: type[BaseException], exc: BaseException, _tb: TracebackType | None
) -> None:
    traceback.print_exception(exc)
    messagebox.showerror(TITLE, str(exc))


Item = ttk.Widget | tuple[tk.Menu, str]  # a widget, or a menu entry by its label


def enable(item: Item, on: bool = True) -> None:
    if isinstance(item, tuple):
        menu, label = item
        menu.entryconfig(label, state=tk.NORMAL if on else tk.DISABLED)
    else:
        item.state(["!disabled" if on else tk.DISABLED])


def pane(parent: ttk.Frame, column: int, title: str, hint: str = "") -> tk.Canvas:
    frame = ttk.Frame(parent, padding=8)
    frame.grid(row=3, column=column, sticky="nsew")
    frame.columnconfigure(1, weight=1)
    frame.rowconfigure(1, weight=1)
    ttk.Label(frame, text=title, font="TkHeadingFont").grid(row=0, column=0)
    ttk.Label(frame, text=hint, foreground="gray", font="TkSmallCaptionFont").grid(
        row=0, column=1, sticky="e", padx=(16, 0)
    )
    # place, unlike grid, does not size the area after the canvas
    area = ttk.Frame(frame, width=VIEW, height=VIEW)
    area.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
    canvas = tk.Canvas(area, highlightthickness=0)
    canvas.place(relx=0.5, rely=0.5, anchor="center")
    return canvas


def placeholder(canvas: tk.Canvas, text: str) -> None:
    canvas.delete("all")
    x, y = int(canvas["width"]) / 2, int(canvas["height"]) / 2
    canvas.create_text(x, y, text=text, fill="gray")


class App:
    def __init__(self, root: tk.Tk) -> None:
        root.title(TITLE)
        root.report_callback_exception = show_error
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)
        main.rowconfigure(3, weight=1)
        main.columnconfigure((0, 2), weight=1, uniform="pane")
        # the canvas scrolls the toolbar in a window narrower than it
        strip = tk.Canvas(main, highlightthickness=0)
        strip.grid(row=0, column=0, columnspan=3, sticky="ew")
        scroll = ttk.Scrollbar(main, orient="horizontal", command=strip.xview)
        scroll.grid(row=1, column=0, columnspan=3, sticky="ew")

        def autoscroll(first: str, last: str) -> None:
            scroll.set(first, last)
            if float(first) <= 0 and float(last) >= 1:
                scroll.grid_remove()
            else:
                scroll.grid()

        strip.config(xscrollcommand=autoscroll)
        bar = ttk.Frame(strip, padding=(8, 6))
        strip.create_window(0, 0, window=bar, anchor="nw")
        bar.bind(
            "<Configure>",
            lambda e: strip.config(
                width=e.width, height=e.height, scrollregion=(0, 0, e.width, e.height)
            ),
        )
        ttk.Separator(main).grid(row=2, column=0, columnspan=3, sticky="ew")

        def button(
            text: str,
            command: Callable[[], object],
            state: str = tk.DISABLED,
            **options: object,
        ) -> ttk.Button:
            widget = ttk.Button(bar, text=text, command=command, state=state, **options)
            widget.pack(side="left", padx=2)
            return widget

        def label(text: str) -> None:
            ttk.Label(bar, text=text).pack(side="left", padx=(2, 4))

        def separator() -> None:
            ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        def icon(
            text: str, command: Callable[[], object], state: str = tk.DISABLED
        ) -> ttk.Button:
            # Toolbutton, unlike TButton, has no minimum width on macOS
            return button(text, command, state, style="Toolbutton", width=2)

        icon("\N{OPEN FILE FOLDER}", self.open, tk.NORMAL)
        open_mask = icon("\N{PERFORMING ARTS}", self.open_mask)
        # asks whether the mask or the result is meant; its menu pops up below it,
        # as a ttk.Menubutton looks like a select box on macOS
        save_menu = tk.Menu(bar, tearoff=False)
        save_icon = icon(
            "\N{FLOPPY DISK}",
            lambda: save_menu.tk_popup(
                save_icon.winfo_rootx(),
                save_icon.winfo_rooty() + save_icon.winfo_height(),
            ),
            tk.NORMAL,
        )
        save_menu.add_command(label="Mask…", command=self.save_mask, state=tk.DISABLED)
        save_menu.add_command(label="Result…", command=self.save, state=tk.DISABLED)
        self.needs_image: list[Item] = [open_mask, (save_menu, "Mask…")]
        self.needs_result: list[Item] = [(save_menu, "Result…")]
        self.file_menus(root)
        separator()
        label("Brush")
        self.brush = tk.IntVar(value=20)  # diameter in screen pixels
        ttk.Scale(
            bar,
            from_=1,
            to=100,
            length=100,
            variable=self.brush,
            command=lambda v: self.brush.set(round(float(v))),
        ).pack(side="left")
        ttk.Label(bar, textvariable=self.brush, width=3).pack(side="left", padx=(4, 8))
        self.shape = tk.StringVar(value="round")
        for shape, symbol in SHAPES.items():
            ttk.Radiobutton(
                bar,
                text=symbol,
                value=shape,
                variable=self.shape,
                style="Toolbutton",
                width=2,
            ).pack(side="left")
        separator()
        self.layer = tk.StringVar(value="mask")
        for name in MASKS:
            ttk.Radiobutton(
                bar, text=name, value=name, variable=self.layer, style="Toolbutton"
            ).pack(side="left")
        clear = button("Clear", self.clear)
        detect = button("Detect", self.detect)
        label("\N{GREATER-THAN OR EQUAL TO}")
        self.threshold = tk.IntVar(value=255)  # the holes of inpaint without a mask
        ttk.Spinbox(bar, from_=0, to=255, width=4, textvariable=self.threshold).pack(
            side="left", padx=(0, 4)
        )
        separator()
        label("patch_size")
        self.patch_size = tk.IntVar(value=3)
        ttk.Spinbox(
            bar, from_=1, to=15, width=3, state="readonly", textvariable=self.patch_size
        ).pack(side="left", padx=(0, 4))
        label("seed")
        self.seed = tk.IntVar(value=1212)  # the default of the native library
        seed = ttk.Spinbox(
            bar,
            from_=0,
            to=2**32 - 1,
            width=10,
            textvariable=self.seed,
            state=tk.DISABLED,  # while random_seed is set
        )
        seed.pack(side="left", padx=(0, 4))
        self.random_seed = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bar,
            text="random",
            variable=self.random_seed,
            command=lambda: enable(seed, not self.random_seed.get()),
        ).pack(side="left", padx=4)
        inpaint = button("Inpaint", self.inpaint)
        separator()
        label("Zoom")
        zoom_out = icon("\N{MINUS SIGN}", lambda: self.zoom_at(*self.center(), 0.8))
        zoom_in = icon("+", lambda: self.zoom_at(*self.center(), 1.25))
        self.zoom_label = ttk.Label(bar, width=6, anchor="e")
        self.zoom_label.pack(side="left")
        self.needs_image += [clear, detect, inpaint, zoom_out, zoom_in]

        self.left = pane(main, 0, "Source", "paint: drag · move: right or Shift drag")
        ttk.Separator(main, orient="vertical").grid(row=3, column=1, sticky="ns")
        self.right = pane(main, 2, "Result", "move: drag · zoom: scroll")
        self.left.master.bind("<Configure>", self.resize)  # both areas are equal
        self.left.bind("<ButtonPress-1>", self.press)
        self.left.bind("<B1-Motion>", self.paint)
        self.left.bind("<Motion>", self.hover)
        self.left.bind("<Leave>", self.hover)
        self.left.config(cursor="crosshair")
        self.right.bind("<ButtonPress-1>", self.grab)
        self.right.bind("<B1-Motion>", self.drag)
        for canvas in (self.left, self.right):
            for press, drag in [
                ("<ButtonPress-2>", "<B2-Motion>"),
                ("<ButtonPress-3>", "<B3-Motion>"),
                ("<Shift-ButtonPress-1>", "<Shift-B1-Motion>"),
            ]:
                canvas.bind(press, self.grab)
                canvas.bind(drag, self.drag)
            canvas.bind(
                "<MouseWheel>",
                lambda e: self.zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8),
            )
            # Tk 9 reports trackpads separately, dx and dy in 16 bits each of delta
            with contextlib.suppress(tk.TclError):
                canvas.bind(
                    "<TouchpadScroll>",
                    lambda e: self.zoom_at(
                        e.x, e.y, 1.01 ** (((e.delta & 0xFFFF) ^ 0x8000) - 0x8000)
                    ),
                )
        self.photos: dict[tk.Canvas, ImageTk.PhotoImage] = {}  # Tk keeps no reference
        self.image: Image.Image | None = None
        self.result: Image.Image | None = None
        self.last: Pixel | None = None
        self.pointer: tk.Event[tk.Canvas] | None = None  # over the source view
        self.area = (VIEW, VIEW)

    def file_menus(self, root: tk.Tk) -> None:
        """Fill the File menu and bind its shortcuts."""
        self.file = tk.Menu(root, tearoff=False)
        menubar = tk.Menu(root)
        menubar.add_cascade(label="File", menu=self.file)
        root.config(menu=menubar)
        aqua = root.tk.call("tk", "windowingsystem") == "aqua"
        modifier = "Command" if aqua else "Control"
        for text, command, needs, key in [
            ("Open Image…", self.open, None, "o"),
            ("Open Mask…", self.open_mask, self.needs_image, "O"),
            ("Save Mask…", self.save_mask, self.needs_image, "S"),
            ("Save Result…", self.save, self.needs_result, "s"),
        ]:
            shift = "Shift-" if key.isupper() else ""
            self.file.add_command(
                label=text,
                command=command,
                state=tk.NORMAL if needs is None else tk.DISABLED,
                accelerator=f"{modifier}-{shift}{key.upper()}",
            )
            if needs is not None:
                needs.append((self.file, text))
            # disabled entries ignore invoke
            root.bind(f"<{modifier}-{key}>", lambda _e, t=text: self.file.invoke(t))

    def layout(self) -> None:
        """Make both views as large as their area allows for the image."""
        w, h = self.area if self.image is None else self.image.size
        fit = min(self.area[0] / w, self.area[1] / h)
        self.size = (max(round(w * fit), 1), max(round(h * fit), 1))
        for canvas in (self.left, self.right):
            canvas.config(width=self.size[0], height=self.size[1])
        if self.image is None:
            placeholder(self.left, "Open an image to paint its holes")
        if self.result is None:
            placeholder(self.right, "Inpaint to see the result")

    def resize(self, event: tk.Event[tk.Misc]) -> None:
        self.area = (event.width, event.height)
        if self.image is None:
            self.layout()
            return
        # keep the zoom relative to the whole image and the centre of the views
        relative = self.zoom * self.image.width / self.size[0]
        cx = self.x + self.size[0] / 2 / self.zoom
        cy = self.y + self.size[1] / 2 / self.zoom
        self.layout()
        zoom = relative * self.size[0] / self.image.width
        self.set_view(zoom, cx - self.size[0] / 2 / zoom, cy - self.size[1] / 2 / zoom)

    def open(self) -> None:
        path = filedialog.askopenfilename()
        if not path:
            return
        self.path = Path(path)
        self.image = Image.open(path).convert("RGB")
        self.result = None
        self.layout()
        self.masks = {name: Image.new("L", self.image.size) for name in MASKS}
        self.set_view(0, 0, 0)
        for item in self.needs_image:
            enable(item)
        for item in self.needs_result:
            enable(item, False)

    def set_view(self, zoom: float, x: float, y: float) -> None:
        """Show the same part of the image and of the result in both views."""
        self.zoom, self.x, self.y = clamp(zoom, x, y, self.image.size, self.size)
        self.zoom_label.config(text=f"{self.zoom:.0%}")
        self.base = self.view_of(self.image)
        if self.result is not None:
            self.put(self.right, self.view_of(self.result))
        self.show_mask()

    def view_of(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        box = (
            self.x,
            self.y,
            min(self.x + self.size[0] / self.zoom, w),
            min(self.y + self.size[1] / self.zoom, h),
        )
        # nearest neighbour shows the pixels when zoomed in
        resample = (
            Image.Resampling.NEAREST if self.zoom >= 1 else Image.Resampling.BILINEAR
        )
        return image.resize(self.size, resample, box, reducing_gap=2)

    def show_mask(self) -> None:
        view = self.base.copy()
        view.paste(HOLE, mask=self.view_of(self.masks["mask"]))
        # half opaque, as these pixels keep their values
        keep = self.view_of(self.masks["global_mask"]).point(lambda v: v // 2)
        view.paste(GLOBAL, mask=keep)
        self.put(self.left, view)
        self.show_brush()

    def show_brush(self) -> None:
        """Outline the pixels that the brush would paint under the pointer."""
        self.left.delete("brush")
        if self.image is None or self.pointer is None:
            return
        r, shape = self.radius(), self.shape.get()
        (x, y), (hx, hy) = self.pixel(self.pointer), extent(shape, r)
        box = [
            (x - hx - self.x) * self.zoom,
            (y - hy - self.y) * self.zoom,
            (x + hx + 1 - self.x) * self.zoom,
            (y + hy + 1 - self.y) * self.zoom,
        ]
        if oval(shape, r):
            create, coords = self.left.create_oval, box
        else:
            create, coords = self.left.create_polygon, corners(shape, box)
        for color, width in [("black", 3), ("white", 1)]:
            create(coords, fill="", outline=color, width=width, tags="brush")

    def put(self, canvas: tk.Canvas, image: Image.Image) -> None:
        self.photos[canvas] = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=self.photos[canvas])

    def clear(self) -> None:
        mask = self.masks[self.layer.get()]
        mask.paste(0, (0, 0, *mask.size))
        self.show_mask()

    def pixel(self, event: tk.Event[tk.Canvas]) -> Pixel:
        """The image pixel under the pointer."""
        return (
            math.floor(self.x + (event.x + 0.5) / self.zoom),
            math.floor(self.y + (event.y + 0.5) / self.zoom),
        )

    def radius(self) -> int:
        """The brush radius in image pixels."""
        return round(self.brush.get() / 2 / self.zoom)

    def hover(self, event: tk.Event[tk.Canvas]) -> None:
        self.pointer = event if event.type == tk.EventType.Motion else None
        self.show_brush()

    def press(self, event: tk.Event[tk.Canvas]) -> None:
        if self.image is not None:
            self.last = self.pixel(event)
            self.paint(event)

    def paint(self, event: tk.Event[tk.Canvas]) -> None:
        if self.last is None:
            return
        a, self.last = self.last, self.pixel(event)
        self.pointer = event
        r, shape, layer = self.radius(), self.shape.get(), self.layer.get()
        # a pixel is in the mask painted last only, so the view shows one color
        for name, mask in self.masks.items():
            fill = 255 if name == layer else 0
            stroke(ImageDraw.Draw(mask), a, self.last, r, shape, fill)
        self.show_mask()

    def grab(self, event: tk.Event[tk.Canvas]) -> None:
        self.anchor = (event.x, event.y)
        self.last = None  # releasing Shift must not continue the last stroke

    def drag(self, event: tk.Event[tk.Canvas]) -> None:
        if self.image is None:
            return
        (ax, ay), self.anchor = self.anchor, (event.x, event.y)
        self.pointer = event if event.widget is self.left else None
        dx, dy = (ax - event.x) / self.zoom, (ay - event.y) / self.zoom
        self.set_view(self.zoom, self.x + dx, self.y + dy)

    def center(self) -> tuple[float, float]:
        return self.size[0] / 2, self.size[1] / 2

    def zoom_at(self, x: float, y: float, factor: float) -> None:
        """Zoom both views; the image point at view point (x, y) stays in place."""
        if self.image is None:
            return
        zoom = clamp(self.zoom * factor, 0, 0, self.image.size, self.size)[0]
        s = 1 / self.zoom - 1 / zoom
        self.set_view(zoom, self.x + x * s, self.y + y * s)

    def detect(self) -> None:
        """Add the pixels whose channels all reach the threshold to the holes."""
        # outside the global mask, as patchmatch.inpaint without a mask
        t = self.threshold.get()
        found = Image.fromarray((np.asarray(self.image) >= t).all(axis=2))
        found.paste(0, mask=self.masks["global_mask"])
        if found.getbbox() is None:
            message = f"No pixel outside the global mask has all channels ≥ {t}."
            messagebox.showwarning(TITLE, message)
            return
        self.masks["mask"].paste(255, mask=found)
        self.show_mask()

    def inpaint(self) -> None:
        if self.masks["mask"].getbbox() is None:
            messagebox.showwarning(
                TITLE, "The mask is empty. Paint or detect the holes."
            )
            return
        if self.random_seed.get():
            # shown in the field, to repeat a good fill; secrets avoids Sonar's S2245
            self.seed.set(secrets.randbelow(2**32))
        patchmatch.set_random_seed(self.seed.get())
        # ponytail: blocks the window during the fill, use a thread if fills get slow
        filled = patchmatch.inpaint(
            self.image,
            self.masks["mask"],
            global_mask=self.masks["global_mask"],
            patch_size=self.patch_size.get(),
        )
        self.result = Image.fromarray(filled)
        self.put(self.right, self.view_of(self.result))
        for item in self.needs_result:
            enable(item)

    def mask_file(self) -> str:
        """The default file of the selected mask, next to the image."""
        return f"{self.path.stem}_{self.layer.get()}.png"

    def open_mask(self) -> None:
        """Load the selected mask; like paint, it clears its pixels in the other."""
        path = filedialog.askopenfilename(
            initialdir=self.path.parent, initialfile=self.mask_file()
        )
        if not path:
            return
        mask = Image.open(path).convert("L")
        if mask.size != self.image.size:
            (w, h), (iw, ih) = mask.size, self.image.size
            raise ValueError(f"The mask is {w}x{h} pixels, the image {iw}x{ih}.")
        # every non-zero pixel is set, as for patchmatch.inpaint
        layer = self.layer.get()
        self.masks[layer] = mask.point(lambda v: 255 if v else 0)
        for name, other in self.masks.items():
            if name != layer:
                other.paste(0, mask=self.masks[layer])
        self.show_mask()

    def save_mask(self) -> None:
        """Save the selected mask like examples/images/camera_damaged_mask.png."""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialdir=self.path.parent,
            initialfile=self.mask_file(),
        )
        if path:
            self.masks[self.layer.get()].save(path, optimize=True)

    def save(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            self.result.save(path)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
