import asyncio
import tkinter as tk
from tkinter import font, ttk

import numpy as np
from PIL import (
    Image,
    ImageChops,
    ImageColor,
    ImageDraw,
    ImageEnhance,
    ImageOps,
    ImageTk,
)

from ..devices.MightexBufCmos import Camera, Frame
from ..functions.image import get_roi_box, image_math, roi_copy
from .config import Configuration
from .utils import Cyclic, make_task, valid_float, valid_int


class CameraPanel(Cyclic):
    """Camera UI Panel is made of 3 LabelFrames"""

    def __init__(self, parent: ttk.Frame, config: Configuration, camera: Camera | None):
        self.config = config

        # Task variables
        self.tasks: set[asyncio.Task] = set()
        self.extra_init = True

        # UI variables
        self.camera_model = tk.StringVar(value="")
        self.camera_serial = tk.StringVar(value="")
        self.camera_run_mode = tk.IntVar(value=self.config.camera_run_mode)
        self.camera_bits = tk.IntVar(value=self.config.camera_bits)
        self.camera_res_x = tk.StringVar(value=str(self.config.camera_resolution[0]))
        self.camera_res_y = tk.StringVar(value=str(self.config.camera_resolution[1]))
        self.camera_bin_mode = tk.IntVar(value=self.config.camera_bin_mode)
        self.camera_exp_t = tk.StringVar(value=str(self.config.camera_exposure_time))
        self.camera_fps = tk.StringVar(value=str(self.config.camera_fps))
        self.camera_gain = tk.StringVar(value=str(self.config.camera_gain))
        self.camera_freq = tk.StringVar(value=str(32 >> self.config.camera_freq_mode))
        self.img_props = tk.StringVar(value="1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11")
        self.roi_x_entry = tk.StringVar(value=str(self.config.roi_size[0]))
        self.roi_y_entry = tk.StringVar(value=str(self.config.roi_size[1]))
        self.roi_zoom_entry = tk.StringVar(value="10")
        self.full_threshold_entry = tk.StringVar(value=str(config.image_full_threshold))
        self.roi_threshold_entry = tk.StringVar(value=str(config.image_roi_threshold))
        self.use_roi_stats = tk.BooleanVar(value=config.image_use_roi_stats)
        self.full_threshold_hist = tk.BooleanVar(value=False)
        self.roi_threshold_hist = tk.BooleanVar(value=False)
        self.hide_histogram = tk.BooleanVar(value=False)

        # camera variables
        self.camera = camera
        self.roi_zoom = int(self.roi_zoom_entry.get())
        self.update_resolution_flag = False

        # make panel slices
        settings_frame = ttk.LabelFrame(
            parent, text="Camera Settings", labelanchor=tk.N
        )
        self.make_camera_info_slice(settings_frame)
        self.make_camera_mode_slice(settings_frame)
        self.make_camera_resolution_slice(settings_frame)
        self.make_camera_binning_slice(settings_frame)
        self.make_camera_exposure_time_slice(settings_frame)
        self.make_camera_fps_slice(settings_frame)
        self.make_camera_gain_slice(settings_frame)
        self.make_camera_frequency_slice(settings_frame)
        self.make_settings_buttons(settings_frame)
        settings_frame.grid(column=0, row=0, sticky=tk.NSEW)

        full_frame = ttk.LabelFrame(parent, text="Full Frame", labelanchor=tk.N)
        self.make_full_frame_preview_slice(full_frame)
        self.make_image_properties_slice(full_frame)
        # self.make_histogram(full_frame)
        full_frame.grid(column=1, row=0, rowspan=2, sticky=tk.NSEW)

        roi_frame = ttk.LabelFrame(parent, text="Region of Interest", labelanchor=tk.N)
        self.make_roi_input_slice(roi_frame)
        self.make_roi_preview_slice(roi_frame)
        self.make_roi_histogram(roi_frame)
        roi_frame.grid(column=2, row=0, rowspan=4, sticky=tk.NSEW)

        # update UI to match camera object
        self.restore_camera_entries()
        # write settings to camera and then update UI to match camera object
        self.set_cam_ctrl()

    ### Camera Settings Slices ###
    def make_camera_info_slice(self, parent):
        ttk.Label(parent, text="Model").grid(column=0, row=0, sticky=tk.E, padx=10)
        ttk.Label(parent, textvariable=self.camera_model).grid(
            column=1, row=0, columnspan=2, sticky=tk.W
        )
        ttk.Label(parent, text="Serial #").grid(column=0, row=1, sticky=tk.E, padx=10)
        ttk.Label(parent, textvariable=self.camera_serial).grid(
            column=1, row=1, columnspan=2, sticky=tk.W
        )

    def make_camera_mode_slice(self, parent):
        ttk.Label(parent, text="Mode").grid(column=0, row=2, sticky=tk.E, padx=10)
        ttk.Radiobutton(
            parent, text="Stream", value=Camera.NORMAL, variable=self.camera_run_mode
        ).grid(column=1, row=2, sticky=tk.W)
        ttk.Radiobutton(
            parent, text="Trigger", value=Camera.TRIGGER, variable=self.camera_run_mode
        ).grid(column=2, row=2, sticky=tk.W)
        ttk.Radiobutton(parent, text="8 bit", value=8, variable=self.camera_bits).grid(
            column=1, row=3, sticky=tk.W
        )
        ttk.Radiobutton(
            parent, text="12 bit", value=12, variable=self.camera_bits
        ).grid(column=2, row=3, sticky=tk.W)

    def make_camera_resolution_slice(self, parent):
        ttk.Label(parent, text="Resolution").grid(column=0, row=4, sticky=tk.E, padx=10)
        ttk.Entry(
            parent,
            width=5,
            textvariable=self.camera_res_x,
            validatecommand=(parent.register(valid_int), "%P"),
            invalidcommand=parent.register(self.restore_camera_entries),
            validate="focus",
        ).grid(column=1, row=4, sticky=tk.E)
        ttk.Entry(
            parent,
            width=5,
            textvariable=self.camera_res_y,
            validatecommand=(parent.register(valid_int), "%P"),
            invalidcommand=parent.register(self.restore_camera_entries),
            validate="focus",
        ).grid(column=2, row=4, sticky=tk.W)

    def make_camera_binning_slice(self, parent):
        ttk.Label(parent, text="Binning").grid(column=0, row=5, sticky=tk.E, padx=10)
        ttk.Radiobutton(
            parent, text="No Bin", value=Camera.NO_BIN, variable=self.camera_bin_mode
        ).grid(column=1, row=5, columnspan=2, sticky=tk.W)
        ttk.Radiobutton(
            parent, text="1:2", value=Camera.BIN1X2, variable=self.camera_bin_mode
        ).grid(column=1, row=6, sticky=tk.W)
        ttk.Radiobutton(
            parent, text="1:3", value=Camera.BIN1X3, variable=self.camera_bin_mode
        ).grid(column=2, row=6, sticky=tk.W)
        ttk.Radiobutton(
            parent, text="1:4", value=Camera.BIN1X4, variable=self.camera_bin_mode
        ).grid(column=1, row=7, sticky=tk.W)
        ttk.Radiobutton(
            parent, text="1:4 skip", value=Camera.SKIP, variable=self.camera_bin_mode
        ).grid(column=2, row=7, sticky=tk.W)

    def make_camera_exposure_time_slice(self, parent):
        ttk.Label(parent, text="Exp. Time (ms)").grid(
            column=0, row=8, sticky=tk.E, padx=10
        )
        ttk.Entry(
            parent,
            width=5,
            textvariable=self.camera_exp_t,
            validatecommand=(parent.register(valid_float), "%P"),
            invalidcommand=parent.register(self.restore_camera_entries),
            validate="focus",
        ).grid(column=1, row=8, sticky=tk.W)

    def make_camera_fps_slice(self, parent):
        ttk.Label(parent, text="Frames/Sec").grid(column=0, row=9, sticky=tk.E, padx=10)
        ttk.Entry(
            parent,
            width=5,
            textvariable=self.camera_fps,
            validatecommand=(parent.register(valid_float), "%P"),
            invalidcommand=parent.register(self.restore_camera_entries),
            validate="focus",
        ).grid(column=1, row=9, sticky=tk.W)

    def make_camera_gain_slice(self, parent):
        ttk.Label(parent, text="Gain [6-41]dB").grid(
            column=0, row=10, sticky=tk.E, padx=10
        )
        ttk.Entry(
            parent,
            width=5,
            textvariable=self.camera_gain,
            validatecommand=(parent.register(valid_int), "%P"),
            invalidcommand=parent.register(self.restore_camera_entries),
            validate="focus",
        ).grid(column=1, row=10, sticky=tk.W)

    def make_camera_frequency_slice(self, parent):
        ttk.Label(parent, text="Frequency (MHz)").grid(
            column=0, row=11, sticky=tk.E, padx=10
        )
        ttk.Combobox(
            parent,
            textvariable=self.camera_freq,
            values=["32", "16", "8", "4", "2"],
            state="readonly",
            width=4,
        ).grid(column=1, row=11, sticky=tk.W)

    def make_settings_buttons(self, parent):
        ttk.Button(parent, text="Write to Camera", command=self.set_cam_ctrl).grid(
            column=0, row=12, columnspan=3, pady=(10, 0), padx=10, sticky=tk.S
        )
        # # NOTE: secret reset button
        # ttk.Button(parent, text="Reset Camera", command=self.reset_camera).grid(
        #     column=0, row=13, columnspan=3, pady=(10, 0), padx=10
        # )

    ### Full Frame Slices ###
    def make_full_frame_preview_slice(self, parent):
        self.full_frame_preview = ttk.Label(parent)
        self.full_frame_preview.grid(
            column=0, row=0, columnspan=3, rowspan=11, sticky=tk.N
        )

    def make_image_properties_slice(self, parent):
        l = ttk.Label(parent, textvariable=self.img_props)
        l.grid(column=3, row=0, rowspan=11, columnspan=2, padx=10, sticky=tk.NW)

    def make_histogram(self, parent):
        self.histogram = tk.Canvas(parent, width=500, height=200)
        self.histogram.grid(column=0, row=12, columnspan=3, sticky=tk.W)
        ttk.Checkbutton(
            parent,
            text="Limit histogram to threshold",
            variable=self.full_threshold_hist,
        ).grid(column=0, row=13, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(
            parent, text="Hide Histogram", variable=self.hide_histogram
        ).grid(column=2, row=13)

    ### ROI Frame Slices ###
    def make_roi_input_slice(self, parent):
        input_frame = ttk.Frame(parent)
        ttk.Label(input_frame, text="Bounds").grid(column=0, row=0, padx=10)
        ttk.Entry(
            input_frame,
            width=5,
            textvariable=self.roi_x_entry,
            validatecommand=(parent.register(valid_int), "%P"),
            invalidcommand=(
                parent.register(self.roi_x_entry.set),
                self.camera_res_x.get(),
            ),
            validate="focus",
        ).grid(column=1, row=0, sticky=tk.E)
        ttk.Entry(
            input_frame,
            width=5,
            textvariable=self.roi_y_entry,
            validatecommand=(parent.register(valid_int), "%P"),
            invalidcommand=(
                parent.register(self.roi_y_entry.set),
                self.camera_res_y.get(),
            ),
            validate="focus",
        ).grid(column=2, row=0, sticky=tk.W)
        ttk.Label(input_frame, text="Zoom").grid(column=0, row=1, padx=10)
        ttk.Entry(
            input_frame,
            width=5,
            textvariable=self.roi_zoom_entry,
            validatecommand=(parent.register(valid_int), "%P"),
            invalidcommand=(parent.register(self.roi_zoom_entry.set), str(1)),
            validate="focus",
        ).grid(column=1, row=1, sticky=tk.E)
        ttk.Button(input_frame, text="Set ROI", command=self.set_roi).grid(
            column=3, row=0, rowspan=2, pady=(10, 0), padx=10
        )
        input_frame.grid(column=0, row=0, sticky=tk.NW)

    def make_roi_preview_slice(self, parent):
        roi_prev_frame = ttk.Frame(parent)
        self.roi_preview = ttk.Label(roi_prev_frame)
        self.roi_preview.grid(column=0, row=0)
        # cross-cuts
        self.cc_x = tk.Canvas(roi_prev_frame)
        self.cc_x.grid(column=0, row=1)
        self.cc_y = tk.Canvas(roi_prev_frame)
        self.cc_y.grid(column=1, row=0)
        roi_prev_frame.grid(column=0, row=1)

    def make_roi_histogram(self, parent):
        self.roi_histogram = tk.Canvas(parent, width=500, height=200)
        self.roi_histogram.grid(column=0, row=2, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(
            parent,
            text="Limit histogram to threshold",
            variable=self.roi_threshold_hist,
        ).grid(column=0, row=3, columnspan=2, sticky=tk.W)

    ### Functions ###
    def set_cam_ctrl(self):
        """Set camera to new settings"""
        if self.camera:
            make_task(self.set_cam_ctrl_async(), self.tasks)

    async def set_cam_ctrl_async(self):
        if self.camera:
            # set up camera object
            await self.camera.set_mode(
                run_mode=self.camera_run_mode.get(), bits=self.camera_bits.get()
            )
            self.config.camera_run_mode = self.camera_run_mode.get()
            self.config.camera_bits = self.camera_bits.get()
            resolution = (int(self.camera_res_x.get()), int(self.camera_res_y.get()))
            await self.camera.set_resolution(
                resolution=resolution, bin_mode=self.camera_bin_mode.get()
            )
            self.config.camera_resolution = resolution
            self.config.camera_bin_mode = self.camera_bin_mode.get()
            await self.camera.set_exposure_time(
                exposure_time=float(self.camera_exp_t.get())
            )
            self.config.camera_exposure_time = float(self.camera_exp_t.get())
            await self.camera.set_fps(fps=float(self.camera_fps.get()))
            self.config.camera_fps = float(self.camera_fps.get())
            await self.camera.set_gain(gain=int(self.camera_gain.get()))
            self.config.camera_gain = int(self.camera_gain.get())
            freq_div = int.bit_length(32 // int(self.camera_freq.get())) - 1
            await self.camera.set_frequency(freq_mode=freq_div)
            self.config.camera_freq_mode = freq_div

            # write to camera, clear buffer, update UI
            await self.camera.write_configuration()
            await self.camera.clear_buffer()
            self.restore_camera_entries()

    def snap_img(self):
        """Snap an image"""
        if self.camera:
            make_task(self.camera.trigger(), self.tasks)

    def restore_camera_entries(self):
        """Restore camera entry boxes from camera

        except resolution which is set from update because it
        needs current frame information
        """
        if self.camera:
            self.camera_run_mode.set(self.camera.run_mode)
            self.config.camera_run_mode = self.camera_run_mode.get()
            self.camera_bin_mode.set(self.camera.bin_mode)
            self.config.camera_bin_mode = self.camera_bin_mode.get()
            self.camera_exp_t.set(str(self.camera.exposure_time))
            self.config.camera_exposure_time = float(self.camera_exp_t.get())
            self.camera_fps.set(str(self.camera.fps))
            self.config.camera_fps = float(self.camera_fps.get())
            self.camera_gain.set(str(self.camera.gain))
            self.config.camera_gain = int(self.camera_gain.get())
            self.camera_freq.set(str(32 >> self.camera.freq_mode))
            self.config.camera_freq_mode = (
                int.bit_length(32 // int(self.camera_freq.get())) - 1
            )

            # update resolution
            self.update_resolution_flag = True

    def set_camera_info(self, info: dict[str, str]):
        """Update the GUI with the camera info"""
        self.camera_model.set(info["ModuleNo"])
        self.camera_serial.set(info["SerialNo"])

    def reset_camera(self):
        if self.camera:
            make_task(self.camera.reset(), self.tasks)

    def set_thresholds(self):
        self.config.image_full_threshold = float(self.full_threshold_entry.get())
        self.config.image_roi_threshold = float(self.roi_threshold_entry.get())
        self.config.image_use_roi_stats = self.use_roi_stats.get()

    def restore_thresholds(self):
        self.full_threshold_entry.set(str(self.config.image_full_threshold))
        self.roi_threshold_entry.set(str(self.config.image_roi_threshold))

    def set_roi(self):
        """Set the region of interest bounds and zoom"""

        # get inputs and clip to valid, then write back valid values
        size_x = min(max(int(self.roi_x_entry.get()), 1), int(self.camera_res_x.get()))
        size_y = min(max(int(self.roi_y_entry.get()), 1), int(self.camera_res_y.get()))
        self.config.roi_size = (size_x, size_y)
        self.roi_x_entry.set(str(size_x))
        self.roi_y_entry.set(str(size_y))
        self.roi_zoom = int(self.roi_zoom_entry.get())
        self.update_full_frame_preview()
        self.update_roi_img()

    def get_roi_box(self) -> tuple[int, int, int, int]:
        """Return the (left, lower, right, upper)
        box representing the region of interest
        """
        f_size_x = self.config.full_img.size[0]
        f_size_y = self.config.full_img.size[1]
        return get_roi_box((f_size_x, f_size_y), self.config.roi_size)

    def update_roi_img(self):
        """Update the region of interest image"""

        # crop to ROI, then blow up ROI by zoom factor
        box = self.get_roi_box()
        x = box[2] - box[0]
        y = box[3] - box[1]
        z = self.roi_zoom
        roi_img = self.config.full_img.crop(box)
        zoomed = roi_img.resize(
            size=(z * x, z * y), resample=Image.Resampling.NEAREST
        ).convert("RGB")

        # draw crosshairs
        ImageDraw.Draw(zoomed).line(
            [(0, z * (y // 2) + z // 2), (z * x, z * (y // 2) + z // 2)],
            fill=ImageColor.getrgb("yellow"),
        )
        ImageDraw.Draw(zoomed).line(
            [(z * (x // 2) + z // 2, 0), (z * (x // 2) + z // 2, z * y)],
            fill=ImageColor.getrgb("yellow"),
        )

        # draw FWHM
        # NOTE: when drawing the zoomed-in view for the ROI, add half a pixel to both x & y.
        #       Pixels are drawn as boxes, where their position is nominally their top-left
        #       corner, so adding half a pixel to both dimensions puts the pixel position in
        #       the center of the pixel, which looks better.
        hwhm = self.config.image_fwhm / 2
        ImageDraw.Draw(zoomed).ellipse(
            (
                z * (self.config.image_centroid[0] - hwhm - box[0] + 1 / 2),
                z * (self.config.image_centroid[1] - hwhm - box[1] + 1 / 2),
                z * (self.config.image_centroid[0] + hwhm - box[0] + 1 / 2),
                z * (self.config.image_centroid[1] + hwhm - box[1] + 1 / 2),
            ),
            outline=ImageColor.getrgb("red"),
        )
        disp_img = ImageTk.PhotoImage(zoomed)
        self.roi_preview.img = disp_img  # type: ignore # protect from garbage collect
        self.roi_preview.configure(image=disp_img)
        self.update_crosscuts()

    def update_crosscuts(self):
        """Update cross cuts"""

        # delete drawings from last update
        self.cc_x.delete("all")
        self.cc_y.delete("all")

        # extract cross-cuts from full image (8 bit)
        roi_box = self.get_roi_box()
        x_cut = np.array(
            self.config.full_img.crop(
                (
                    roi_box[0],
                    self.config.full_img.size[1] // 2,
                    roi_box[2],
                    self.config.full_img.size[1] // 2 + 1,
                )
            )
        ).flatten()
        y_cut = np.array(
            self.config.full_img.crop(
                (
                    self.config.full_img.size[0] // 2,
                    roi_box[1],
                    self.config.full_img.size[0] // 2 + 1,
                    roi_box[3],
                )
            )
        ).flatten()
        max_pixel = (1 << 8) - 1

        # set size of cross-cuts to match roi image
        self.cc_x.configure(
            width=self.roi_preview.img.width(), height=100  # type: ignore
        )
        self.cc_y.configure(
            width=100, height=self.roi_preview.img.height()  # type: ignore
        )

        # NOTE: it seems Tk adds a 2-pixel all-around padding between the label border and the image,
        #       so this is to account for that so things line up with the cross-cuts.
        offset = 2
        bar_width_x = (self.roi_preview.img.width()) // len(x_cut)  # type: ignore
        bar_width_y = (self.roi_preview.img.height()) // len(y_cut)  # type: ignore

        for i in range(len(x_cut)):
            self.cc_x.create_rectangle(
                (
                    i * bar_width_x + offset,
                    0,
                ),
                (
                    (i + 1) * bar_width_x + offset,
                    self.cc_x.winfo_reqheight() * x_cut[i] / max_pixel,
                ),
                fill="gray",
            )
        for i in range(len(y_cut)):
            self.cc_y.create_rectangle(
                (
                    0,
                    i * bar_width_y + offset,
                ),
                (
                    self.cc_y.winfo_reqwidth() * y_cut[i] / max_pixel,
                    (i + 1) * bar_width_y + offset,
                ),
                fill="gray",
            )

        # guidelines
        # NOTE: the -3 keeps it on the canvas, probably something to do with borders
        saturation_height = self.cc_x.winfo_reqheight() - 3
        saturation_width = self.cc_y.winfo_reqwidth() - 3
        threshold = (
            self.config.image_roi_threshold
            if self.config.image_use_roi_stats
            else self.config.image_full_threshold
        )
        threshold_height = saturation_height * threshold / 100
        threshold_width = saturation_width * threshold / 100
        self.cc_x.create_line(
            0,
            saturation_height,
            self.cc_x.winfo_reqwidth(),
            saturation_height,
            fill="red",
        )
        self.cc_y.create_line(
            saturation_width,
            0,
            saturation_width,
            self.cc_y.winfo_reqheight(),
            fill="red",
        )
        self.cc_x.create_line(
            0,
            threshold_height,
            self.cc_x.winfo_reqwidth(),
            threshold_height,
            fill="blue",
        )
        self.cc_y.create_line(
            threshold_width,
            0,
            threshold_width,
            self.cc_y.winfo_reqheight(),
            fill="blue",
        )

    def update_image_props(self, camera_frame: Frame):
        """Update image properties"""
        prop_str = ""
        for p in [
            "bits",
            "rows",
            "cols",
            "bin",
            "gGain",
            "expTime",
            "frameTime",
            "timestamp",
            "triggered",
            "nTriggers",
            "freq",
        ]:
            prop_str += str(p) + ": " + str(getattr(camera_frame, p)) + "\n"
        self.img_props.set(prop_str.strip())

    def update_image_stats(self):
        """Update image statistics"""

        # do image math unless it's being computed in another function
        if not self.config.image_math_in_function:
            # if the camera is present, use its frame;
            # otherwise, use "full_img" which is likely a simulated image
            if self.config.camera_frame:
                image = self.config.camera_frame.img_array
                bits = self.config.camera_frame.bits
            else:
                image = np.array(self.config.full_img)
                bits = 8

            # use ROI if selected
            if self.config.image_use_roi_stats:
                image = roi_copy(image, self.config.roi_size)
                threshold = self.config.image_roi_threshold
            else:
                threshold = self.config.image_full_threshold

            centroid, fwhm, max_value, n_saturated = image_math(
                image, bits, threshold, self.config.image_fwhm_method
            )

            # translate to full-frame pixel coordinates
            if self.config.image_use_roi_stats:
                box = self.get_roi_box()
                centroid = (centroid[0] + box[0], centroid[1] + box[1])

            # store values
            self.config.image_centroid = centroid
            self.config.image_fwhm = fwhm
            self.config.image_max_value = max_value
            self.config.image_n_saturated = n_saturated

    def update_full_frame_preview(self):
        """Update the full frame preview"""
        # draw roi box
        roi_box = self.get_roi_box()
        img = self.config.full_img.convert("RGB")
        ImageDraw.Draw(img).rectangle(
            roi_box, width=3, outline=ImageColor.getrgb("yellow")
        )

        # draw FWHM if it exists
        if not (
            np.isnan(self.config.image_centroid[0])
            or np.isnan(self.config.image_centroid[1])
        ):
            hwhm = self.config.image_fwhm / 2
            if np.isnan(hwhm):
                hwhm = 0
            ImageDraw.Draw(img).ellipse(
                (
                    self.config.image_centroid[0] - hwhm,
                    self.config.image_centroid[1] - hwhm,
                    self.config.image_centroid[0] + hwhm,
                    self.config.image_centroid[1] + hwhm,
                ),
                width=3,
                outline=ImageColor.getrgb("red"),
            )
        # display, scale to screen resolution
        # NOTE: arbitrary width of 6 * 12 characters of size 6 font, height scaled to same ratio
        font_width = font.Font(font="TkDefaultFont 6").measure(text="123456")
        ratio = round(img.width / (font_width * 12))
        disp_img = ImageTk.PhotoImage(
            img.resize((img.width // ratio, img.height // ratio))
        )
        self.full_frame_preview.img = disp_img  # type: ignore # protect from garbage collect
        self.full_frame_preview.configure(image=disp_img)

    def update_histogram(
        self,
        histogram_canvas: tk.Canvas,
        img_array: np.ndarray,
        bits: int,
        threshold: float = 50.0,
        threshold_en: bool = False,
    ):
        """Draw histogram and labels

        histogram_canvas: canvas to draw to
        img_array: image to analyze
        bits: number of bits per pixel
        threshold: threshold percentage
        threshold_en: enable threshold limit for histogram
        """
        # measure font size and set canvas size
        f = font.Font(font="TkDefaultFont 6")
        font_width = f.measure(text="123456")
        font_height = f.metrics("linespace")
        histogram_canvas.configure(width=(font_width * 12), height=(font_height * 13))

        # delete drawings from last update
        histogram_canvas.delete("all")

        # make text labels, reserve margin space for text
        margin_h = font_height * 2
        margin_v = font_height * 3
        histogram_canvas.create_text(
            histogram_canvas.winfo_reqwidth() / 2,
            0,
            anchor="n",
            text="Histogram of Pixel Values",
        )
        histogram_canvas.create_text(
            histogram_canvas.winfo_reqwidth() / 2,
            histogram_canvas.winfo_reqheight(),
            anchor="s",
            text="pixel value",
        )
        histogram_canvas.create_text(
            0,
            histogram_canvas.winfo_reqheight() / 2,
            angle=90,
            anchor="n",
            text="# of pixels",
        )

        max_pixel = (1 << bits) - 1

        # get threshold value and apply if enabled
        t_val = max_pixel * threshold / 100
        if threshold_en:
            # NOTE: this removes the elements below the threshold and flattens the array to 1-D
            img_array = img_array[img_array > t_val]

        # get histogram data and compute bar width
        values, edges = np.histogram(img_array, range=(0, max_pixel))
        bar_width = (histogram_canvas.winfo_reqwidth() - 2 * margin_h) / len(values)

        for i in range(len(values)):
            if max(values) != 0:
                histogram_canvas.create_rectangle(
                    (
                        i * bar_width + margin_h,
                        histogram_canvas.winfo_reqheight() - margin_v,
                    ),
                    (
                        (i + 1) * bar_width + margin_h,
                        histogram_canvas.winfo_reqheight()
                        - margin_v
                        - (histogram_canvas.winfo_reqheight() - 2 * margin_v)
                        * values[i]
                        / max(values),
                    ),
                    fill="gray",
                )
                histogram_canvas.create_text(
                    i * bar_width + margin_h,
                    histogram_canvas.winfo_reqheight()
                    - margin_v
                    - (histogram_canvas.winfo_reqheight() - 2 * margin_v)
                    * values[i]
                    / max(values),
                    anchor="nw",
                    text=f"{values[i]}",
                    font="TkDefaultFont 6",
                    fill="white",
                )
            histogram_canvas.create_text(
                i * bar_width + margin_h,
                histogram_canvas.winfo_reqheight() - margin_v,
                anchor="n",
                text=edges[i],
                font="TkDefaultFont 6",
            )
        # last bar's end
        histogram_canvas.create_text(
            len(values) * bar_width + margin_h,
            histogram_canvas.winfo_reqheight() - margin_v,
            anchor="n",
            text=edges[-1],
            font="TkDefaultFont 6",
        )
        # threshold line
        t_x = (
            t_val * (histogram_canvas.winfo_reqwidth() - 2 * margin_h) / max_pixel
            + margin_h
        )
        histogram_canvas.create_line(
            t_x,
            histogram_canvas.winfo_reqheight() - margin_v,
            t_x,
            margin_v,
            fill="blue",
        )
        histogram_canvas.create_text(
            t_x,
            margin_v,
            anchor="s",
            text=f"{t_val}",
            font="TkDefaultFont 6",
            fill="blue",
        )

    def update_all_histograms(self):
        """Update full frame and ROI histograms"""
        # if self.hide_histogram.get():
        #     self.histogram.grid_remove()
        # else:
        #     self.histogram.grid()

        box = self.get_roi_box()
        if self.config.camera_frame:
            # self.update_histogram(
            #     self.histogram,
            #     self.config.camera_frame.img_array,
            #     self.config.camera_frame.bits,
            #     self.config.image_full_threshold,
            #     self.full_threshold_hist.get(),
            # )
            self.update_histogram(
                self.roi_histogram,
                self.config.camera_frame.img_array[box[1] : box[3], box[0] : box[2]],
                self.config.camera_frame.bits,
                self.config.image_roi_threshold,
                self.roi_threshold_hist.get(),
            )
        else:
            # self.update_histogram(
            #     self.histogram,
            #     np.array(self.config.full_img),
            #     8,
            #     self.config.image_full_threshold,
            #     self.full_threshold_hist.get(),
            # )
            self.update_histogram(
                self.roi_histogram,
                np.array(self.config.full_img.crop(box)),
                8,
                self.config.image_roi_threshold,
                self.roi_threshold_hist.get(),
            )

    async def update_resolution(self):
        """Update resolution from camera, matching newest frame"""
        resolution: tuple[int, int] = (await self.camera.query_buffer())["resolution"]  # type: ignore
        self.camera_res_x.set(str(resolution[0]))
        self.camera_res_y.set(str(resolution[1]))
        self.config.camera_resolution = resolution
        self.update_resolution_flag = False

    async def update(self):
        """Update preview image in viewer"""
        if not self.config.image_frozen:
            if self.camera:
                # set camera info on first pass
                if self.extra_init:
                    self.set_camera_info(await self.camera.get_camera_info())
                    self.extra_init = False

                try:
                    self.config.camera_frame = self.camera.get_newest_frame()
                    self.config.full_img = Image.fromarray(
                        self.config.camera_frame.display_array
                    )
                    self.update_image_props(self.config.camera_frame)
                    if self.update_resolution_flag:
                        make_task(self.update_resolution(), self.tasks)
                except IndexError:
                    pass
            else:  # no camera, testing purposes
                # build an image from components
                res = (int(self.camera_res_x.get()), int(self.camera_res_y.get()))
                bk = Image.new(mode="L", size=res, color=0)  # black background
                noise = ImageEnhance.Brightness(
                    Image.effect_noise(size=res, sigma=50)  # dark noise
                ).enhance(0.5)
                gradient = ImageOps.invert(
                    Image.radial_gradient(mode="L")  # gradient spot
                )
                bk.paste(gradient, (85, 400))
                # dark noise with gradient spot overlay
                self.config.full_img = ImageChops.add(noise, bk, 1.5, 10)

            self.update_image_stats()
            self.update_full_frame_preview()
            self.update_roi_img()
            self.update_all_histograms()

    def close(self):
        """Close out all tasks"""
        for t in self.tasks:
            t.cancel()
