import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import zoom, rotate

# Fix imports based on file structure
from moviepy.video.VideoClip import VideoClip, ImageClip, ColorClip, TextClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.tools import compute_position

# ---------------------------------------------------------
# HELPER: KEYFRAMER
# ---------------------------------------------------------
class Keyframer:
    """Helper to interpolate values over time on the CPU."""
    def __init__(self, keyframes):
        # keyframes should be list of tuples: [(time, value), (time, value)]
        # Sort by time to ensure correct interpolation
        keyframes.sort(key=lambda x: x[0])
        self.times = np.array([k[0] for k in keyframes])

        # Handle tuple values (like x,y coordinates) vs scalar values
        first_val = keyframes[0][1]
        if isinstance(first_val, (tuple, list, np.ndarray)):
            self.is_vector = True
            # Stack into (N, D) array
            self.values = np.array([k[1] for k in keyframes])
        else:
            self.is_vector = False
            self.values = np.array([k[1] for k in keyframes])

    def get_value(self, t):
        # Returns the interpolated value at time t
        if self.is_vector:
            # Interpolate each dimension separately
            # self.values is shape (N_keyframes, Dimensions)
            result = []
            for dim in range(self.values.shape[1]):
                val = np.interp(t, self.times, self.values[:, dim])
                result.append(val)
            return tuple(result)
        else:
            return np.interp(t, self.times, self.values)

# ---------------------------------------------------------
# BASE CLASS: GPUVideoClip
# ---------------------------------------------------------
class GPUVideoClip(VideoClip):
    """
    A drop-in replacement for VideoClip where all pixel data lives on the GPU.
    """

    def __init__(self, frame_function=None, is_mask=False, duration=None):
        super().__init__(frame_function=frame_function, is_mask=is_mask, duration=duration)

    def get_frame(self, t):
        """
        Hijacks the original get_frame.
        Ensures that whatever the frame_function returns is strictly a CuPy array.
        """
        frame = self.frame_function(t)

        # If the frame function returned a CPU array by mistake, move it to GPU
        if isinstance(frame, np.ndarray):
            return cp.asarray(frame)
        return frame


    def compose_on(self, background, t, background_mask=None):
        """
        GPU-safe, MoviePy-safe alpha compositing.
        NEVER returns a cropped mask.
        """
    
        # --- ensure GPU ---
        if not isinstance(background, cp.ndarray):
            background = cp.asarray(background)
    
        if background_mask is not None and not isinstance(background_mask, cp.ndarray):
            background_mask = cp.asarray(background_mask)
    
        # --- get frame ---
        ct = t - self.start
        clip_frame = self.get_frame(ct)
    
        # --- get clip mask ---
        clip_mask = None
        if self.mask:
            clip_mask = self.mask.get_frame(ct)
            if not isinstance(clip_mask, cp.ndarray):
                clip_mask = cp.asarray(clip_mask)
    
        bg_h, bg_w = background.shape[:2]
        img_h, img_w = clip_frame.shape[:2]
    
        # --- force mask to match clip size ---
        if clip_mask is not None:
            mh, mw = clip_mask.shape[:2]
            if mh != img_h or mw != img_w:
                clip_mask = zoom(clip_mask, (img_h / mh, img_w / mw), order=0)
    
        # --- position ---
        pos = self.pos(ct)
        x_start, y_start = compute_position(
            (img_w, img_h), (bg_w, bg_h), pos, self.relative_pos
        )
    
        # --- overlap region (THIS IS KING) ---
        x1_bg = max(x_start, 0)
        y1_bg = max(y_start, 0)
        x2_bg = min(x_start + img_w, bg_w)
        y2_bg = min(y_start + img_h, bg_h)
    
        if x1_bg >= x2_bg or y1_bg >= y2_bg:
            return background.copy(), background_mask
    
        x1_img = x1_bg - x_start
        y1_img = y1_bg - y_start
        h = y2_bg - y1_bg
        w = x2_bg - x1_bg
    
        # --- output frame ---
        bg_copy = background.copy()
    
        # --- fast path ---
        if background_mask is None and clip_mask is None:
            bg_copy[y1_bg:y2_bg, x1_bg:x2_bg] = clip_frame[y1_img:y1_img+h, x1_img:x1_img+w]
            return bg_copy, None
    
        # --- blend ---
        fg = clip_frame[y1_img:y1_img+h, x1_img:x1_img+w].astype(cp.float32)
        bg = bg_copy[y1_bg:y2_bg, x1_bg:x2_bg].astype(cp.float32)
    
        alpha_fg = (
            clip_mask[y1_img:y1_img+h, x1_img:x1_img+w]
            if clip_mask is not None
            else cp.ones((h, w), dtype=cp.float32)
        )
    
        alpha_bg = (
            background_mask[y1_bg:y2_bg, x1_bg:x2_bg]
            if background_mask is not None
            else cp.ones((h, w), dtype=cp.float32)
        )
    
        if alpha_fg.ndim == 2:
            alpha_fg = alpha_fg[..., None]
        if alpha_bg.ndim == 2:
            alpha_bg = alpha_bg[..., None]
    
        out_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
        safe_alpha = cp.where(out_alpha == 0, 1.0, out_alpha)
    
        out_rgb = (fg * alpha_fg + bg * alpha_bg * (1 - alpha_fg)) / safe_alpha
        bg_copy[y1_bg:y2_bg, x1_bg:x2_bg] = cp.round(out_rgb).astype(cp.uint8)
    
        # --- 🚨 THE BOOBY TRAP FIX 🚨 ---
        # NEVER return a cropped mask
        if background_mask is not None:
            out_mask = background_mask.copy()
        else:
            out_mask = cp.zeros((bg_h, bg_w), dtype=cp.float32)
    
        out_mask[y1_bg:y2_bg, x1_bg:x2_bg] = out_alpha.squeeze()
    
        return bg_copy, out_mask


    def to_cpu(self):
        """
        Converts this GPU clip back to a standard MoviePy Clip.
        """
        def cpu_frame_function(t):
            # cp.asnumpy() moves data VRAM -> RAM
            return cp.asnumpy(self.get_frame(t))

        return VideoClip(frame_function=cpu_frame_function,
                         duration=self.duration)

     # ----------------------------------------------
    # GPU NATIVE EFFECTS (Fixed: Applies to Mask)
    # ----------------------------------------------

    def resized(self, ratio):
        """ Returns a new GPU clip resized by ratio """
        def func(get_frame, t):
            return zoom(get_frame(t), (ratio, ratio, 1), order=1)

        new_clip = self.transform(func)

        if self.mask:
            def mask_func(get_frame, t):
                # Mask is 2D (H, W), no channels dim
                return zoom(get_frame(t), (ratio, ratio), order=1)
            new_clip.mask = self.mask.transform(mask_func)

        return new_clip

    def rotated(self, angle):
        """ Returns a new GPU clip rotated by angle """
        def func(get_frame, t):
            return rotate(get_frame(t), angle, axes=(0, 1), reshape=True, order=1)

        new_clip = self.transform(func)

        if self.mask:
            def mask_func(get_frame, t):
                return rotate(get_frame(t), angle, axes=(0, 1), reshape=True, order=1)
            new_clip.mask = self.mask.transform(mask_func)

        return new_clip

    def cropped(self, x1, y1, width, height):
        """ Returns a new GPU clip cropped """
        def func(get_frame, t):
            img = get_frame(t)
            return img[y1:y1+height, x1:x1+width]

        new_clip = self.transform(func)

        if self.mask:
            def mask_func(get_frame, t):
                m = get_frame(t)
                return m[y1:y1+height, x1:x1+width]
            new_clip.mask = self.mask.transform(mask_func)

        return new_clip

    # ----------------------------------------------
    # KEYFRAME ANIMATION METHODS (Fixed: Applies to Mask)
    # ----------------------------------------------

    def set_position_keyframes(self, keyframes):
        """
        Animates position. (Does not affect mask size, just placement)
        Args: keyframes: List of tuples [(time, x, y), ...]
        """
        k = Keyframer([(t, (x,y)) for t,x,y in keyframes])

        def pos_func(t):
            res = k.get_value(t)
            return (int(res[0]), int(res[1]))

        return self.with_position(pos_func)

    def set_scale_keyframes(self, keyframes):
        """
        Animates scale/zoom.
        Args: keyframes: List of tuples [(time, scale_ratio), ...]
        """
        k = Keyframer(keyframes)

        def filter(get_frame, t):
            scale = k.get_value(t)
            return zoom(get_frame(t), (scale, scale, 1), order=1)

        new_clip = self.transform(filter)

        if self.mask:
            def mask_filter(get_frame, t):
                scale = k.get_value(t)
                # Mask is 2D
                return zoom(get_frame(t), (scale, scale), order=1)
            new_clip.mask = self.mask.transform(mask_filter)

        return new_clip

    def set_rotate_keyframes(self, keyframes):
        """
        Animates rotation.
        Args: keyframes: List of tuples [(time, degrees), ...]
        """
        k = Keyframer(keyframes)

        def filter(get_frame, t):
            angle = k.get_value(t)
            return rotate(get_frame(t), angle, axes=(0, 1), reshape=True, order=1)

        new_clip = self.transform(filter)

        if self.mask:
            def mask_filter(get_frame, t):
                angle = k.get_value(t)
                return rotate(get_frame(t), angle, axes=(0, 1), reshape=True, order=1)
            new_clip.mask = self.mask.transform(mask_filter)

        return new_clip

    def set_opacity_keyframes(self, keyframes):
        """
        Animates Opacity (Fade in/out).
        """
        k = Keyframer(keyframes)

        if self.mask is None:
            white_mask = cp.ones((self.h, self.w), dtype=cp.float32)
            self.mask = GPUVideoClip(frame_function=lambda t: white_mask, is_mask=True, duration=self.duration)

        def mask_filter(get_mask_frame, t):
            op = k.get_value(t)
            mask = get_mask_frame(t)
            return mask * op

        new_clip = self.copy()
        new_clip.mask = self.mask.transform(mask_filter)
        return new_clip

# ---------------------------------------------------------
# SUBCLASSES
# ---------------------------------------------------------

class GPUImageClip(GPUVideoClip):
    def __init__(self, filename, **kwargs):
        cpu_clip = ImageClip(filename, **kwargs)
        self.gpu_data = cp.asarray(cpu_clip.img)

        def gpu_frame(t):
            return self.gpu_data

        super().__init__(frame_function=gpu_frame,
                         duration=cpu_clip.duration,
                         is_mask=cpu_clip.is_mask)

        if cpu_clip.mask:
            mask_data = cp.asarray(cpu_clip.mask.get_frame(0))
            self.mask = GPUVideoClip(frame_function=lambda t: mask_data, is_mask=True, duration=cpu_clip.duration)

class GPUVideoFileClip(GPUVideoClip):
    def __init__(self, filename, **kwargs):
        self.cpu_clip = VideoFileClip(filename, **kwargs)

        def gpu_frame_func(t):
            # Read CPU frame -> Move to VRAM
            return cp.asarray(self.cpu_clip.get_frame(t))

        super().__init__(frame_function=gpu_frame_func,
                         duration=self.cpu_clip.duration,
                         is_mask=False)
        self.fps = self.cpu_clip.fps
        self.size = self.cpu_clip.size

    def close(self):
        self.cpu_clip.close()

class GPUColorClip(GPUVideoClip):
    def __init__(self, size, color, duration=None, is_mask=False):
        w, h = size
        temp = ColorClip(size=(1, 1), color=color, is_mask=is_mask)
        color_value = temp.get_frame(0)[0][0]

        if is_mask:
            self.gpu_data = cp.full((h, w), color_value, dtype=cp.float32)
        else:
            pixel_val = cp.array(color_value, dtype=cp.uint8)
            self.gpu_data = cp.empty((h, w, 3), dtype=cp.uint8)
            self.gpu_data[:] = pixel_val

        def gpu_frame_func(t):
            return self.gpu_data

        super().__init__(frame_function=gpu_frame_func,
                         is_mask=is_mask,
                         duration=duration)
        self.size = size

class GPUTextClip(GPUImageClip):
    def __init__(self, text, font, font_size, color, bg_color=None, **kwargs):
        cpu_text = TextClip(
            text=text, font=font, font_size=font_size,
            color=color, bg_color=bg_color, **kwargs
        )
        super().__init__(cpu_text.img, duration=kwargs.get('duration'))
        self.size = cpu_text.size

        if cpu_text.mask:
            mask_data = cp.asarray(cpu_text.mask.get_frame(0))
            self.mask = GPUVideoClip(
                frame_function=lambda t: mask_data,
                is_mask=True,
                duration=self.duration
            )

class GPUCompositeVideoClip(CompositeVideoClip):
    def __init__(self, clips, size=None, bg_color=None, is_mask=False, **kwargs):
        super().__init__(clips, size, bg_color, is_mask, **kwargs)

    def to_cpu(self):
        def cpu_frame_function(t):
            frame = self.get_frame(t)
            if hasattr(frame, 'get'):
                return frame.get()
            elif isinstance(frame, cp.ndarray):
                return cp.asnumpy(frame)
            return frame

        return VideoClip(frame_function=cpu_frame_function, duration=self.duration)
