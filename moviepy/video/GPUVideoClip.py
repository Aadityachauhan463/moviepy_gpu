import numpy as np
import cupy as cp
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# ---------------------------------------------------------
# FIX: 'videoclip' -> 'video.VideoClip'
# ---------------------------------------------------------
from moviepy.video.VideoClip import VideoClip, ImageClip

# This one is correct (tools.py is at moviepy/tools.py)
from moviepy.tools import compute_position

# If you are using the effects (resize/rotate) we discussed earlier, 
# you will also need this:
from cupyx.scipy.ndimage import zoom, rotate

class GPUVideoClip(VideoClip):
    """
    A VideoClip where all frame data lives in VRAM (GPU Memory).
    """

    def __init__(self, frame_function=None, is_mask=False, duration=None):
        super().__init__(frame_function, is_mask, duration)
        
    def to_cpu(self):
        """Helper to return a standard MoviePy clip (moves data to RAM)"""
        def cpu_frame_func(t):
            frame = self.get_frame(t)
            # cp.asnumpy moves data from GPU -> CPU
            return cp.asnumpy(frame) 
        
        return VideoClip(frame_function=cpu_frame_func, 
                         is_mask=self.is_mask, 
                         duration=self.duration)

    # We override the core compositing method from lines 520-665 of your file
    def compose_on(self, background, t, background_mask=None):
        # 1. Ensure Background is on GPU
        if not isinstance(background, cp.ndarray):
            background = cp.asarray(background)
            
        if background_mask is not None and not isinstance(background_mask, cp.ndarray):
            background_mask = cp.asarray(background_mask)

        ct = t - self.start
        
        # 2. Get the GPU Frame
        clip_frame = self.get_frame(ct) # This should return a CuPy array
        
        # Validate data types
        if clip_frame.dtype != cp.uint8 and not self.is_mask:
            clip_frame = clip_frame.astype(cp.uint8)

        # 3. Handle Masks (GPU)
        clip_mask = None
        if self.mask is not None:
            clip_mask = self.mask.get_frame(ct)
            if not isinstance(clip_mask, cp.ndarray):
                clip_mask = cp.asarray(clip_mask)

        # Dimensions
        bg_h, bg_w = background.shape[:2]
        img_h, img_w = clip_frame.shape[:2]

        # 4. Calculate Position (CPU is fine for this math)
        pos = self.pos(ct)
        x_start, y_start = compute_position(
            (img_w, img_h), (bg_w, bg_h), pos, self.relative_pos
        )

        # Calculate slicing coordinates
        y1_bg = max(y_start, 0)
        y2_bg = min(y_start + img_h, bg_h)
        x1_bg = max(x_start, 0)
        x2_bg = min(x_start + img_w, bg_w)

        y1_img = max(-y_start, 0)
        y2_img = y1_img + (y2_bg - y1_bg)
        x1_img = max(-x_start, 0)
        x2_img = x1_img + (x2_bg - x1_bg)

        # 5. The Heavy Lifting: Compositing on GPU
        # We copy the background so we don't mutate the original
        bg_copy = background.copy()
        
        # If no masks, fast Blit
        if background_mask is None and clip_mask is None:
            bg_copy[y1_bg:y2_bg, x1_bg:x2_bg] = clip_frame[y1_img:y2_img, x1_img:x2_img]
            return bg_copy, None

        # Prepare for float math
        frame_region = clip_frame[y1_img:y2_img, x1_img:x2_img].astype(cp.float32)
        bg_region = bg_copy[y1_bg:y2_bg, x1_bg:x2_bg].astype(cp.float32)
        
        # Handle Alpha Mixing
        # This mirrors the logic in your file lines 646+ but uses CuPy
        
        alpha_clip = clip_mask[y1_img:y2_img, x1_img:x2_img] if clip_mask is not None else cp.ones((y2_img-y1_img, x2_img-x1_img))
        alpha_bg = background_mask[y1_bg:y2_bg, x1_bg:x2_bg] if background_mask is not None else cp.ones((y2_bg-y1_bg, x2_bg-x1_bg))

        # Ensure dimensions match for broadcasting
        if alpha_clip.ndim == 2: alpha_clip = alpha_clip[..., None]
        if alpha_bg.ndim == 2: alpha_bg = alpha_bg[..., None]

        # Porter-Duff 'Over' Composition
        final_alpha = alpha_clip + alpha_bg * (1 - alpha_clip)
        safe_alpha = cp.where(final_alpha == 0, 1.0, final_alpha)
        
        result = (frame_region * alpha_clip + bg_region * alpha_bg * (1 - alpha_clip)) / safe_alpha

        # Write back to result
        bg_copy[y1_bg:y2_bg, x1_bg:x2_bg] = cp.round(result).astype(cp.uint8)
        
        # Return result and the new mask (squeezed back to 2D)
        return bg_copy, final_alpha.squeeze()



class GPUImageClip(GPUVideoClip):
    def __init__(self, img, **kwargs):
        # Initialize the standard ImageClip to load the file
        temp_clip = ImageClip(img, **kwargs)
        
        # Extract the numpy array and move it to GPU immediately
        self.gpu_img = cp.asarray(temp_clip.img)
        
        # Re-define the frame function to return the GPU array
        def gpu_frame_func(t):
            return self.gpu_img
            
        super().__init__(frame_function=gpu_frame_func, 
                         is_mask=temp_clip.is_mask, 
                         duration=temp_clip.duration)
        
        # Handle Mask if it exists
        if temp_clip.mask:
            # We assume the mask is static for an image clip
            mask_data = temp_clip.mask.get_frame(0)
            self.mask = GPUVideoClip(
                frame_function=lambda t: cp.asarray(mask_data),
                is_mask=True,
                duration=temp_clip.duration
            )


class GPUCompositeVideoClip(CompositeVideoClip):
    """
    A CompositeVideoClip that understands GPU clips.
    It inherits all the logic from standard MoviePy, but adds the ability
    to export back to CPU.
    """
    def __init__(self, clips, size=None, bg_color=None, is_mask=False, **kwargs):
        super().__init__(clips, size, bg_color, is_mask, **kwargs)

    def to_cpu(self):
        """
        Converts the GPU composite result back to a standard CPU clip 
        for FFmpeg writing.
        """
        def cpu_frame_function(t):
            # Get the frame (which comes out as a CuPy array because children are GPU)
            frame = self.get_frame(t)
            
            # If it's a CuPy array, convert to NumPy
            if hasattr(frame, 'get'): 
                return frame.get() # .get() is the CuPy way to move to CPU
            elif isinstance(frame, cp.ndarray):
                return cp.asnumpy(frame)
            
            return frame
            
        return VideoClip(frame_function=cpu_frame_func, duration=self.duration)
