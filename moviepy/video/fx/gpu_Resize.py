from cupyx.scipy.ndimage import zoom

def gpu_resize(clip, new_size_ratio):
    """
    Args:
        clip: GPUVideoClip
        new_size_ratio: float (e.g., 0.5 for half size)
    """
    def filter_func(t):
        frame = clip.get_frame(t) # CuPy Array
        # Perform resize on GPU
        # zoom takes (zoom_y, zoom_x, zoom_channels)
        return zoom(frame, (new_size_ratio, new_size_ratio, 1), order=1)

    return clip.with_updated_frame_function(filter_func)