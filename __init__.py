import os
import sys
import folder_paths

# Add current directory to system path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Import node classes from our node module
from stepvideo_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# This adds our nodes to ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 