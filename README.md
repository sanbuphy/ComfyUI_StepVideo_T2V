# ComfyUI StepVideo T2V Node

This is a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that integrates the [StepVideo](https://github.com/step-fnc/stepvideo) text-to-video generation model.

## Installation

1. Clone this repository into the `custom_nodes` directory of ComfyUI:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI_StepVideo_T2V.git
```

2. Install the required dependencies:
```bash
cd ComfyUI_StepVideo_T2V
pip install -r requirements.txt
```

## Node Description

This extension provides three main nodes:

### StepVideoModelLoader

Loads the StepVideo model. If the model doesn't exist, it will be automatically downloaded to ComfyUI's model directory.

**Output:**
- `model`: The loaded StepVideo model, which can be connected to the StepVideoT2V node

### StepVideoT2V

Generates videos from text prompts using the StepVideo model.

**Input:**
- `model`: The model obtained from the StepVideoModelLoader node
- `prompt`: Text prompt describing the content of the video to be generated
- `negative_prompt`: Text prompt describing content to avoid
- `num_inference_steps`: Number of inference steps (default: 30)
- `cfg_scale`: Classifier-free guidance scale (default: 9.0)
- `num_frames`: Number of frames to generate (default: 51)
- `seed`: Random seed (default: 42)
- `tiled`: Whether to use tiled generation (default: true)
- `tile_size_x`: Tile size in X direction (default: 34)
- `tile_size_y`: Tile size in Y direction (default: 34)
- `tile_stride_x`: Tile stride in X direction (default: 16)
- `tile_stride_y`: Tile stride in Y direction (default: 16)
- `fps`: Frame rate of the video (default: 25)

**Output:**
- `video`: The generated video, which can be connected to the SaveVideo node

### SaveVideo

Saves the generated video to a specified location.

**Input:**
- `video`: The video to save
- `filename`: The filename to save (without extension)
- `fps`: Frame rate of the video (default: 25)
- `quality`: Video quality (1-10, default: 5)
- `denoise`: Whether to apply denoising (default: true)

## Example Workflow

The repository includes an example workflow file `example_workflows.json` that demonstrates how to use these nodes.

### Loading the Example Workflow

1. In ComfyUI, click the "Load" button in the top right corner
2. Navigate to the `ComfyUI/custom_nodes/ComfyUI_StepVideo_T2V` directory
3. Select the `example_workflows.json` file
