<h1 align="center">DeepDepth</h1>
The code performs monocular depth estimation using the MiDaS model. It preprocesses an input image, estimates its depth, resizes the depth map, applies a color heatmap for visualization, and saves the result. PyTorch handles the model, while OpenCV manages image processing tasks.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install numpy torch torchvision opencv-python
   ```

2. Enter the path of the input image and where you want to save the result.

3. Run the code and it will provide the result.

## Result:

  Input Image:

  ![image](https://github.com/user-attachments/assets/14fc2c76-7201-4ff3-a14d-c5b4b3967418)

  Output Image:

  ![output](https://github.com/user-attachments/assets/7bc8d015-8d13-4f1c-9469-301496e7ebf0)

## Overview:
This code implements a **monocular depth estimation pipeline** using the MiDaS (Monocular Depth Sensing) model from Intel ISL. The pipeline processes a single input image and outputs a heatmap representing the depth information. Here is a detailed overview:

### **1. Import Required Libraries**
- **`numpy`**: For numerical computations (not explicitly used here but commonly included).
- **`torch`**: For working with the PyTorch framework to load and run the MiDaS model.
- **`cv2` (OpenCV)**: For image processing tasks like reading, transforming, resizing, and saving images.

### **2. Load MiDaS Model**
**Function: `load_midas_model()`:**
  - Loads the MiDaS model using `torch.hub.load`. The `DPT_Large` model type is used, providing high accuracy at the cost of computational resources.
  - Determines whether a **GPU** (if available) or **CPU** will be used for computation.
  - Loads the appropriate transformation pipeline based on the model type.

### **3. Preprocess Input Image:**
**Function: `preprocess_image(image_path, transform)`:**
  - Reads the input image using OpenCV and converts it to the RGB color space (MiDaS requires RGB input).
  - Applies the MiDaS transformation pipeline to the image, preparing it for depth estimation.

### **4. Perform Depth Estimation**
**Function: `estimate_depth(model, input_batch, device)`:**
  - Moves the preprocessed image batch to the appropriate device (GPU/CPU).
  - Passes the batch through the model in a no-gradient computation mode (`torch.no_grad()`), producing a depth map as the output.
  - Converts the depth map to a NumPy array for further processing.

### **5. Resize Depth Map**
**Function: `resize_depth_map(depth_map, original_image)`:** Ensures that the estimated depth map matches the dimensions of the original input image by resizing it with linear interpolation.

### **6. Visualize and Save Heatmap**
**Function: `save_depth_map(depth_map, output_path)`:**
  - Normalizes the depth map values to the range [0, 255] for visualization.
  - Applies a **color map** (COLORMAP_MAGMA) to the depth map to create a visually interpretable heatmap.
  - Saves the resulting heatmap as an image to the specified file path.

### **7. Main Execution**
The **main block** integrates all the steps:
  1. Reads the input image from `image_path`.
  2. Loads the MiDaS model and transformation pipeline.
  3. Preprocesses the input image.
  4. Estimates the depth map using the MiDaS model.
  5. Resizes the depth map to match the original image dimensions.
  6. Saves the depth map heatmap to the specified `output_path`.
