# Import the required libraries
import numpy as np
import torch
import cv2

# Load MiDaS model
def load_midas_model():
    model_type = "DPT_Large" 
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load transformation pipeline
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform if "DPT" in model_type else torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    return model, transform, device

# Preprocess input image
def preprocess_image(image_path, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the transform 
    input_batch = transform(image) 
    return input_batch, image

# Perform depth estimation
def estimate_depth(model, input_batch, device):
    input_batch = input_batch.to(device)
    with torch.no_grad():
        depth = model(input_batch)
    depth = depth.squeeze().cpu().numpy()
    return depth

# Visualize and save the heatmap
def save_depth_map(depth_map, output_path):
    # Normalize depth map to range [0, 255] for visualization
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # Save the heatmap to the specified path
    cv2.imwrite(output_path, depth_colormap)
    print(f"Depth map saved to: {output_path}")

# Resize depth map to match original image dimensions
def resize_depth_map(depth_map, original_image):
    height, width = original_image.shape[:2]
    resized_depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_depth_map

# Modify the main function
if __name__ == "__main__":
    image_path = r"C:\Users\krish\OneDrive\Desktop\image.jpg"
    output_path = r"C:\Users\krish\OneDrive\Desktop\output.jpg"

    # Load model and transformation
    model, transform, device = load_midas_model()

    # Preprocess image
    input_batch, original_image = preprocess_image(image_path, transform)

    # Estimate depth
    depth_map = estimate_depth(model, input_batch, device)

    # Resize depth map to original image dimensions
    depth_map_resized = resize_depth_map(depth_map, original_image)

    # Save the depth map heatmap
    save_depth_map(depth_map_resized, output_path)
