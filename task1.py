import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def load_images(left_path, right_path):
    left_img = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)
    return left_img, right_img

def compute_disparity(left_img, right_img, window_size=5, min_disp=0, num_disp=16):
    """
    Compute disparity map using block matching algorithm.
    window_size: Size of the matching window
    min_disp: Minimum disparity value
    num_disp: Number of disparity levels
    """
    height, width = left_img.shape
    disparity = np.zeros((height, width), dtype=np.float32)
    
    # Pad images to handle window size
    pad_size = window_size // 2
    left_padded = cv2.copyMakeBorder(left_img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    right_padded = cv2.copyMakeBorder(right_img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    
    # For each pixel in the left image
    for y in range(pad_size, height + pad_size):
        for x in range(pad_size, width + pad_size):
            # Extract window from left image
            left_window = left_padded[y-pad_size:y+pad_size+1, x-pad_size:x+pad_size+1]
            
            # Search in right image
            min_ssd = float('inf')
            best_disp = 0
            
            # Search range
            for d in range(min_disp, min_disp + num_disp):
                if x - d - pad_size < 0:
                    continue
                    
                right_window = right_padded[y-pad_size:y+pad_size+1, x-d-pad_size:x-d+pad_size+1]
                
                # Compute SSD (Sum of Squared Differences)
                ssd = np.sum((left_window - right_window) ** 2)
                
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disp = d
            
            disparity[y-pad_size, x-pad_size] = best_disp
    
    return disparity

def disparity_to_depth(disparity, baseline, focal_length):
    """Convert disparity map to depth map."""
    # Avoid division by zero
    depth = np.where(disparity > 0, baseline * focal_length / disparity, 0)
    return depth

def create_depth_heatmap(depth):
    """Create a color-coded depth map."""
    # Normalize depth values to 0-255 range
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return heatmap

def main():
    # Parameters
    baseline = 0.1  # Distance between cameras in meters
    focal_length = 500  # Focal length in pixels
    
    # Load images
    left_img, right_img = load_images('left.png', 'right.png')
    
    # Compute disparity map
    print("Computing disparity map...")
    disparity = compute_disparity(left_img, right_img)
    
    # Convert disparity to depth
    print("Converting disparity to depth...")
    depth = disparity_to_depth(disparity, baseline, focal_length)
    
    # Create color-coded depth map
    print("Creating depth heatmap...")
    depth_heatmap = create_depth_heatmap(depth)
    
    # Save results
    cv2.imwrite('depth.png', depth_heatmap)
    print("Depth map saved as 'depth.png'")
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(left_img, cmap='gray')
    plt.title('Left Image')
    
    plt.subplot(132)
    plt.imshow(right_img, cmap='gray')
    plt.title('Right Image')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Depth Map')
    
    plt.tight_layout()
    plt.savefig('stereo_results.png')
    plt.close()

if __name__ == "__main__":
    main() 