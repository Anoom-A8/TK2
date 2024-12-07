import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the image
image_path = "image_data_scraping.png"  # Update this to the correct path
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocessing: Apply Gaussian Blur
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Adjust the min_contour_area to capture more contours
min_contour_area = 25  # Lower the threshold to capture smaller contours

# Thresholding: Adjust the block size and constant for better capture
thresh = cv2.adaptiveThreshold(
    gray_image, 
    255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY_INV, 
    15, 3  # Larger block size and slightly higher constant
)

# Morphological Operations: Experiment with different kernels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Larger kernel size for better closing
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours with different retrieval modes to capture more details
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Use RETR_LIST

# Filter contours by area to remove noise
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

# Display the filtered contours
contour_image = np.zeros_like(gray_image)
cv2.drawContours(contour_image, filtered_contours, -1, 255, 1)
plt.imshow(contour_image, cmap='gray')
plt.title("Filtered Contours")
plt.axis("off")
plt.show()

# Extract (x_px, y_px) coordinates from filtered contours
pixel_data = []
for contour in filtered_contours:
    for point in contour:
        pixel_data.append(tuple(point[0]))  # Extract x and y pixels

# Remove duplicates and sort by y_px, then x_px
unique_pixel_data = sorted(set(pixel_data), key=lambda p: (p[1], p[0]))

# Unpack coordinates
x_px, y_px = zip(*unique_pixel_data)

# Display the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image with Raw Pixels Overlaid")
plt.axis("off")

# Overlay the raw pixel data on the image
plt.scatter(x_px, y_px, c='red', s=2, label="Raw Pixels")  # Small red dots
plt.legend(loc='upper right')
plt.show()

# Compute differences between consecutive x_px
x_diffs = np.diff(x_px)

# Map y_px to y_real values using given scale
x_r = [317, 143, 88, 65, 50, 41, 35, 29, 27, 25]  # Provided x_r values
y_real = np.arange(0, 50, 5)  # y (meters) in the problem statement

# Compute real-world coordinates using scaling formula
real_coords = []
max_iterations = 10  # Process up to t = 540 seconds (10 time points)

# Ensure we are handling the data properly, including index bounds
for i in range(1, min(len(x_px), max_iterations)):
    y_index = min(len(x_r) - 1, i)  # Ensure index is within bounds of x_r
    # Properly calculate dx considering the scale factors
    dx = (2.5 / x_r[y_index]) * (x_px[i] - x_px[i - 1])
    last_x = real_coords[-1][0] if real_coords else 0
    real_coords.append((last_x + dx, y_real[y_index]))

# Add the initial point at (0, 0)
real_coords.insert(0, (0, 0))

# Prepare data for the table
time_frames = np.arange(0, len(real_coords) * 60, 60)
table_data = {'t': time_frames[:len(real_coords)],
              'xt': [coord[0] for coord in real_coords],
              'yt': [coord[1] for coord in real_coords]}

# Create a pandas DataFrame
df = pd.DataFrame(table_data)

# Function for smoothing the data
def smooth_data(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Smooth the xt values
smoothed_xt = smooth_data(df['xt'], window_size=5)

# Update the DataFrame with the smoothed values
df['smoothed_xt'] = np.nan  # Initialize new column for smoothed xt
df.loc[4:, 'smoothed_xt'] = smoothed_xt  # Skip the first few points due to smoothing

# Visualize both raw and smoothed track paths
plt.figure(figsize=(10, 6))
plt.plot(df['xt'], df['yt'], 'r-', label="Raw Track Path")  # Raw path (red)
plt.plot(df['smoothed_xt'].dropna(), df['yt'][4:], 'b-', label="Smoothed Track Path")  # Smoothed path (blue)
plt.title("Track Coordinates: Raw vs Smoothed")
plt.xlabel("Horizontal Distance (meters)")
plt.ylabel("Vertical Distance (meters)")
plt.legend()
plt.show()

# Save the processed data to a CSV file
df.to_csv("processed_data_with_smoothed.csv", index=False)
