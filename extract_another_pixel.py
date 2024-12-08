import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the image
image_path = "image_data_scraping3.png"  # Replace with your image file path
image = cv2.imread(image_path)

# Convert to HSV and mask for purple
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_purple = np.array([130, 50, 50])
upper_purple = np.array([160, 255, 255])
mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

# Detect contours of purple dots
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract coordinates
coordinates = []
bottom_threshold = 800  # Exclude points below this y-coordinate (adjust based on the image)
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if cy < bottom_threshold:  # Only include points above the threshold
            coordinates.append((cx, cy))

# Sort the coordinates
coordinates = sorted(coordinates, key=lambda point: (point[1], point[0]))

# Data for scaling (Table 1)
frame_times = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540]
xr_values = [317, 143, 88, 65, 50, 41, 35, 29, 27, 25]  # Scaling factors (px/m)
y_real = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # Real-world vertical distances (m)

# Organize raw pixel data for each frame
raw_pixel_data = {"t": [], "xpx": [], "ypx": []}

# Group coordinates by frame
coordinates_per_frame = len(coordinates) // len(frame_times)
index = 0
for t in frame_times:
    for i in range(coordinates_per_frame):
        if index < len(coordinates):
            cx, cy = coordinates[index]
            raw_pixel_data["t"].append(t)
            raw_pixel_data["xpx"].append(cx)
            raw_pixel_data["ypx"].append(cy)
            index += 1

# Save raw pixel data to a DataFrame
raw_pixel_df = pd.DataFrame(raw_pixel_data)

# Save raw pixel coordinates to CSV
raw_pixel_output_path = "raw_pixel_coordinates.csv"
raw_pixel_df.to_csv(raw_pixel_output_path, index=False)
print(f"Raw pixel data saved to {raw_pixel_output_path}")

# Visualize the points (excluding car)
for coord in coordinates:
    cv2.circle(image, coord, 5, (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Points (Track Only)")
plt.axis("off")
plt.show()
