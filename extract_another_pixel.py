import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the image
image_path = "image_data_scraping2.png"  # Replace with your image file path
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

# Calculate real-world coordinates (xt, yt)
data = {"t": [], "xt": [], "yt": []}
index = 0
for t, xr, y in zip(frame_times, xr_values, y_real):
    for i in range(len(coordinates) // len(frame_times)):  # Divide evenly among frames
        if index + 1 < len(coordinates):
            xpx_curr, ypx_curr = coordinates[index]
            xpx_next, ypx_next = coordinates[index + 1]

            # Calculate dx using the formula provided
            dx = (2.5 / xr) * abs(xpx_next - xpx_curr)

            # Append to the data table
            data["t"].append(t)
            data["xt"].append(dx)  # Horizontal distance
            data["yt"].append(y)   # Vertical distance

            index += 1

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_csv_path = "scaled_track_coordinates_no_car.csv"
df.to_csv(output_csv_path, index=False)
print(f"Data saved to {output_csv_path}")

# Visualize the points (excluding car)
for coord in coordinates:
    cv2.circle(image, coord, 5, (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Points (Track Only)")
plt.axis("off")
plt.show()
