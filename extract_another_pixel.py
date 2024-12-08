import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

image_path = "image_data_scraping3.png"
image = cv2.imread(image_path)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_purple = np.array([130, 50, 50])
upper_purple = np.array([160, 255, 255])
mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

coordinates = []
bottom_threshold = 800
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if cy < bottom_threshold:
            coordinates.append((cx, cy))

coordinates = sorted(coordinates, key=lambda point: (point[1], point[0]))

rows = {}
tolerance = 4
for x, y in coordinates:
    found_row = False
    for row_y in rows.keys():
        if abs(y - row_y) <= tolerance:
            rows[row_y].append((x, y))
            found_row = True
            break
    if not found_row:
        rows[y] = [(x, y)]

sorted_rows = sorted(rows.items(), key=lambda item: item[0])

while len(sorted_rows) < 10:
    sorted_rows.append(sorted_rows[-1])

final_rows = []
for idx, (y, points) in enumerate(sorted_rows):
    if idx == 0:
        points = points[:6]
    else:
        if len(points) > 10:
            points = points[:10]
        elif len(points) < 10:
            points = points * (10 // len(points)) + points[:10 % len(points)]
    final_rows.append((y, points))

frame_times = list(range(60, 541, 60))
data = {"t": [], "x_coordinates": [], "y_pixel": [], "x_scaled": [], "y_scaled": []}

xr_values = [317, 143, 88, 65, 50, 41, 35, 29, 27]
y_real = [i * 5 for i in range(1, 10)]

for t, (row_y, points), xr, y_scale in zip(frame_times, reversed(final_rows), xr_values, y_real):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_scaled = []
    for k in range(1, len(x_coords)):
        dx = (2.5 / xr) * (x_coords[k] - x_coords[k - 1])
        x_scaled.append(dx)
    x_scaled.insert(0, 0)

    y_scaled = [y_scale] * len(x_coords)

    data["t"].append(t)
    data["x_coordinates"].append(", ".join(map(str, x_coords)))
    data["y_pixel"].append(row_y)
    data["x_scaled"].append(", ".join(map(str, x_scaled)))
    data["y_scaled"].append(", ".join(map(str, y_scaled)))

df = pd.DataFrame(data)

output_csv_path = "final_scaled_track_coordinates_t_540_with_fixed_rows.csv"
df.to_csv(output_csv_path, index=False)
print(f"Formatted data saved to {output_csv_path}")

for coord in coordinates:
    cv2.circle(image, coord, 5, (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Points (Track Only)")
plt.axis("off")
plt.show()
