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
tolerance = 10
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

final_rows = []
for y, points in sorted_rows:
    if len(points) > 10:
        mid_index = len(points) // 2
        final_rows.append((y, points[:mid_index]))
        final_rows.append((y + tolerance, points[mid_index:]))
    else:
        final_rows.append((y, points))

frame_times = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540]
data = {"t": [], "x_coordinates": [], "y": []}

for t, (row_y, points) in zip(frame_times, reversed(final_rows)):
    data["t"].append(t)
    data["x_coordinates"].append(", ".join(map(str, [p[0] for p in points])))
    data["y"].append(row_y)

df = pd.DataFrame(data)

output_csv_path = "corrected_track_coordinates.csv"
df.to_csv(output_csv_path, index=False)
print(f"Formatted data saved to {output_csv_path}")

for coord in coordinates:
    cv2.circle(image, coord, 5, (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Points (Track Only)")
plt.axis("off")
plt.show()
