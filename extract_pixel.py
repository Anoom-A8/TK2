import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

image_path = "image_data_scraping.png"  
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

min_contour_area = 25  

thresh = cv2.adaptiveThreshold(
    gray_image, 
    255, 
    cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY_INV, 
    15, 3
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  

filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

contour_image = np.zeros_like(gray_image)
cv2.drawContours(contour_image, filtered_contours, -1, 255, 1)
plt.imshow(contour_image, cmap='gray')
plt.title("Filtered Contours")
plt.axis("off")
plt.show()

pixel_data = []
for contour in filtered_contours:
    for point in contour:
        pixel_data.append(tuple(point[0]))  

unique_pixel_data = sorted(set(pixel_data), key=lambda p: (p[1], p[0]))

x_px, y_px = zip(*unique_pixel_data)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image with Raw Pixels Overlaid")
plt.axis("off")

plt.scatter(x_px, y_px, c='red', s=2, label="Raw Pixels")  
plt.legend(loc='upper right')
plt.show()

x_diffs = np.diff(x_px)

x_r = [317, 143, 88, 65, 50, 41, 35, 29, 27, 25]  
y_real = np.arange(0, 50, 5)  

real_coords = []
max_iterations = 10

for i in range(1, min(len(x_px), max_iterations)):
    y_index = min(len(x_r) - 1, i)  
    dx = (2.5 / x_r[y_index]) * (x_px[i] - x_px[i - 1])
    last_x = real_coords[-1][0] if real_coords else 0
    real_coords.append((last_x + dx, y_real[y_index]))

real_coords.insert(0, (0, 0))

time_frames = np.arange(0, len(real_coords) * 60, 60)
table_data = {'t': time_frames[:len(real_coords)],
              'xt': [coord[0] for coord in real_coords],
              'yt': [coord[1] for coord in real_coords]}

df = pd.DataFrame(table_data)

def smooth_data(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_xt = smooth_data(df['xt'], window_size=5)

df['smoothed_xt'] = np.nan
df.loc[4:, 'smoothed_xt'] = smoothed_xt  

plt.figure(figsize=(10, 6))
plt.plot(df['xt'], df['yt'], 'r-', label="Raw Track Path")  
plt.plot(df['smoothed_xt'].dropna(), df['yt'][4:], 'b-', label="Smoothed Track Path")  
plt.title("Track Coordinates: Raw vs Smoothed")
plt.xlabel("Horizontal Distance (meters)")
plt.ylabel("Vertical Distance (meters)")
plt.legend()
plt.show()

df.to_csv("processed_data_with_smoothed.csv", index=False)
