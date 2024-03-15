import cv2  # Or import PIL from PIL import Image
image1 = cv2.imread('image1.raw', cv2.IMREAD_UNCHANGED)
image2 = cv2.imread('image2.raw', cv2.IMREAD_UNCHANGED)
if image1.shape != image2.shape:
    print("Error: Images have different dimensions. Pixel-wise comparison cannot be performed.")
    exit()



diff = cv2.absdiff(image1, image2)


import cv2

image1 = cv2.imread('image1.raw', cv2.IMREAD_UNCHANGED)
image2 = cv2.imread('image2.raw', cv2.IMREAD_UNCHANGED)

if image1.shape != image2.shape:
    print("Error: Images have different dimensions.")
    exit()

# Separate channels (assuming they are stacked together):
b1, g1, r1 = cv2.split(image1)
b2, g2, r2 = cv2.split(image2)

diff_b = cv2.absdiff(b1, b2)
diff_g = cv2.absdiff(g1, g2)
diff_r = cv2.absdiff(r1, r2)



hist_b1 = cv2.calcHist([b1], [0], None, [256], [0, 256])
hist_g1 = cv2.calcHist([g1], [0], None, [256], [0, 256])
hist_r1 = cv2.calcHist([r1], [0], None, [256], [0, 256])

hist_b2 = cv2.calcHist([b2], [0], None, [256], [0, 256])
hist_g2 = cv2.calcHist([g2], [0], None, [256], [0, 256])
hist_r2 = cv2.calcHist([r2], [0], None, [256], [0, 256])

chi_square_b = cv2.compareHist(hist_b1, hist_b2, cv2.HISTCMP_CHISQR)
chi_square_g = cv2.compareHist(hist_g1, hist_g2, cv2.HISTCMP_CHISQR)
chi_square_r = cv2.compareHist(hist_r1, hist_r2, cv2.HISTCMP_CHISQR)

corr_coeff_b = cv2.compareHist(hist_b1, hist_b2, cv2.HISTCMP_CORREL)
corr_coeff_g = cv2.compareHist(hist_g1, hist_g2, cv2.HISTCMP_CORREL)
corr_coeff_r = cv2.compareHist(hist_r1, hist_r2, cv2.HISTCMP_CORREL)

print("Histogram comparisons:")
print("Chi-Square Statistic (Blue





import cv2
import matplotlib.pyplot as plt

# ... (pixel-wise absolute difference calculations)

plt.figure(figsize=(10, 5))

# Create a grayscale colormap for representing absolute difference
cmap = plt.cm.get_cmap('gray')

# Normalize absolute difference values (optional)
diff_normalized = diff_b / 255  # Normalize each channel's difference (0-1 range)
diff_all_channels = cv2.merge((diff_normalized, diff_normalized, diff_normalized))

plt.imshow(diff_all_channels, cmap=cmap)
plt.colorbar(label='Absolute Difference (Normalized)')
plt.title('Absolute Difference Visualization (Normalized)')
plt.show()





import cv2
import matplotlib.pyplot as plt

# ... (pixel-wise absolute difference calculations)

plt.figure(figsize=(10, 5))
plt.imshow(diff_b, cmap='hot')  # Use 'hot' or other colormaps for heatmap visualization
plt.colorbar(label='Absolute Difference (Blue Channel)')
plt.title('Absolute Difference Heatmap (Blue Channel)')
plt.show()

# Repeat for other channels (green, red)




# ... (histogram calculations)

# Plot histograms (using libraries like matplotlib or Seaborn)
plt.figure(figsize=(12, 6))

plt.subplot(221)  # Top-left subplot
plt.plot(hist_b1, label='Image 1 (Blue)')
plt.plot(hist_b2, label='Image 2 (Blue)')
plt.title('Blue Channel Histogram')
plt.legend()

plt.subplot(222)  # Top-right subplot
plt.plot(hist_g1, label='Image 1 (Green)')
plt.plot(hist_g2, label='Image 2 (Green)')
plt.title('Green Channel Histogram')
plt.legend()

plt.subplot(223)  # Bottom-left subplot
plt.plot(hist_r1, label='Image 1 (Red)')
plt.plot(hist_r2, label='Image 2 (Red)')
plt.title('Red Channel Histogram')
plt.legend()

plt.subplot(224)  # Bottom-right subplot
# Display chi-square and correlation coefficient values
plt.text(0.2, 0.5, f"Chi-Square (Blue): {chi_square_b:.2f}", fontsize=10)
plt.text(0.2, 0.35, f"Correlation (Blue): {corr_coeff_b:.2f}", fontsize=10)
plt.text(0.6, 0.5, f"Chi-Square (Green): {chi_square_g:.2f}", fontsize=10)
plt.text(0.6, 0.35, f"Correlation (Green): {corr_coeff_g:.2f}", fontsize=10)
plt.text(0.2, 0.2, f"Chi-Square (Red): {chi_square_r:.2f}", fontsize=
