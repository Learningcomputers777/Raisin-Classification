import cv2
import numpy as np


def process_image(image_path):
    # Load the color image
    image = cv2.imread(image_path)

    # Define the color range for the black raisin (adjust this based on your image)
    lower_black = np.array([0, 0, 0])  # Lower bound for black color
    upper_black = np.array([90, 190, 240])  # Upper bound for black color

    # Create a binary mask for the raisin
    mask = cv2.inRange(image, lower_black, upper_black)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours, e.g., choose the largest contour
    raisin_contour = max(contours, key=cv2.contourArea)

    # Draw the raisin contour on the original image
    cv2.drawContours(image, [raisin_contour], -1, (0, 255, 0), 2)

    # Extract features from the contour
    area = cv2.contourArea(raisin_contour)

    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(raisin_contour)
    major_axis_length, minor_axis_length = ellipse[1]

    # Calculate eccentricity with a check to avoid negative values inside the sqrt
    if major_axis_length > minor_axis_length:
        eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
    else:
        eccentricity = 0

    # Calculate convex hull and its area
    convex_hull = cv2.convexHull(raisin_contour)
    convex_area = cv2.contourArea(convex_hull)

    # Additional features
    extent = area / convex_area
    perimeter = cv2.arcLength(raisin_contour, True)

    # Return the extracted features
    return area, major_axis_length, minor_axis_length, eccentricity, convex_area, extent, perimeter, image

# # Example usage
# image_path = 'WhatsApp Image 2023-11-14 at 21.50.06.jpeg'
# area, major_axis_length, minor_axis_length, eccentricity, convex_area, extent, perimeter, processed_image = process_image(image_path)

# # Print the extracted features
# print("Area:", area)
# print("Major Axis Length:", major_axis_length)
# print("Minor Axis Length:", minor_axis_length)
# print("Eccentricity:", eccentricity)
# print("Convex Area:", convex_area)
# print("Extent:", extent)
# print("Perimeter:", perimeter)

# # Display the processed image with the highlighted raisin
# cv2.imshow('raisin',processed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
