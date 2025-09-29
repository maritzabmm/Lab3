import cv2 
import numpy as np 
from datetime import datetime

# Read image 
image = cv2.imread('./images/drone_image.jpg') 

# Define coordinates, color and thickness for rectangle
x1, y1, x2, y2 = 100, 100, 200, 200
rectangle_B, rectangle_G, rectangle_R = 255, 0, 0
rectangle_color = (120, 39, 89) # (B, G, R)
rectangle_thickness = 2

# Define center and radius for circle
center_x, center_y, radius = 1200, 400, 50
circle_color = (14, 166, 250) # (B, G, R)
circle_thickness = 3

# Define vertices and color for polygon (triangle)
triangle_pts = np.array( [[400, 400], [500, 400], [450, 300]], np.int32)
triangle_pts = triangle_pts.reshape((-1, 1, 2))
triangle_color = (224, 157, 128) # (B, G, R)

# Draw shapes 
cv2.rectangle(image, (x1, y1), (x2, y2), rectangle_color, rectangle_thickness) 
cv2.circle(image, (center_x, center_y), radius, circle_color, circle_thickness)
cv2.fillPoly(image, [triangle_pts], color=triangle_color)

# Write names
font = cv2.FONT_HERSHEY_SIMPLEX
names_font_scale = 3
names_color = (255, 255, 255) # (B, G, R)
names_thickness = 2
cv2.putText(image,'Adolfo + Maritza',(1020,80), font, names_font_scale,names_color,names_thickness,cv2.LINE_AA)

# Get current timestamp
current_timestamp = datetime.now()

# Write timestamp
timestamp_font=cv2.FONT_HERSHEY_DUPLEX
timestamp_font_scale=1
timestamp_color = (0, 240, 74)
timestamp_thickness = 1
cv2.putText(image, str(current_timestamp), (10, 1050), timestamp_font, timestamp_font_scale, timestamp_color, timestamp_thickness)

processed_img = cv2.imwrite('./images/task1_processed.jpg', image)

print("Image processed and saved as 'task1_processed.jpg'")