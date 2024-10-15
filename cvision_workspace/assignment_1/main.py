import cv2
import numpy as np
import easyocr
import time


# Decode the predictions
def decode_predictions(scores, geometry, conf_threshold=0.5):
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < conf_threshold:
                continue

            (offset_x, offset_y) = (x * 4.0, y * 4.0)
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = max(int(end_x - w), 0)
            start_y = max(int(end_y - h), 0)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rects, confidences


reader = easyocr.Reader(['ch_sim', 'en'])
# Load the pre-trained EAST text detector
net = cv2.dnn.readNet('./frozen_east_text_detection.pb')
# Load the image
st = time.time()
image = cv2.imread('sample3.png')
orig = image.copy()
origH, origW = image.shape[:2]

# calculate ratios that will be used to scale bounding box coordinates
newW, newH = 640, 640
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply blurring
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# Apply thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.int8), iterations=1)
cv2.imshow('thresh', thresh)

# # Edge detection
# edges = cv2.Canny(blurred, 20, 150, L2gradient=False)
# # edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.int8), iterations=1)
# cv2.imshow('edges', edges)
#
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # Create an empty image to draw filled contours
# filled_image = np.zeros_like(edges)
#
# # Iterate over contours and their hierarchy
# for i, contour in enumerate(contours):
#     color = int(filled_image[contour[0][0][1], contour[0][0][0]])
#     cv2.drawContours(filled_image, [contour], 0, (255 - color,), thickness=cv2.FILLED)
# for i, contour in enumerate(contours):
#     if hierarchy[0][i][3] != -1:
#         color = int(filled_image[contour[0][0][1], contour[0][0][0]])
#         if hierarchy[0][hierarchy[0][i][3]][3] != -1 and hierarchy[0][hierarchy[0][hierarchy[0][i][3]][3]][3] != -1:
#             color = 255 - color
#         cv2.drawContours(filled_image, [contour], 0, (255 - color,), thickness=cv2.FILLED)
#
# filled_image = cv2.GaussianBlur(filled_image, (5, 5), 0)
# cv2.imshow("filled_image", filled_image)

input_ = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Prepare the image for the network
blob = cv2.dnn.blobFromImage(input_, 1.0, (W, H), (123.68, 116.78, 103.94), False, False)
net.setInput(blob)

# Get the output layer names
layer_names = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# Forward pass
scores, geometry = net.forward(layer_names)

(rects, confidences) = decode_predictions(scores, geometry)

indices = cv2.dnn.NMSBoxes(rects, confidences, 0.3, 0.5)

# merge bounding boxes that are close to each other
boxes = []
for i in indices:
    (start_x, start_y, end_x, end_y) = rects[i]
    boxes.append([start_x, start_y, end_x, end_y])


boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
boxes = np.array(boxes)
boxes = boxes[boxes[:, 1].argsort()]

for i in range(1, len(boxes)):
    if boxes[i][1] - boxes[i-1][1] < 30:
        boxes[i][1] = min(boxes[i][1], boxes[i-1][1])
        boxes[i][3] = max(boxes[i][3], boxes[i-1][3])
        boxes[i][0] = min(boxes[i][0], boxes[i-1][0])
        boxes[i][2] = max(boxes[i][2], boxes[i-1][2])

print(boxes)
sorted_rects = [boxes[0]]
for i in range(1, len(boxes)):
    if boxes[i][1] == boxes[i-1][1] and boxes[i][3] == boxes[i-1][3]:
        boxes[i-1][0] = min(boxes[i][0], boxes[i-1][0])
    elif boxes[i][1] == boxes[i-1][1] and boxes[i][0] == boxes[i-1][0]:
        boxes[i-1][2] = max(boxes[i][2], boxes[i-1][2])
        boxes[i-1][3] = max(boxes[i][3], boxes[i-1][3])
    elif boxes[i][1] == boxes[i-1][1] and boxes[i][2] == boxes[i-1][2]:
        boxes[i-1][0] = min(boxes[i][0], boxes[i-1][0])

    else:
        sorted_rects.append(boxes[i])

# Draw bounding boxes
# for (start_x, start_y, end_x, end_y) in boxes:
for i in sorted_rects:
    (start_x, start_y, end_x, end_y) = i
    start_x = int(start_x * rW)
    start_y = int(start_y * rH)
    end_x = int(end_x * rW)
    end_y = int(end_y * rH)
    roi = orig[start_y:end_y, start_x:end_x]
    resp = reader.readtext(roi)
    if not resp:
        continue
    (_, text, confidence) = resp[0]
    print("OCR TEXT: ", text, "Confidence: ", confidence)
    cv2.rectangle(orig, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    cv2.putText(orig, f"{text}[{confidence:.2f}]", (start_x, start_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

ed = time.time()
print("Time taken: ", ed - st)

# Display the annotated image
cv2.imshow('Annotated Image', orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
