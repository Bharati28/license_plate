import cv2
import imutils
import pytesseract
import numpy as np

# Specify the path to the Tesseract executable (Mac path)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# FUNCTION 1: Preprocess the image
def preprocess_image(image):
    image = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(smooth, 170, 200)
    return image, edges

# FUNCTION 2: Find the license plate contour
def find_plate_contour(edges):
    contours, _ = cv2.findContours(edges.copy(),
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            # License plates are wider than tall (ratio 2 to 6)
            if 2 <= aspect_ratio <= 6:
                return approx, x, y, w, h

    return None, None, None, None, None

# FUNCTION 3: Preprocess cropped plate for OCR
def preprocess_plate(crp_image):
    crp_image = cv2.resize(crp_image, (300, 100))
    gray = cv2.cvtColor(crp_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# FUNCTION 4: Extract text using OCR
def extract_text(plate_image):
    whitelist = '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    for psm in ['--psm 8', '--psm 7', '--psm 6']:
        text = pytesseract.image_to_string(
            plate_image, config=f'{psm} {whitelist}')
        text = text.strip()
        if text:
            return text
    return "Could not read plate"

# MAIN EXECUTION
image = cv2.imread('car3.jpg')
if image is None:
    print("Error: Unable to load image. Check the file path.")
    exit()

# Step 1: Preprocess
image, edges = preprocess_image(image)

# Step 2: Find plate contour
plate_contour, x, y, w, h = find_plate_contour(edges)

if plate_contour is not None:
    # Step 3: Crop the plate
    crp_image = image[y:y + h, x:x + w]
    cv2.imwrite('NumberPlate.png', crp_image)

    # Step 4: Preprocess plate for OCR
    plate_processed = preprocess_plate(crp_image)

    # Step 5: Draw detected plate on original image
    cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)

    # Step 6: Show the 3 most important windows
    cv2.imshow("Final Image - Plate Detected", image)
    cv2.imshow("Cropped Plate", crp_image)
    cv2.imshow("Processed Plate for OCR", plate_processed)
    cv2.waitKey(0)

    # Step 7: Extract and print text
    text = extract_text(plate_processed)
    print(f"Detected Number Plate: {text}")

else:
    print("Number plate not detected.")
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()