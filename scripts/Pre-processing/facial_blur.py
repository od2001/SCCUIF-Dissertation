import dlib
import cv2

# Load the pre-trained face detection model
detector = dlib.get_frontal_face_detector()

img_path = ""
image = cv2.imread(img_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Loop over the detected faces
for face in faces:
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()

    # Extract the face region from the image
    face_region = image[y:y1, x:x1]

    # Apply Gaussian blur to this face region
    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)

    # Replace the original image's face region with the blurred face
    image[y:y1, x:x1] = blurred_face



# Display the result
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 600, 400)
cv2.imshow("Result", image)
cv2.imwrite("Blured.jpeg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
