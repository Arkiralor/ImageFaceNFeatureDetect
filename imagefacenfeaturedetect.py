# NOTE: The included Cascade classifier apparently does NOT work well with dar-skinned individuals wearing white clothes
# on a white background.

# Classifier set provided by OpenCV @ https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2

img = cv2.imread("Image Samples/Sample_012.jpg", 1)

print(img.shape)

cv2.imshow("Original Images",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier("Classifiers/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("Classifiers/haarcascade_eye.xml")
glasses_cascade = cv2.CascadeClassifier("Classifiers/haarcascade_eye_tree_eyeglasses.xml")
profile_cascade = cv2.CascadeClassifier("Classifiers/haarcascade_profileface.xml")

gry_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gry_img, scaleFactor = 1.05, minNeighbors = 10)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y),(x + w, y + h), (255, 0, 0), 1) # Blue
    gry_crp = gry_img[y:y+h, x:x+w]
    crp_img = img[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(gry_crp, scaleFactor = 1.05, minNeighbors = 5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(crp_img, (x, y), (x + w, y + h), (255, 255, 24), 1) # Cyan

    glasses = glasses_cascade.detectMultiScale(gry_crp, scaleFactor=1.35, minNeighbors=5)
    for (x, y, w, h) in glasses:
        cv2.rectangle(crp_img, (x, y), (x + w, y + h), (31, 255, 31), 1) # Green.

profile_faces = profile_cascade.detectMultiScale(gry_img, scaleFactor = 1.25, minNeighbors = 5)
for x,y,w,h in profile_faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (17, 255, 255), 1) # Yellow.
    gry_crp = gry_img[y:y + h, x:x + w]
    crp_img = img[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(gry_crp, scaleFactor=1.05, minNeighbors=5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(crp_img, (x, y), (x + w, y + h), (255, 255, 24), 1)  # Cyan

    glasses = glasses_cascade.detectMultiScale(gry_crp, scaleFactor=1.35, minNeighbors=5)
    for (x, y, w, h) in glasses:
        cv2.rectangle(crp_img, (x, y), (x + w, y + h), (31, 255, 31), 1)  # Green.

cv2.imshow("Detected Faces",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
