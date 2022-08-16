from PIL import Image

import numpy as np
import cv2
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from keras.models import load_model

width = 640
height = 480
cameraNo = 0

cap = cv2.VideoCapture(cameraNo)
cap.set(3, width)
cap.set(4, height)

model = load_model('finalTrained.h5')

while True:
    success, im_orig = cap.read()
    image = Image.fromarray(im_orig, 'RGB')
    image = image.resize((64, 64))
    img_array = np.array(image)

    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)  # [0][0]
    print(prediction)
    if prediction==0:
        print("CAT")
    elif prediction==1:
        print('DOG')
    else:
        print('No human')
    cv2.putText(im_orig, 'Predicted Digit : ' + str(''),
                (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 255), 1)
    cv2.imshow("Original Image", im_orig)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
