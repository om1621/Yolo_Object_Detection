import requests
import json
import io
import sys
import numpy as np
import cv2

im = cv2.imread(sys.argv[1])
im = np.array(im)
meta = io.StringIO(json.dumps({'shape': list(im.shape)}))
data = io.BytesIO(bytearray(im))
r = requests.post('http://0.0.0.0:8000/predict',
                  files={'meta': meta, 'img' : data})
response = json.loads(r.content)
img = np.uint8(np.array(response))
print("Output Image")
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows
