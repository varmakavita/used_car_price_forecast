import cv2

# import the necessary packages
from config import car_config as config
import numpy as np
import mxnet as mx
import os
import pickle

imagePath = "./cars_dataset/11822.jpg"

# sym = './checkpoints/vggnet-symbol.json'
# params = './checkpoints/vggnet-0065.params'

# # Standard Imagenet input - 3 channels, 224 * 224
# in_shapes = [(1, 3, 224, 224)]
# in_types = [np.float32]

# # Path of the output file
# onnx_file = './mxnet_exported_resnet18.onnx'

# converted_model_path = mx.contrib.onnx.export_model(sym, params, in_shapes, in_types, onnx_file)


le = pickle.loads(open(config.LABEL_ENCODER_PATH, "rb").read())
rows = open(config.TEST_MX_LIST).read().strip().split("\n")
rows = np.random.choice(rows, size=10)

print("[INFO] loading pre-trained model...")
checkpointsPath = os.path.sep.join([config.CHECKPOINT_PATH, config.CHECKPOINT_PREFIX])
model = mx.model.FeedForward.load(checkpointsPath, 56)

model = mx.model.FeedForward(ctx=[mx.gpu(0)], symbol=model.symbol, arg_params=model.arg_params, aux_params=model.aux_params)

image = cv2.imread(imagePath)

orig = image.copy()

(B, G, R) = cv2.split(image.astype("float32"))

# subtract the means for each channel
R -= config.R_MEAN
G -= config.G_MEAN
B -= config.B_MEAN

# merge the channels back together and return the image
image = cv2.merge([B, G, R])

image = np.moveaxis(image, 2, 0)
image = np.expand_dims(image, axis=0)

preds = model.predict(image)[0]
idxs = np.argsort(preds)[::-1][:5]
print([idxs[0]])
label = le.inverse_transform([idxs[0]])
print(label)
label = label[0].replace(":", " ")
label = "{}: {:.2f}%".format(label, preds[idxs[0]] * 100)
cv2.putText(orig, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

for (i, prob) in zip(idxs, preds):
    print("\t[INFO] predicted={}, probability={:.2f}%".format(le.inverse_transform([i]), preds[i] * 100))

# show the image
cv2.imshow("Image", orig)
cv2.waitKey(0)