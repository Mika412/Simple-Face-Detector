
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

import keras2onnx
import onnxruntime
import onnx

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

temp_model_file = 'model.onnx'
onnx.save_model(onnx_model, temp_model_file)