from utils import get_model
import tensorflow as tf

MODEL_KEY = 'resnet50_gru'  # 'FER' / 'FER_LSTM' / 'resnet50' / 'resnet50_gru'
PRETRAINED = True
WINDOW_SIZE = 10

# Model load to global variable
MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED, window_size=WINDOW_SIZE)


print(MODEL.summary())

input_ = tf.ones((1, 10, 112, 112, 3))

output_ = MODEL.predict(input_)

print(output_)