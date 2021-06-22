from utils import get_model, metric_CCC, read_pickle, Dataloader, read_csv, load_image, read_txt, CCC_score_np
import tensorflow as tf


MODEL = get_model(key='FER_LSTM', preTrained=True,
                  weight_path='/home/gsethan/Desktop/ABAW2021/results/620_17_33_FER_LSTM/best_weights',
                  input_size = (224, 224),
                  dropout_rate=0.2)

input_ = tf.ones((1, 10, 224, 224, 3))

output_ = MODEL(input_)
output_pred = MODEL.predict(input_)

print(output_)
print(output_pred)