import os
from glob import glob
import csv
import pickle
import numpy as np
import tensorflow as tf
import glob
from base_model.ResNet import ResNet34
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, ConvLSTM2D, BatchNormalization, Dropout
from tensorflow.keras.applications import ResNet50, VGG19
import cv2
from skimage.metrics import structural_similarity as ssim


def read_pickle(path) :
    with open(path, 'rb') as f:
        content = pickle.load(f)
    return content

def read_txt(path) :
    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip().split(',') for x in content]

    return content

def read_csv(path) :
    lines = []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            start = len(lines)
            lines[start:start] = line

    return lines

# Model Load Function
def get_model(key='FER', preTrained = True, weight_path=os.path.join(os.getcwd(), 'base_model', 'ResNeXt34_Parallel_add', 'checkpoint_4_300000-320739.ckpt'),
              weight_seq_path=os.path.join(os.getcwd(), 'results', '618_0_42_FER_LSTM', '1epoch_weights'),
              window_size = 10, input_size=(224,224),
              dropout_rate = 0.2) :
    if key == 'FER' :
        # Model load
        model = ResNet34(cardinality = 32, se = 'parallel_add')
        
        if preTrained :
            # load pre-trained weights
            # weight_path = os.path.join(os.getcwd(), 'base_model', 'ResNeXt34_Parallel_add', 'checkpoint_4_300000-320739.ckpt')
            # weight_path = os.path.join(os.getcwd(), 'results', '614_1315_FER', 'best_weights')
            assert len(glob.glob(weight_path + '*')) > 1, 'There is no weight file | {}'.format(weight_path)
            model.load_weights(weight_path)
            print("The model weights has been load")
            print(weight_path)

        return model

    elif key == 'FER_LSTM' :
        # Base model load
        base_model = ResNet34(cardinality=32, se='parallel_add')

        # load pre-trained weights of base model
        assert len(glob.glob(weight_path + '*')) > 1, 'There is no weight file | {}'.format(weight_path)
        base_model.load_weights(weight_path)
        print("The model weights has been load")
        print(weight_path)

        base_model.build(input_shape=(None, input_size[0], input_size[1], 3))
        #############################
        sub_model = tf.keras.Sequential()
        sub_model.add(tf.keras.Input(shape=(input_size[0], input_size[1], 3)))
        for i in range(6):
            sub_model.add(base_model.layers[i])


        input_ = tf.keras.Input(shape=(window_size, input_size[0], input_size[1], 3))
        for i in range(window_size) :
            out_ = sub_model(input_[:,i,:,:,:])

            if i == 0 :
                out_0 = tf.expand_dims(out_, axis = 1)
            elif i == 1 :
                out_1 = tf.expand_dims(out_, axis = 1)
                output_ = tf.concat([out_0, out_1], axis = 1)
            else :
                out_3 = tf.expand_dims(out_, axis = 1)
                output_ = tf.concat([output_, out_3], axis = 1)

        lstm = LSTM(256, input_shape=(window_size, 512), dropout=dropout_rate)(output_)
        do1 = Dropout(rate=dropout_rate)(lstm)
        fo1 = Dense(256, activation = 'tanh')(do1)
        fo2 = Dense(2, activation='tanh')(fo1)

        model = Model(inputs=input_, outputs=fo2)

        if preTrained:
            assert len(glob.glob(weight_seq_path + '*')) > 1, 'There is no weight file | {}'.format(weight_seq_path)
            model.load_weights(weight_seq_path)
            print("The model weights has been load")
            print(weight_seq_path)

        for layer in model.layers :
            layer.trainable = False
        model.layers[-1].trainable = True
        model.layers[-2].trainable = True

        return base_model, model

    elif key == 'FER_ConvLSTM' :
        # Base model load
        base_model = ResNet34(cardinality=32, se='parallel_add')

        # load pre-trained weights of base model
        assert len(glob.glob(weight_path + '*')) > 1, 'There is no weight file | {}'.format(weight_path)
        base_model.load_weights(weight_path)
        print("The model weights has been load")
        print(weight_path)
        
        model = Sequential()
        model.add(Input(shape=(10, 224, 224, 3)))
        '''
        model.add(ConvLSTM2D(filters=20, kernel_size=(3, 3),
                             padding='same', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                             padding='same', return_sequences=True))
        model.add(BatchNormalization())
        model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                             padding='same', return_sequences=True))
        model.add(BatchNormalization())
        '''
        model.add(ConvLSTM2D(filters=3, kernel_size=(3, 3),
                             padding='same', return_sequences=False))
        model.add(BatchNormalization())
        model.add(base_model)

        if preTrained:
            assert len(glob.glob(weight_seq_path + '*')) > 1, 'There is no weight file | {}'.format(weight_seq_path)
            model.load_weights(weight_seq_path)
            print("The model weights has been load")
            print(weight_seq_path)
        print(model.summary())

        return base_model, model

    elif key == 'resnet50' :
        if preTrained :
            base_model = ResNet50(include_top=False,
                                  weights='imagenet',
                                  input_shape=(input_size[0], input_size[1], 3), pooling='avg')
        else :
            base_model = ResNet50(include_top=False,
                                                    weights=None,
                                                    input_shape=(input_size[0], input_size[1], 3),
                                                    pooling='avg')

        x = base_model.output
        x = Dense(1024, activation='relu')(x)
        x = Dense(500, activation='relu')(x)
        output_ = Dense(2, activation='tanh')(x)

        model = Model(inputs=base_model.input,
                                      outputs=output_)

        return model

    elif key == 'resnet50_gru' :
        if preTrained :
            base_model = ResNet50(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(input_size[0], input_size[1], 3),
                                                    pooling='avg')
        else :
            base_model = ResNet50(include_top=False,
                                                    weights=None,
                                                    input_shape=(input_size[0], input_size[1], 3),
                                                    pooling='avg')

        input_ = tf.keras.Input(shape=(window_size, input_size[0], input_size[1], 3))
        for i in range(window_size):
            feature = base_model(input_[:, i, :, :, :])

            if i == 0:
                out_0 = tf.expand_dims(feature, axis=1)
            elif i == 1:
                out_1 = tf.expand_dims(feature, axis=1)
                output_ = tf.concat([out_0, out_1], axis=1)
            else:
                out_3 = tf.expand_dims(feature, axis=1)
                output_ = tf.concat([output_, out_3], axis=1)

        gru1 = GRU(1024, return_sequences=True)(output_)
        gru2 = GRU(512)(gru1)
        fo = Dense(2, activation='tanh')(gru2)

        model = Model(inputs=input_, outputs=fo)

        return model

    elif key == 'vgg19_gru' :
        if preTrained :
            base_model = VGG19(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(input_size[0], input_size[1], 3),
                                                    pooling='avg')
        else :
            base_model = VGG19(include_top=False,
                                                    weights=None,
                                                    input_shape=(input_size[0], input_size[1], 3),
                                                    pooling='avg')

        input_ = tf.keras.Input(shape=(window_size, input_size[0], input_size[1], 3))
        for i in range(window_size):
            feature = base_model(input_[:, i, :, :, :])

            if i == 0:
                out_0 = tf.expand_dims(feature, axis=1)
            elif i == 1:
                out_1 = tf.expand_dims(feature, axis=1)
                output_ = tf.concat([out_0, out_1], axis=1)
            else:
                out_3 = tf.expand_dims(feature, axis=1)
                output_ = tf.concat([output_, out_3], axis=1)

        gru = GRU(256)(output_)
        fo = Dense(2, activation='tanh')(gru)

        model = Model(inputs=input_, outputs=fo)

        return model

# @tf.function
def load_image(filename, image_size):
    # print(filename)
    try :
        raw = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, [image_size[0], image_size[1]])
        image = image / 255.0
    except :
        print("Image load error : ", filename)
        image = tf.zeros([image_size[0], image_size[1], 3])
    return image

def loadNresize(img_path, image_size) :
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size[0], image_size[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def normalize(img) :
    img = img / 255.0
    img = tf.expand_dims(img, axis=-1)
    return img

def load_td_image(images, image_size) :
    if os.path.isfile(images[0]) * os.path.isfile(images[1]) * os.path.isfile(images[2]) :
        img1 = loadNresize(images[0], image_size)
        img2 = loadNresize(images[1], image_size)
        img3 = loadNresize(images[2], image_size)

        _, diff1 = ssim(img1, img2, full=True)
        _, diff2 = ssim(img1, img3, full=True)

        base = normalize(img1)
        diff1 = normalize(diff1*-1)
        diff2 = normalize(diff2*-1)

        image_x = tf.concat([base, diff1, diff2], axis = -1)

        return image_x

    else :
        print(images)
        return -1

# Dataloader
class Dataloader_td(Sequence):
    def __init__(self, x, y, image_path, image_size, batch_size=1, shuffle=False):
        self.x, self.y, = x, y
        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x)) / float(self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))

        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # batch_x = [self.x[i] for i in indices]
        batch_y = []
        images = []
        for i in indices :
            image_x = load_td_image([os.path.join(self.image_path, self.x[i][(k*-1)]) for k in range(1, 6, 2)], self.image_size)
            images.append(image_x)
            batch_y.append(self.y[i])

        return tf.convert_to_tensor(images), tf.convert_to_tensor(batch_y)


class Dataloader(Sequence) :
    def __init__(self, x, y, image_path, image_size, batch_size=1, shuffle=False):
        self.x, self.y = x, y
        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x)) / float(self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))

        if self.shuffle == True :
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        image_x = [load_image(os.path.join(self.image_path, file_name), self.image_size) for file_name in batch_x]
        batch_y = [self.y[i] for i in indices]

        return tf.convert_to_tensor(image_x), tf.convert_to_tensor(batch_y)

class Dataloader_sequential(Sequence) :
    def __init__(self, x, y, i, image_path, image_size, batch_size=1, shuffle=False):
        self.x, self.y, self.i = x, y, i
        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x)) / float(self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))

        if self.shuffle == True :
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        images = []
        for file_list in batch_x :
            image_x = [load_image(os.path.join(self.image_path, file_name), self.image_size) for file_name in file_list]
            images.append(image_x)
        batch_y = [self.y[i] for i in indices]

        return tf.convert_to_tensor(images), tf.convert_to_tensor(batch_y)

def CCC_score_np(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)

    sxy = np.mean((x - x_mean) * (y - y_mean))

    ccc = (2 * sxy) / (x_var + y_var + np.power(x_mean - y_mean, 2))
    return ccc

def CCC_score(x, y):
    x_mean = tf.math.reduce_mean(x)
    y_mean = tf.math.reduce_mean(y)
    x_var = tf.math.reduce_variance(x)
    y_var = tf.math.reduce_variance(y)

    sxy = tf.math.reduce_mean((x - x_mean) * (y - y_mean))

    ccc = (2 * sxy) / (x_var + y_var + tf.math.pow(x_mean - y_mean, 2))
    return ccc

# Loss function
def loss_ccc(x, y) :
    items = [CCC_score(x[:, 0], y[:, 0]), CCC_score(x[:, 1], y[:, 1])]
    total_ccc = tf.math.reduce_mean(items)
    loss = 1 - total_ccc
    return loss

# Metric function
def metric_CCC(x, y):
    cccs = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    # total_ccc = tf.math.reduce_mean(items)
    return cccs

# Fill in the blank
def fib(path) :
    file_list = os.listdir(path)

    save_path = os.path.join(path, 'fib')

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for file in file_list :
        txt_list = read_txt(os.path.join(path, file))
        save_file_path = os.path.join(save_path, file)

        flag = False

        for i, txt in enumerate(txt_list) :
            if i > 0 :
                line = [float(x) for x in txt]
                v = line[0]
                a = line[1]

                if v == -5 or a == -5 :
                    if not flag :
                        content = "{},{}\n".format(v, a)
                        f.write(content)
                    else :
                        content = "{},{}\n".format(v_pre, a_pre)
                        f.write(content)
                else :
                    content = "{},{}\n".format(v, a)
                    f.write(content)

                    v_pre = v
                    a_pre = a

                    flag = True

            else :
                f = open(save_file_path, "w")
                content = "valence,arousal\n"
                f.write(content)

        f.close()

def compare(path) :
    prediction_path = os.path.join(path, '*.txt')
    prediction_lists = glob.glob(prediction_path)
    # prediction_lists.pop(prediction_lists.index('Weight.txt'))
    print(prediction_lists)

    total_ccc_V = []
    total_ccc_A = []

    for i, prediction_list in enumerate(prediction_lists) :
        predictions = read_txt(os.path.join(prediction_path, prediction_list))

        pred = []
        for i in range(len(predictions)-1) :
            pred.append([float(x) for x in predictions[(i+1)]])

        pred = np.array(pred)

        predictions_V = pred[:, :1]
        predictions_A = pred[:, 1:]

        gts = read_txt(os.path.join(PATH_DATA, 'annotations', 'VA_Set', 'Validation_Set', prediction_list))

        gt = []
        for j in range(len(gts)-1) :
            gt.append([float(x) for x in gts[(j+1)]])
        gt = np.array(gt)

        gts_V = gt[:, :1]
        gts_A = gt[:, 1:]

        valence_ccc_score = CCC_score_np(predictions_V, gts_V[:len(predictions_V)])
        arousal_ccc_score = CCC_score_np(predictions_A, gts_A[:len(predictions_A)])

        total_ccc_V.append(valence_ccc_score)
        total_ccc_A.append(arousal_ccc_score)

        print("{} : {:.4f}, {:.4f}".format(prediction_list, valence_ccc_score, arousal_ccc_score))

    ccc_V = sum(total_ccc_V) / len(total_ccc_V)
    ccc_A = sum(total_ccc_A) / len(total_ccc_A)
    ccc_M = (ccc_V + ccc_A) / 2

    print("")
    print("Comparision result!!")
    print("The CCC value of valence is {:.4f}".format(ccc_V))
    print("The CCC value of arousal is {:.4f}".format(ccc_A))
    print("Total CCC value is {:.4f}".format(ccc_M))


if __name__ == '__main__' :
    import argparse
    import configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='/home/gsethan/Desktop/ABAW2021/results/evaluation/614_13_15_FER/raw/',
                        help='Enter the path which has desired files')

    parser.add_argument('--mode', default='compare',
                        help='Enter the desired mode')
    parser.add_argument('--location', default='205',
                        help='Enter the server environment to be trained on')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('./config.ini')

    PATH_DATA = config[args.location]['PATH_DATA']

    if args.mode == 'compare' :
        compare(args.path)
    elif args.mode == 'fib' :
        fib(args.path)
