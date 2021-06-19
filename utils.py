import os
from glob import glob
import random
import csv
import pickle
import numpy as np
import tensorflow as tf
import copy
import glob
import time
from base_model.ResNet import ResNet34
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import Loss
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Dense, ConvLSTM2D, BatchNormalization
from tensorflow.keras.applications import ResNet50, VGG19
import cv2
from skimage.metrics import structural_similarity as ssim


# PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
# PATH_SWITCH_INFO = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError')
# PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
# PATH_DATA = os.path.join(os.getcwd(), 'data')

# INPUT_IMAGE_SIZE = (224, 224)

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
              window_size = 10, input_size=(224,224)) :
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

        lstm = LSTM(256, input_shape=(window_size, 512))(output_)
        fo = Dense(2, activation = 'tanh')(lstm)

        model = Model(inputs=input_, outputs=fo)

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

def reshape(img, image_size) :
    img = tf.image.resize(img, [image_size[0], image_size[1]])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    return img

def load_td_image(images, image_size) :
    img1 = cv2.imread(images[0])
    img2 = cv2.imread(images[1])
    img3 = cv2.imread(images[2])

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    _, diff1 = ssim(img1_gray, img2_gray, full=True)
    _, diff2 = ssim(img1_gray, img3_gray, full=True)

    base = reshape(img1_gray, image_size)
    diff1 = reshape(diff1*-1, image_size)
    diff2 = reshape(diff2*-1, image_size)

    image_x = tf.concat([base, diff1, diff2], axis = 0)

    return image_x

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

        batch_x = [self.x[i] for i in indices]
        images = []
        for file_list in batch_x:
            image_x = load_td_image([os.path.join(self.image_path, file_list[(k*-1)]) for k in range(1, 6, 2)], self.image_size)
            images.append(image_x)

        batch_y = [self.y[i] for i in indices]

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

'''
class CCC(Loss) :
    def __init__(self, name="ccc"):
        super().__init__(name=name)

    def CCC_score(self, x, y):
        
        vx = x - np.mean(x)
        vy = y - np.mean(y)
        rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
        x_m = np.mean(x)
        y_m = np.mean(y)
        x_s = np.std(x)
        y_s = np.std(y)
        ccc = 2 * rho * x_s * y_s / (x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)
        
        # Concordance Correlation Coefficient
        # sxy = np.sum((x - np.mean(x)) * (y - np.mean(y))) / x.shape[0]
        # ccc = 2 * sxy / (np.var(x) + np.var(y) + (np.mean(x) - np.mean(y)) ** 2)
        x_mean = tf.math.reduce_mean(x)
        y_mean = tf.math.reduce_mean(y)
        x_var = tf.math.reduce_variance(x)
        y_var = tf.math.reduce_variance(y)

        sxy = tf.math.reduce_mean((x - x_mean) * (y - y_mean))

        ccc = (2 * sxy) / (x_var + y_var + tf.math.pow(x_mean-y_mean, 2))

        return ccc

    def call(self, y_pred, y_true):
        items = [self.CCC_score(y_pred[:, 0], y_true[:, 0]), self.CCC_score(y_pred[:, 1], y_true[:, 1])]
        total_ccc = tf.math.reduce_mean(items)
        return total_ccc, items


# Dataset
class Dataset_generator() :
    def __init__(self, path, batch_size = 1, random_split = False):
        self.path = path
        self.list_subjects = sorted([x.rstrip('.csv') for x in os.listdir(path) if x.endswith('.csv')])
        self.total_samples = self.get_samples()
        self.list_trains, self.list_vals = self.split(random_split=random_split)
        self.num_subjects = len(self.list_subjects)
        self.batch_size = batch_size

    def count(self, dic):
        count = 0
        count_na = 0

        for k in dic :
            count += len(dic[k])
            count_na += dic[k].count("")        

        return count-count_na


    def get_count(self) :
        return self.count(self.list_trains), self.count(self.list_vals)

    def reset(self):
        self.list_trains, self.list_vals = self.split()

    def split_samples(self, subject_list):
        dic = {}
        for subject in subject_list :
            if subject in self.total_samples.keys() :
                dic[subject] = copy.deepcopy(self.total_samples[subject])
        return dic

    def split(self, random_split):

        if not random_split :
            list_trains = read_csv(os.path.join(PATH_DATA, 'va_train_set.csv'))
            list_vals = read_csv(os.path.join(PATH_DATA, 'va_val_set.csv'))

            dic_trains = self.split_samples(list_trains)
            dic_vals = self.split_samples(list_vals)

            return dic_trains, dic_vals

        else :
            list_total = self.list_subjects
            random.shuffle(list_total)
            idx = int(len(list_total) * 0.2)
            list_vals = list_total[:idx]
            list_trains = list_total[idx:]

            dic_trains = self.split_samples(list_trains)
            dic_vals = self.split_samples(list_vals)

            return dic_trains, dic_vals


    def get_samples(self):
        samples = {}

        for i, name in enumerate(self.list_subjects) :

            file_path = os.path.join(self.path, name + '.csv')

            samples[name] = read_csv(file_path)

        return samples


    def get_trainData(self):
        candi_subject_name = []
        candi_img_name = []

        while len(candi_img_name) < self.batch_size :

            candi_subject = random.choice(list(self.list_trains.keys()))
            candi_inputs = self.list_trains[candi_subject]
            while "" in candi_inputs :
                candi_inputs.remove("")

            if len(candi_inputs) > 0 :
                candi = random.choice(candi_inputs)
                candi_idx = self.list_trains[candi_subject].index(candi)

                candi_subject_name.append(candi_subject)
                candi_img_name.append(self.list_trains[candi_subject].pop(candi_idx))

            else :
                del self.list_trains[candi_subject]
                continue

        print(candi_subject_name, candi_img_name)
        # get inputs
        inputs = self.get_inputs(candi_subject_name, candi_img_name)

        # ger labels
        labels = self.get_labels(candi_subject_name, candi_img_name)

        return inputs, labels

    def get_valData(self):
        candi_subject_name = []
        candi_img_name = []

        while len(candi_img_name) < self.batch_size :

            candi_subject = random.choice(list(self.list_vals.keys()))
            candi_inputs = self.list_vals[candi_subject]
            while "" in candi_inputs :
                candi_inputs.remove("")

            if len(candi_inputs) > 0 :
                candi = random.choice(candi_inputs)
                candi_idx = self.list_vals[candi_subject].index(candi)

                candi_subject_name.append(candi_subject)
                candi_img_name.append(self.list_vals[candi_subject].pop(candi_idx))

            else :
                del self.list_vals[candi_subject]
                continue

        # get inputs
        inputs = self.get_inputs(candi_subject_name, candi_img_name)

        # ger labels
        labels = self.get_labels(candi_subject_name, candi_img_name)

        return inputs, labels

    @tf.function
    def load_image(self, filename):
        raw = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, [INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]])
        image = image / 255.0
        return image

    def get_inputs(self, list_subject_name, list_img_name):
        # assert len(list_video_name) == len(list_img_name), 'There is as error in get_trainData function.'
        # inputs = tf.zeros((self.batch_size, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1], 3))

        switch_images = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_images.pickle'))
        switch_subjects = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_subjects.pickle'))

        for i in range(self.batch_size) :
            subject_name = list_subject_name[i]
            img_name = list_img_name[i]

            ###############################
            if subject_name in switch_images.keys() :
                if img_name in switch_images[subject_name] :
                    alti_subject_name = switch_subjects[subject_name]
                    image_path = os.path.join(PATH_DATA, 'images', 'cropped', alti_subject_name, img_name)
                else :
                    image_path = os.path.join(PATH_DATA, 'images', 'cropped', subject_name, img_name)


            else :
                image_path = os.path.join(PATH_DATA, 'images', 'cropped', subject_name, img_name)
            ###############################

#            try :
            image = self.load_image(image_path)
#            except:
#                continue

            image = tf.expand_dims(image, axis = 0)

#            if i == 0 :
#                inputs = image
            try :
                inputs = tf.concat([inputs, image], axis = 0)
            except:
                inputs = image

        return inputs


    def get_labels(self, list_subject_name, list_img_name):
        # labels = tf.zeros((self.batch_size, 2))

        for i in range(self.batch_size) :
            subject_name = list_subject_name[i]
            img_name = list_img_name[i]

            idx = self.total_samples[subject_name].index(img_name)

            path = os.path.join(PATH_DATA, 'annotations', 'VA_Set', '**', subject_name + '.txt')
            path = glob.glob(path)[0]
            list_labels = read_txt(path)
            label = tf.cast([float(x) for x in list_labels[(idx+1)]], tf.float32)

            label = tf.expand_dims(label, axis = 0)

            if i == 0 :
                labels = label
            else :
                labels = tf.concat([labels, label], axis = 0)

        return labels
'''

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



if __name__ == '__main__' :

    dg = Dataset_generator(PATH_DATA_GUIDE, batch_size=128, random_split=True)
    num_tarin, num_val = dg.get_count()
    print(num_val)
    
    for i in range(num_val // 128) :
        input_, label_ = dg.get_valData()
#        print(input_.shape, label_.shape)

        _, num_val = dg.get_count()
        print(num_val)

#    ccc_v = CCC_score(label_[:,0], label__[:,0])
#    ccc_a = CCC_score(label_[:,1], label__[:,1])
  
#    print(ccc_v)
#    print(ccc_a)

#    items, mean_items = metric_CCC(label_, label__)
#    print(items)
#    print(mean_items)
    # sample_list = dg.total_samples['5-60-1920x1080-3']
    # for i, image in enumerate(sample_list) :
    #     if image == "" :
    #         print(i)




    # model = get_model()
    # model.build(input_shape = (None, 244, 244, 3))
    # print(model.summary())
    #
    # input_ = np.ones((1, 224, 224, 3))
    # output_ = model.predict(input_)
    # print(output_.shape)
    # print(output_)
