import os
from glob import glob
import random
import csv
import pickle
import numpy as np
import tensorflow as tf
import copy
import glob
from base_model.ResNet import ResNet34

PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
PATH_SWITCH_INFO = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError')
PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'

INPUT_IMAGE_SIZE = (224, 224)

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
def get_model(key='FER', preTrained = True) :
    if key == 'FER' :
        # Model load
        model = ResNet34(cardinality = 32, se = 'parallel_add')
        
        if preTrained :
            # load pre-trained weights
            weight_path = os.path.join(os.getcwd(), 'base_model', 'ResNeXt34_Parallel_add', 'checkpoint_4_300000-320739.ckpt')
            assert len(glob.glob(weight_path + '*')) > 1, 'There is no weight file | {}'.format(weight_path)
            model.load_weights(weight_path)
        
        return model

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
        switch_subjecs = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_subjects.pickle'))

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

# Metric
def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def metric_CCC(x, y):
    items = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    return items, sum(items) / 2



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
