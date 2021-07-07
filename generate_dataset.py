import os
import copy
import pickle
import glob
import numpy as np
from utils import read_pickle, read_csv, read_txt
import argparse
import configparser
import cv2

# Basic configuration
parser = argparse.ArgumentParser()
parser.add_argument('--location', default='205',
                    help='Enter the server environment to be trained on')
parser.add_argument('--mode', default='test',
                        help='Enter the desired mode, train or test')
parser.add_argument('--type', default='sequence',
                        help='Enter the desired data type, single or sequence')

args = parser.parse_args()

config = configparser.ConfigParser()
config.read('./config.ini')

PATH_DATA = config[args.location]['PATH_DATA']
PATH_DATA_GUIDE = config[args.location]['PATH_DATA_GUIDE']
PATH_SWITCH_INFO = config[args.location]['PATH_SWITCH_INFO']

WINDOW_SIZE = int(config['INPUT']['WINDOW_SIZE'])
FPS = 30
STRIDE = 10

def count(dic):
    count = 0
    count_na = 0

    for k in dic :
        count += len(dic[k])
        count_na += dic[k].count("")

    return count-count_na

def get_samples(dic, switch_images, switch_subjects) :
    list_x = []
    list_y = []

    na_count = 0
    error_list = []

    for i, name in enumerate(dic.keys()) :

        for j, image in enumerate(dic[name]) :

            if image == "" :
                na_count += 1
                continue

            if name in switch_subjects.keys() :
                if image in switch_images[name] :
                    object = switch_subjects[name]
                else :
                    object = name
            else:
                object = name

            if not os.path.isfile(os.path.join(PATH_DATA, 'images', 'cropped', os.path.join(object, image))) :
                # print("{} / {} || {} / {} || {} / {} / {}".format(i + 1, len(dic.keys()), j + 1, len(dic[name]),
                #                                                   len(list_x), len(list_y), na_count), end='\r')
                # print(name, object, image)
                error_list.append([name, object, image])
                # assert os.path.isfile(os.path.join(PATH_DATA, 'images', 'cropped', os.path.join(object, image))), "{} file is not exist".format(os.path.join(object, image))

            list_x.append(os.path.join(object, image))

            idx = dic[name].index(image)
            path = os.path.join(PATH_DATA, 'annotations', 'VA_Set', '**', name + '.txt')
            path = glob.glob(path)[0]

            list_labels = read_txt(path)
            list_y.append([float(x) for x in list_labels[(idx+1)]])

            print("{} / {} || {} / {} || {} / {} / {} / {}".format(i+1, len(dic.keys()), j+1, len(dic[name]), len(list_x), len(list_y), na_count, len(error_list)), end='\r')
    print('')
    print(error_list)
    return list_x, list_y

def filtering_invalid(dic) :
    lists = []
    for i in range(len(dic['y'])) :
        if dic['y'][i][0] == -5 or dic['y'][i][1] == -5 :
            lists.append(i)

    print("{} invalid data is detected".format(len(lists)))

    lists.reverse()
    for j in lists :
        keys = dic.keys()
        for key in keys :
            dic[key].pop(j)

    return dic

def filtering_topfull(dic) :
    lists = []
    for i in range(len(dic['x'])) :
        count = 0
        for j, image_path in enumerate(dic['x'][i]) :
            image = image_path.split('/')[1].split('.')[0]
            if image == "" :
                count += 1


        if count == (FPS * WINDOW_SIZE // STRIDE) + 1 :
            lists.append(i)

    print("{} invalid data is detected".format(len(lists)))

    lists.reverse()
    for j in lists :
        keys = dic.keys()
        for key in keys :
            dic[key].pop(j)

    return dic


def generate_single_train() :
    list_subjects = sorted([x.rstrip('.csv') for x in os.listdir(PATH_DATA_GUIDE) if x.endswith('.csv')])
    print(list_subjects)
    return -1
    total_samples = {}

    for i, name in enumerate(list_subjects):
        file_path = os.path.join(PATH_DATA_GUIDE, name + '.csv')

        total_samples[name] = read_csv(file_path)

    list_trains = read_csv(os.path.join(PATH_DATA, 'va_train_set.csv'))
    list_vals = read_csv(os.path.join(PATH_DATA, 'va_val_set.csv'))

    dic_trains = {}
    for subject in list_trains:
        if subject in total_samples.keys():
            dic_trains[subject] = copy.deepcopy(total_samples[subject])

    dic_vals = {}
    for subject in list_vals:
        if subject in total_samples.keys():
            dic_vals[subject] = copy.deepcopy(total_samples[subject])

    dic_trains_count = count(dic_trains)
    dic_vals_count = count(dic_vals)

    print(dic_trains_count)
    print(dic_vals_count)

    switch_images = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_images.pickle'))
    switch_subjects = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_subjects.pickle'))

    print('Training data')
    train_x_list, train_y_list = get_samples(dic_trains, switch_images, switch_subjects)
    print(len(train_x_list))
    print(len(train_y_list))

    train_data = {
        'x' : train_x_list,
        'y' : train_y_list
    }

    train_data = filtering_invalid(train_data)

    train_save_path = os.path.join(PATH_DATA, 'va_train_single.pickle')

    with open(train_save_path, 'wb') as f :
        pickle.dump(train_data, f)
        print("Total {} samples are generated".format(len(train_data['x'])))


    print('Validation data')

    val_x_list, val_y_list = get_samples(dic_vals, switch_images, switch_subjects)
    print(len(val_x_list))
    print(len(val_y_list))

    val_data = {
        'x' : val_x_list,
        'y' : val_y_list
    }

    val_data = filtering_invalid(val_data)

    val_save_path = os.path.join(PATH_DATA, 'va_val_single.pickle')

    with open(val_save_path, 'wb') as f :
        pickle.dump(val_data, f)
        print("Total {} samples are generated".format(len(val_data['x'])))

def switching(name, image, switch_images, switch_subjects) :
    if name in switch_subjects.keys():
        if image in switch_images[name]:
            object = switch_subjects[name]
        else:
            object = name
    else:
        object = name

    return object

def get_sequence_test(subject_name) :
    total_x = []
    total_idx = []

    if "_" in subject_name:
        if subject_name.split('_')[-1] == 'right' or subject_name.split('_')[-1] == 'left':
            video_pos = os.path.join(PATH_DATA, 'videos', '_'.join(subject_name.split('_')[:-1]) + '.*')
        else:
            video_pos = os.path.join(PATH_DATA, 'videos', subject_name + '.*')
    else:
        video_pos = os.path.join(PATH_DATA, 'videos', subject_name + '.*')

    if not len(glob.glob(video_pos)) == 1:
        print("Video path is not vaild : {}".format(subject_name))
        return -1

    video_path = glob.glob(video_pos)[0]

    # count total number of frame
    capture = cv2.VideoCapture(video_path)
    total_len = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    base_dir = os.path.join(PATH_DATA, 'test_images_for_demo')

    for i in range(int(total_len)) :
        if i >= FPS * WINDOW_SIZE:

            # get images
            list_x = []
            for j in range((FPS * WINDOW_SIZE // STRIDE) + 1):

                idx = i - (STRIDE * j)

                image_path = os.path.join(base_dir, 'cropped', subject_name, '{:0>5}'.format(idx + 1) + '.jpg')

                if not os.path.isfile(image_path) :
                    if idx == 0 :
                        list_x.append(os.path.join(subject_name, ""))
                    else :
                        for k in range(idx - 1, idx - STRIDE, -1):
                            flag = False
                            image_path = os.path.join(base_dir, 'cropped', subject_name,
                                                      '{:0>5}'.format(idx + 1) + '.jpg')
                            if os.path.isfile(image_path) :
                                list_x.append(os.path.join(subject_name, '{:0>5}'.format(idx + 1) + '.jpg'))
                                flag = True
                                break

                        if not flag :
                            list_x.append(os.path.join(subject_name, ""))
                else :
                    list_x.append(os.path.join(subject_name, '{:0>5}'.format(idx + 1) + '.jpg'))

            list_x.reverse()

            # get labels & index
            try:
                total_x.append(list_x)
                total_idx.append([subject_name, i])

            except:
                print(subject_name, i)

    return total_x, total_idx



def get_sequence_data(subject_name, images_list, switch_images, switch_subjects) :
    total_x = []
    total_y = []
    total_idx = []

    path = os.path.join(PATH_DATA, 'annotations', 'VA_Set', '**', subject_name + '.txt')
    path = glob.glob(path)[0]
    list_labels = read_txt(path)

    for i in range(len(images_list)) :
        if i >= FPS * WINDOW_SIZE :

            # get images
            list_x = []
            for j in range((FPS*WINDOW_SIZE // STRIDE)+1) :

                idx = i-(STRIDE * j)

                if images_list[idx] is "" :

                    if idx == 0 :
                        name = switching(subject_name, images_list[idx], switch_images, switch_subjects)
                        list_x.append(os.path.join(name, images_list[idx]))

                    else :
                        for k in range(idx-1, idx-STRIDE, -1) :
                            flag = False

                            if not images_list[k] is "" :
                                name = switching(subject_name, images_list[k], switch_images, switch_subjects)
                                list_x.append(os.path.join(name, images_list[k]))
                                flag = True
                                break

                        if not flag :
                            name = switching(subject_name, images_list[idx], switch_images, switch_subjects)
                            list_x.append(os.path.join(name, images_list[idx]))

                else :
                    name = switching(subject_name, images_list[idx], switch_images, switch_subjects)
                    list_x.append(os.path.join(name, images_list[idx]))

            list_x.reverse()

            # get labels & index
            try :
                total_y.append([float(x) for x in list_labels[(i + 1)]])
                total_x.append(list_x)
                total_idx.append([subject_name, i])

            except :
                print(subject_name, i)

    return total_x, total_y, total_idx


def generate_sequential_data(type = 'test') :
    if type == 'test' :
        # test data
        test_data = {
            'x': [],
            'i': []
        }

        test_subject_lists = read_csv(os.path.join(PATH_DATA, 'va_test_set.csv'))
        for j, test_subject_list in enumerate(test_subject_lists):

            test_x, test_idx = get_sequence_test(test_subject_list)

            test_data['x'] += test_x
            test_data['i'] += test_idx

            print(j, len(test_subject_lists), np.array(test_data['x']).shape, np.array(test_x).shape)

        test_data = filtering_topfull(test_data)

        if np.array(test_data['x']).shape[0] == np.array(test_data['i']).shape[0]:
            with open(os.path.join(PATH_DATA, 'va_test_seq_list.pickle'), 'wb') as f:
                pickle.dump(test_data, f)


    else :
        # read switch info
        switch_images = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_images.pickle'))
        switch_subjects = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_subjects.pickle'))

        # Train data
        train_data = {
            'x': [],
            'y': [],
            'i': []
        }

        train_subject_lists = read_csv(os.path.join(PATH_DATA, 'va_train_set.csv'))
        for i, train_subject_list in enumerate(train_subject_lists) :
            train_images = read_csv(os.path.join(PATH_DATA_GUIDE, train_subject_list+'.csv'))

            train_x, train_y, train_idx = get_sequence_data(train_subject_list, train_images, switch_images, switch_subjects)

            train_data['x'] += train_x
            train_data['y'] += train_y
            train_data['i'] += train_idx

            print(i, len(train_subject_lists), np.array(train_data['x']).shape, np.array(train_data['y']).shape, np.array(train_x).shape)

        train_data = filtering_topfull(train_data)

        print(np.array(train_data['x']).shape, np.array(train_data['y']).shape)
        if np.array(train_data['x']).shape[0] == np.array(train_data['y']).shape[0] :
            with open(os.path.join(PATH_DATA, 'va_train_seq_list.pickle'), 'wb') as f:
                pickle.dump(train_data, f)


        # Validation data
        val_data = {
            'x': [],
            'y': [],
            'i': []
        }

        val_subject_lists = read_csv(os.path.join(PATH_DATA, 'va_val_set.csv'))
        for j, val_subject_list in enumerate(val_subject_lists):
            val_images = read_csv(os.path.join(PATH_DATA_GUIDE, val_subject_list + '.csv'))

            val_x, val_y, val_idx = get_sequence_data(val_subject_list, val_images, switch_images, switch_subjects)

            val_data['x'] += val_x
            val_data['y'] += val_y
            val_data['i'] += val_idx

            print(j, len(val_subject_lists), np.array(val_data['x']).shape, np.array(val_data['y']).shape, np.array(val_x).shape)

        val_data = filtering_topfull(val_data)

        print(np.array(val_data['x']).shape, np.array(val_data['y']).shape)
        if np.array(val_data['x']).shape[0] == np.array(val_data['y']).shape[0] :
            with open(os.path.join(PATH_DATA, 'va_val_seq_list.pickle'), 'wb') as f:
                pickle.dump(val_data, f)




if __name__ == "__main__" :

    if args.mode == 'test' :
        if args.type == 'sequence':
            generate_sequential_data()

    elif args.mode == 'train' :
        if args.type == 'single' :
            generate_single_train()
        elif args.type == 'sequence' :
            generate_sequential_data(type = 'val')
        else :
            print("Type variable is not valid")

    else :
        print("Mode variable is not valid")
