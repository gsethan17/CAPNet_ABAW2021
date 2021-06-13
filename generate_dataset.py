import os
import copy
import pickle
import glob
import numpy as np
from utils import read_pickle, read_csv, read_txt


PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
PATH_SWITCH_INFO = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError')
PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
# PATH_DATA = os.path.join(os.getcwd(), 'data')

FPS = 30

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

def generate_single_data() :
    list_subjects = sorted([x.rstrip('.csv') for x in os.listdir(PATH_DATA_GUIDE) if x.endswith('.csv')])

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

    train_save_path = os.path.join(PATH_DATA, 'va_train_list.pickle')

    with open(train_save_path, 'wb') as f :
        pickle.dump(train_data, f)


    print('Validation data')

    val_x_list, val_y_list = get_samples(dic_vals, switch_images, switch_subjects)
    print(len(val_x_list))
    print(len(val_y_list))

    val_data = {
        'x' : val_x_list,
        'y' : val_y_list
    }

    val_save_path = os.path.join(PATH_DATA, 'va_val_list.pickle')

    with open(val_save_path, 'wb') as f :
        pickle.dump(val_data, f)

def get_sequence_data(subject_name, images_list, window_size, stride) :
    total_x = []
    total_y = []

    for i in range(len(images_list)) :
        if i >= FPS * window_size :

            # get images
            list_x = []
            for j in range((FPS*window_size // stride)+1) :

                idx = i-(stride * j)

                if images_list[idx] is "" :

                    if idx == 0 :
                        list_x.append(os.path.join(subject_name, images_list[idx] + '.jpg'))

                    else :
                        for k in range(idx-1, idx-stride, -1) :
                            flag = False

                            if not images_list[k] is "" :
                                list_x.append(os.path.join(subject_name, images_list[k] + '.jpg'))
                                flag = True
                                break

                        if not flag :
                            list_x.append(os.path.join(subject_name, images_list[idx] + '.jpg'))

                else :
                    list_x.append(os.path.join(subject_name, images_list[idx] + '.jpg'))

            list_x.reverse()
            total_x.append(list_x)

            # get labels
            path = os.path.join(PATH_DATA, 'annotations', 'VA_Set', '**', subject_name + '.txt')
            path = glob.glob(path)[0]

            list_labels = read_txt(path)
            try :
                total_y.append([float(x) for x in list_labels[(i + 1)]])
            except :
                print(subject_name, i)

    return total_x, total_y


def generate_sequential_data(window_size, stride) :
    # Train data
    train_data = {
        'x': [],
        'y': []
    }

    train_subject_lists = read_csv(os.path.join(PATH_DATA, 'va_train_set.csv'))
    for train_subject_list in train_subject_lists :
        train_images = read_csv(os.path.join(PATH_DATA_GUIDE, train_subject_list+'.csv'))

        train_x, train_y = get_sequence_data(train_subject_list, train_images, window_size, stride)

        train_data['x'] += train_x
        train_data['y'] += train_y

        print(np.array(train_data['x']).shape, np.array(train_data['y']).shape, np.array(train_x).shape)

    print(np.array(train_data['x']).shape, np.array(train_data['y']).shape)
    if np.array(train_data['x']).shape[0] == np.array(train_data['y']).shape[0] :
        with open(os.path.join(PATH_DATA, 'va_train_seq_list.pickle'), 'wb') as f:
            pickle.dump(train_data, f)


    # Validation data
    val_data = {
        'x': [],
        'y': []
    }

    val_subject_lists = read_csv(os.path.join(PATH_DATA, 'va_val_set.csv'))
    for val_subject_list in val_subject_lists:
        val_images = read_csv(os.path.join(PATH_DATA_GUIDE, val_subject_list + '.csv'))

        val_x, val_y = get_sequence_data(val_subject_list, val_images, window_size, stride)

        val_data['x'] += val_x
        val_data['y'] += val_y

        print(np.array(val_data['x']).shape, np.array(val_data['y']).shape, np.array(val_x).shape)

    print(np.array(val_data['x']).shape, np.array(val_data['y']).shape)
    if np.array(val_data['x']).shape[0] == np.array(val_data['y']).shape[0] :
        with open(os.path.join(PATH_DATA, 'va_val_seq_list.pickle'), 'wb') as f:
            pickle.dump(val_data, f)




if __name__ == "__main__" :
    # generate_single_data()

    window_size = 3
    stride = 10

    generate_sequential_data(window_size, stride)