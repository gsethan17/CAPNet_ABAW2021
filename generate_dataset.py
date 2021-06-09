import os
import copy
import pickle
import glob
from utils import read_pickle, read_csv, read_txt


PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
PATH_SWITCH_INFO = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError')
PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
# PATH_DATA = os.path.join(os.getcwd(), 'data')


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

def count(dic):
    count = 0
    count_na = 0

    for k in dic :
        count += len(dic[k])
        count_na += dic[k].count("")

    return count-count_na

dic_trains_count = count(dic_trains)
dic_vals_count = count(dic_vals)

print(dic_trains_count)
print(dic_vals_count)

switch_images = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_images.pickle'))
switch_subjects = read_pickle(os.path.join(PATH_SWITCH_INFO, 'switch_subjects.pickle'))

def get_samples(dic) :
    list_x = []
    list_y = []

    na_count = 0

    for i, name in enumerate(dic.keys()) :

        if not name == "30-30-1920x1080_right" or name == "30-30-1920x1080_left" :
            continue
        for j, image in enumerate(dic[name]) :

            if not image :
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
                print("{} / {} || {} / {} || {} / {} / {}".format(i + 1, len(dic.keys()), j + 1, len(dic[name]),
                                                                  len(list_x), len(list_y), na_count), end='\r')
                print(name, object, image)
                assert os.path.isfile(os.path.join(PATH_DATA, 'images', 'cropped', os.path.join(object, image))), "{} file is not exist".format(os.path.join(object, image))

            list_x.append(os.path.join(object, image))

            idx = dic[name].index(image)
            path = os.path.join(PATH_DATA, 'annotations', 'VA_Set', '**', name + '.txt')
            path = glob.glob(path)[0]

            list_labels = read_txt(path)
            list_y.append([float(x) for x in list_labels[(idx+1)]])

            print("{} / {} || {} / {} || {} / {} / {}".format(i+1, len(dic.keys()), j+1, len(dic[name]), len(list_x), len(list_y), na_count), end='\r')

    return list_x, list_y


print('Training data')
train_x_list, train_y_list = get_samples(dic_trains)
print('')
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

val_x_list, val_y_list = get_samples(dic_vals)
print('')
print(len(val_x_list))
print(len(val_y_list))

val_data = {
    'x' : val_x_list,
    'y' : val_y_list
}

val_save_path = os.path.join(PATH_DATA, 'va_val_list.pickle')

with open(val_save_path, 'wb') as f :
    pickle.dump(val_data, f)

