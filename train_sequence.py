from utils import get_model
import tensorflow as tf
import os
from utils import read_pickle, Dataloader_sequential

PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
# PATH_DATA = os.path.join(os.getcwd(), 'data')
IMAGE_PATH = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/images/cropped'
# IMAGE_PATH = os.path.join(PATH_DATA, 'images', 'cropped')

TRAIN_DATA_PATH = os.path.join(PATH_DATA, 'va_train_seq_list.pickle')   # 'va_train_list.pickle' / 'va_train_seq_list.pickle'
VAL_DATA_PATH = os.path.join(PATH_DATA, 'va_val_seq_list.pickle')   # 'va_val_list.pickle' / 'va_val_seq_list.pickle'

INPUT_IMAGE_SIZE = (224, 224)

EPOCHS = 30
BATCH_SIZE = 1
SHUFFLE = False

# model load
'''
MODEL_KEY = 'FER_LSTM'  # 'FER' / 'FER_LSTM' / 'resnet50' / 'resnet50_gru' / 'vgg19_gru'
PRETRAINED = True

WINDOW_SIZE = 90
STRIDE = 9

# Model load to global variable
MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED, window_size=WINDOW_SIZE // STRIDE))


print(MODEL.summary())
'''

def main() :
    train_data = read_pickle(TRAIN_DATA_PATH)
    val_data = read_pickle(VAL_DATA_PATH)

    lists = []
    for i in range(len(train_data['x'])) :
        count = 0
        for j, image_path in enumerate(train_data['x'][i]) :
            image = image_path.split('/')[1].split('.')[0]
            if image == '' :
                count += 1

        if count == 9 :
            lists.append(i)

    # print(train_data['x'][18694], end='\n')
    # print(train_data['y'][:10], end='\n')
    #
    # print(val_data['x'][:10], end='\n')
    # print(val_data['y'][:10], end='\n')

    train_dataloader = Dataloader_sequential(x=train_data['x'], y=train_data['y'], image_path=IMAGE_PATH,
                                  image_size=INPUT_IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    val_dataloader = Dataloader_sequential(x=val_data['x'], y=val_data['y'], image_path=IMAGE_PATH, image_size=INPUT_IMAGE_SIZE,
                                batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    print(train_dataloader[lists[0]])
    # print(val_dataloader[0])

# input_ = tf.ones((1, 10, 112, 112, 3))

# output_ = MODEL.predict(input_)
#
# print(output_)

if __name__ == '__main__' :
    main()