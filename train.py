from utils import get_model, Dataset_generator, metric_CCC, read_csv, read_pickle, Dataloader, CCC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
import os

PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
# PATH_DATA = os.path.join(os.getcwd(), 'data')
IMAGE_PATH = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/images/cropped'
# IMAGE_PATH = os.path.join(PATH_DATA, 'images', 'cropped')

MODEL_KEY = 'resnet50'  # 'pretrainedFER' / 'resnet50'
PRETRAINED = True

EPOCHS = 15
BATCH_SIZE = 32

LEARNING_RATE = 0.001
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
LOSS = MSE


def main() :
    train_path = os.path.join(PATH_DATA, 'va_train_list.pickle')
    train_data = read_pickle(train_path)

    val_path = os.path.join(PATH_DATA, 'va_val_list.pickle')
    val_data = read_pickle(val_path)

    train_dataloader = Dataloader(x=train_data['x'], y=train_data['y'], image_path=IMAGE_PATH, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = Dataloader(x=val_data['x'], y=val_data['y'], image_path=IMAGE_PATH, batch_size=BATCH_SIZE, shuffle=True)

    # print(train_dataloader[0])

    # Data Loader setup
    # Dataloader = Dataset_generator(PATH_DATA_GUIDE, batch_size=BATCH_SIZE)
    # print(Dataloader.get_count())


    # Model Loader setup
    # model = get_model(key=MODEL_KEY,pretrained=PRETRAINED)
    model = get_model(key=MODEL_KEY, preTrained=PRETRAINED)

    # print(model.summary())

    # Model setup
    model.compile(optimizer=OPTIMIZER, loss=LOSS)

    model.fit(x=train_dataloader,
              epochs=EPOCHS,
              # callbacks=[],
              validation_data=val_dataloader,
              shuffle=True,
              verbose=2)
    '''
    # train
    input_, label_ = Dataloader.get_trainData()
    print(input_.shape)
    output_ = model.predict(input_)
    print(output_.shape)
    
    loss = metric_CCC
    
    loss, metric = loss(output_, label_)
    loss_v = loss[0]
    loss_a = loss[1]
    
    # gradiant tape
    '''

if __name__ == "__main__" :
    main()
