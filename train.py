from utils import get_model, Dataset_generator, metric_CCC, read_csv, read_pickle
import os

PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
IMAGE_PATH = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/images/cropped'

BATCH_SIZE = 32

train_path = os.path.join(PATH_DATA, 'va_train_list.pickle')
train_data = read_pickle(train_path)

val_path = os.path.join(PATH_DATA, 'va_val_list.pickle')
val_data = read_pickle(val_path)

train_dataloader = Dataloader(x=train_data['x'], y=train_data['y'], image_path=IMAGE_PATH, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = Dataloader(x=val_data['x'], y=val_data['y'], image_path=IMAGE_PATH, batch_size=BATCH_SIZE, shuffle=True)


# Data Loader setup
Dataloader = Dataset_generator(PATH_DATA_GUIDE, batch_size=BATCH_SIZE)
print(Dataloader.get_count())


# Model Loader setup
model = get_model()


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
