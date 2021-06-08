from utils import get_model, Dataset_generator, metric_CCC
import os

PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
BATCH_SIZE = 32

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
