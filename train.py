from utils import get_model, Dataset_generator, loss_ccc, metric_CCC, read_csv, read_pickle, Dataloader, CCC
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import time

PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
PATH_DATA = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/'
# PATH_DATA = os.path.join(os.getcwd(), 'data')
IMAGE_PATH = '/home/gsethan/Documents/Aff-Wild2-ICCV2021/images/cropped'
# IMAGE_PATH = os.path.join(PATH_DATA, 'images', 'cropped')

MODEL_KEY = 'pretrainedFER'  # 'pretrainedFER' / 'resnet50'
PRETRAINED = True
# Model load to global variable
MODEL = get_model(key=MODEL_KEY, preTrained=PRETRAINED)

EPOCHS = 30
BATCH_SIZE = 128
SHUFFLE = True

LEARNING_RATE = 0.001
OPTIMIZER = Adam(learning_rate=LEARNING_RATE)
LOSS = loss_ccc
METRIC = metric_CCC

tm = time.localtime(time.time())

SAVE_PATH = os.path.join(os.getcwd(),
                         'results',
                         '{}{}_{}{}_{}'.format(tm.tm_mon,
                                                 tm.tm_mday,
                                                 tm.tm_hour,
                                                 tm.tm_min,
                                                 MODEL_KEY))
if not os.path.isdir(SAVE_PATH) :
    os.makedirs(SAVE_PATH)

@tf.function
def train_step(X, Y) :
    global MODEL
    global LOSS
    global METRIC
    global OPTIMIZER

    with tf.GradientTape() as tape :
        predictions = MODEL(X)
        loss = LOSS(predictions, Y)
        metric = METRIC(predictions, Y)

    gradients = tape.gradient(loss, MODEL.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients, MODEL.trainable_variables))

    return loss, metric

@tf.function
def val_step(X, Y) :
    global MODEL
    global LOSS
    global METRIC

    predictions = MODEL(X)
    loss = LOSS(predictions, Y)
    metric = METRIC(predictions, Y)

    return loss, metric


def main() :
    train_path = os.path.join(PATH_DATA, 'va_train_list.pickle')
    train_data = read_pickle(train_path)

    val_path = os.path.join(PATH_DATA, 'va_val_list.pickle')
    val_data = read_pickle(val_path)

    print("Build the data loader")
    st_build = time.time()
    train_dataloader = Dataloader(x=train_data['x'], y=train_data['y'], image_path=IMAGE_PATH, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    ed_train = time.time()
    print("Train data has been build ({:.1f}seconds).".format(ed_train - st_build))

    val_dataloader = Dataloader(x=val_data['x'], y=val_data['y'], image_path=IMAGE_PATH, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    ed_val = time.time()
    print("Validation data has been build ({:.1f}seconds).".format(ed_val - ed_train))

    # print(train_dataloader[0])

    # Data Loader setup
    # Dataloader = Dataset_generator(PATH_DATA_GUIDE, batch_size=BATCH_SIZE)
    # print(Dataloader.get_count())


    # Model Loader setup
    # model = get_model(key=MODEL_KEY,pretrained=PRETRAINED)
    # model = get_model(key=MODEL_KEY, preTrained=PRETRAINED)

    # print(model.summary())

    # Model setup
    ## use tensorflow API
    # model.compile(optimizer=OPTIMIZER, loss=LOSS)

    # model.fit(x=train_dataloader,
    #           epochs=EPOCHS,
    #           # callbacks=[],
    #           validation_data=val_dataloader,
    #           shuffle=True,
    #           verbose=2)

    ## use gradient tape
    results = {}
    results['train_loss'] = []
    results['train_ccc_V'] = []
    results['train_ccc_A'] = []
    results['train_CCC'] = []

    results['val_loss'] = []
    results['val_ccc_V'] = []
    results['val_ccc_A'] = []
    results['val_CCC'] = []

    print("Training Start...")
    for epoch in range(EPOCHS) :
        train_loss = []
        train_metric_V = []
        train_metric_A = []
        train_metric_C = []

        st_train = time.time()
        for i in range(len(train_dataloader)) :

            x_train, y_train = train_dataloader[i]
            print("Training : {} / {}".format(i + 1, len(train_dataloader)), end="\r")
            train_temp_loss, train_temp_metric = train_step(x_train, y_train)
            train_loss.append(train_temp_loss)
            train_metric_V.append(train_temp_metric[0])
            train_metric_A.append(train_temp_metric[1])
            train_metric_C.append(tf.math.reduce_mean(train_temp_metric))
        ed_train = time.time()
        print("Training is completed...", end="\r")

        results['train_loss'].append(tf.math.reduce_mean(train_loss))
        results['train_ccc_V'].append(tf.math.reduce_mean(train_metric_V))
        results['train_ccc_A'].append(tf.math.reduce_mean(train_metric_A))
        results['train_CCC'].append(tf.math.reduce_mean(train_metric_C))

        val_loss = []
        val_metric_V = []
        val_metric_A = []
        val_metric_C = []

        print("Validation Start...", end="\r")
        for j in range(len(val_dataloader)) :

            x_val, y_val = val_dataloader[j]
            print("Validation : {} / {}".format(j + 1, len(val_dataloader)), end="\r")
            val_temp_loss, val_temp_metric = val_step(x_val, y_val)
            val_loss.append(val_temp_loss)
            val_metric_V.append(val_temp_metric[0])
            val_metric_A.append(val_temp_metric[1])
            val_metric_C.append(tf.math.reduce_mean(val_temp_metric))

        ed_val = time.time()

        if tf.math.reduce_mean(val_temp_metric) > tf.math.reduce_max(results['val_CCC']) :
            # save best weights
            MODEL.save_weights(os.path.join(SAVE_PATH, "best_weights"))

        results['val_loss'].append(tf.math.reduce_mean(val_loss))
        results['val_ccc_V'].append(tf.math.reduce_mean(val_metric_V))
        results['val_ccc_A'].append(tf.math.reduce_mean(val_metric_A))
        results['val_CCC'].append(tf.math.reduce_mean(val_metric_C))
        
        print("{:>3} / {:>3} \t||\t train_loss:{:8.4f}, train_CCC:{:8.4f}, val_loss:{:8.4f}, val_CCC:{:8.4f} \t||\t TIME: Train {:8.1f}sec, Validation {:8.1f}sec".format(epoch + 1, EPOCHS,
                                                                                      results['train_loss'][-1],
                                                                                      results['train_CCC'][-1],
                                                                                      results['val_loss'][-1],
                                                                                      results['val_CCC'][-1],
                                                                                      (ed_train - st_train),
                                                                                      (ed_val - ed_train)))

        # early stop
        if epoch > 5 :
            if results['val_CCC'][-5] > tf.math.reduce_max(results['val_CCC'][-4:]) :
                break

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_PATH, 'Results.csv'), index=False)

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
