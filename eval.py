from utils import get_model, Dataset_generator, metric_CCC
import os


PATH_DATA_GUIDE = os.path.join(os.getcwd(), 'data_guide', 'dropDetectError', 'cropped')
BATCH_SIZE = 64

# Data Loader setup
Dataloader = Dataset_generator(PATH_DATA_GUIDE, batch_size=BATCH_SIZE)
_, num_total_sample_val = Dataloader.get_count()
print(num_total_sample_val)

# Model Loader setup
model = get_model()


# predict

iteration = num_total_sample_val // BATCH_SIZE

# set the dictionary for stack result
val_result = {}
val_result['ccc_v'] = []
val_result['ccc_a'] = []
val_result['ccc_mean'] = []

loss_function = metric_CCC

for i in range(iteration) :
    input_, label_ = Dataloader.get_valData()
    output_ = model.predict(input_)

    loss, metric = loss_function(output_, label_)
    val_result['ccc_v'].append(loss[0])
    val_result['ccc_a'].append(loss[1])
    val_result['ccc_mean'].append(metric)

    _, num_total_sample_val = Dataloader.get_count()

    print("{:>6} / {:>6} {:=8} samples ||\tCCC(val:{:8.4f}, aro:{:8.4f}, mean:{:8.4f})".format(i+1, iteration, num_total_sample_val, val_result['ccc_v'][-1], val_result['ccc_a'][-1], val_result['ccc_mean'][-1]))

print("Evaluation result!!")
print("Total loss value is {:.4f}".format(sum(val_result['ccc_mean']) / len(val_result['ccc_mean'])))

print('Data loader reset!')
Dataloader.reset()
_, num_total_sample_val = Dataloader.get_count()
print(num_total_sample_val)

    

