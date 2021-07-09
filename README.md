# CAPNet & FER-Tuned
CAPNet is a causal affect prediction network, which is the deep learning model to predict affect state with a sequence of past facial images.  
FER-Tuned is a facial expression recognition model, which is deep learning model to recognize affect state with a single current facial image.  
Please refer to the paper as following for the details.  
[Causal affect prediction model using a facial image sequence](https://arxiv.org/abs/2107.03886)

# The Affective Behavior Analysis in-the-wild 2021
We participated in the 2nd Affective Behavior Analysis in-the-wild (ABAW2) Competition with the above two models.  
[ABAW2](https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/)  

We provide a demo test to yield the results submitted to ABAW2 as following.

## Pre-trained weights Download
Please download the pre-trained weights for CAPNet & FER-Tuned.  
[Download link for weights](https://www.dropbox.com/sh/u6w2yx8p36eggfc/AACxgZ72RHA6zIdywN83mBxea?dl=0)  
placement
```
CAPNet_ABAW2021
|-- weights
`-- |-- CAPNet_1
`-- |-- CAPNet_2
`-- |-- CAPNet_3
`-- |-- FER-Tuned
```

## Demo test
with test set of Aff-Wild2
1. Please download the Aff-Wild2 dataset and write the download location in `PATH_DATA` of `config.ini`.  

2. Download the cropped image of test set for demo using following URL.  
[Download link](https://drive.google.com/file/d/1Uu9DlWQRFoRBHVfY3IhKrt5zmbBU11CK/view?usp=sharing)  
Extract to `DATA_PATH`. `test_images_for_demo` folder should be in `DATA_PATH`.  
Please refer to the `README` among the extracted files for the details.

    placement
    `videos` is provided by [Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
    ```
    [PATH_DATA]
    |-- va_test_set.csv
    |-- videos
    |-- test_images_for_demo
    `-- |-- cropped
        `-- |-- 2-30-640x360
            `-- |-- 00001.jpg
            `-- |-- 00002.jpg
            `-- |-- ...
        `-- |-- 2-30-640x360
            `-- |-- 00001.jpg
            `-- |-- 00002.jpg
            `-- |-- ...
        `-- |-- ...
    `-- |-- post_processing_pickles
        `-- |-- keep_past_value.pickle
        `-- |-- values_both_0_and_keep.pickle
        `-- |-- values_to_0.pickle
        `-- |-- values_to_m5.pickle
    ```

3. To demo test  
    1. FER-Tuned test (current single image)  
    Please set the `MODEL_KEY` on `config.ini` before the test.  
        ```
        MODEL_KEY = FER-Tuned
        ```
        To generate prediction results :  
        ```
        python eval.py --type val
        ```  
        To calculate the concordance correlation coefficient (CCC) value :  
        ```
        python utils.py --mode compare --path1 [Path of prediction]
        ```    

    2. CAPNet test (past sequence image)  
    Please set the `MODEL_KEY` and `WINDOW_SIZE` on `config.ini` before the test.  
        ```
        WINDOW_SIZE = 3
        MODEL_KEY = CAPNet
        ```
        To generate sequential dataset for test :
        ```
        python generate_dataset.py --mode sequence --type val
        ```
        To generate prediction results :
        ```
        python eval.py --type val
        ```
        To merge with FER-Tuned prediction results to fill in the blank that occur when there are insufficient sequential images :
        ```
        python utils.py --mode merge --path1 [Path of prediction] --path2 [Path of FER-Tuned]
        ```   
        To calculate the concordance correlation coefficient (CCC) value :
        ```
        python utils.py --mode compare --path1 [Path of prediction]
        ```
    
4. Results  
You can find the results of prediction on the position as follwoing :
    ```
    CAPNet_ABAW2021
    |-- results
    `-- |-- VA-Set
        `-- |-- Test-Set
            `-- |-- CAPNet_1
                `-- |-- merge
            `-- |-- CAPNet_2
                `-- |-- merge
            `-- |-- CAPNet_3
                `-- |-- merge
            `-- |-- FER-Tuned
    ```