# ABAW2021
This is the repository containing the solution for ICCV-2021 ABAW Competition

# Pre-trained weights Download
[Download link for weights](https://www.dropbox.com/sh/u6w2yx8p36eggfc/AACxgZ72RHA6zIdywN83mBxea?dl=0)  
placement
```
|-- weights
`-- |-- CAPNet_1
`-- |-- CAPNet_2
`-- |-- CAPNet_3
`-- |-- FER-Tuned
```

# Demo test
with test set of Aff-wild2

1. Download the cropped image of test set for demo using following URL.  
[Download link](https://drive.google.com/file/d/1Uu9DlWQRFoRBHVfY3IhKrt5zmbBU11CK/view?usp=sharing)

placement
`videos` is provided by [Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
```
PATH_DATA
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

2. Extract to `DATA_PATH` of `config.ini`.  
`test_images_for_demo` folder should be in `DATA_PATH`.  
Please refer to the `README` among the extracted files for the details.

3.
    (1) FER-Tuned test (current single image)  
    Please set the `config.ini` before the test.  
    ```python eval.py --location 231 --type val```  
    ```python utils.py --location 231 --mode compare --path1 [Path of prediction] --path2 [Path of groundtruth]```  

    (2) CAPNet test (past sequence image)  
    Please set the `config.ini` before the test.  
    ```python generate_dataset.py --location 231 --mode sequence --type val```  
    ```python eval.py --location 231 --type val```  
    ```python utils.py --location 231 --mode merge --path1 [Path of prediction] --path2 [Path of FER-Tuned]```   
    ```python utils.py --location 231 --mode compare --path1 [Path of prediction] --path2 [Path of groundtruth]```
    
4. Results 
```
|-- results
`-- |-- VA-Set
`-- |-- |-- Test-Set
`-- |-- |-- |-- CAPNet_1
`-- |-- |-- |-- CAPNet_2
`-- |-- |-- |-- CAPNet_3
`-- |-- |-- |-- FER-Tuned
```