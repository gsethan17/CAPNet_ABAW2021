# ABAW2021
This is the repository containing the solution for ICCV-2021 ABAW Competition


# Demo test
with test set of Aff-wild2

1. Download the cropped image of test set for demo using following URL.  
[Download link](https://drive.google.com/file/d/1Uu9DlWQRFoRBHVfY3IhKrt5zmbBU11CK/view?usp=sharing)

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
    ```python generate_dataset.py --location 231 --type sequence --mode val```  
    ```python eval.py --location 231 --type val```  
    ```python utils.py --location 231 --mode merge --path1 [Path of prediction] --path2 [Path of FER-Tuned]```   
    ```python utils.py --location 231 --mode compare --path1 [Path of prediction] --path2 [Path of groundtruth]```