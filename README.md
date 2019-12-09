# TGS Salt Identification Challenge

This project represents my contribution to the Kaggle's TGS Salt Identification Challenge (https://www.kaggle.com/c/tgs-salt-identification-challenge) held in the second part of the 2018.  
To be able to use the code please follow listed instructions:  

1) Download the competition data from the following page:  
   https://www.kaggle.com/c/tgs-salt-identification-challenge/data  

2) Copy data into "data" subfolder:  
   ```
   data/depths.csv  
   data/sample_submission.csv  
   data/train.csv  
   data/test/images/*.png  
   data/train/images/*.png  
   data/train/masks/*.png  
   ```

3) Execute prepare_data.py to create:  
   ```
   train_x_fixed.npy  
   train_y_fixed.npy  
   test_x.npy  
   ```
   
4) Optionally open model.py and set desired model_type:  
   ```
   model_type = 'my_res_unet'  
   # model_type = 'unet'  
   # model_type = 'fpn'  
   # model_type = 'linknet'  
   # model_type = 'pspnet'  
   ```

5) Execute train.py to train 5 models. In case of a problem, adjust batch size:  
   ```
   batch_size = 32  
   ```
   
6) Execute prepare_submission.py to create submission CSV file (5 trained models are expected to be found in tmp folder). In case of a problem, adjust batch size:  
   ```
   batch_size = 64  
   ```

7) Execute visualize_outputs.py to visualize trained models outputs (5 trained models are expected to be found in tmp folder). In case of a problem, adjust batch size:  
   ```
   batch_size = 64  
   ```
