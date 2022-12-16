# Image Colourization using Conditional Generative Adversarial Neural Networks.

## Desciption -

## Repository and code structure
Files included - 
```
1. main.py - holds the main execution flow
2. dataset.py - helper functions to create dataloaders from the coco dataset
3. infer.py - inference engine
4. loss.py - helper functions and classes for GANLoss implementation
5. models.py - helper functions and classes to generate the different GAN models
6. utils.py - utility functions to log loss and visualization of training/testing outputs.
```
Results\visualizations folder holds 2 images, one showcases the original dataset. And one image showing the sample inference output.

## Dependencies to be added. - 
1.  Torch (version '1.9.1+cu102') - 
   ```
 conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch 
    or 
 pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
2. Fastai (version 2.4) 
``` 
pip install fastai==2.4 
```
3. GPUtil (version 1.4.0)- 
```
pip install GPUtil 
```
4. sklearn (version 1.2.0)
``` 
pip install -U scikit-learn 
```
5. skimage (version 0.19.3)
```
pip install -U scikit-image
```
## Example commands for execution - 

### For training -
To run the baseline model (Unet with BCEwithLogitsLoss)- 
```
python main.py -- dataset <your_path_to_dataset> --cpt_dir <path_to_store_training_checkpoints> --vis_dir <path_to_store_visualizations> --op_dir <path_to_store_output_logs>  --BATCH_SIZE 16 --NUM_EPOCHS 20 --LEARNING_RATE 1e-4
```
To run the Unet model with a resnet18 backbone (Resnet-18 - Unet with BCEwithLogitsLoss) - 
```
python main.py --net resnet-18  -- dataset <your_path_to_dataset> --cpt_dir <path_to_store_training_checkpoints> --vis_dir <path_to_store_visualizations> --op_dir <path_to_store_output_logs>  --BATCH_SIZE 16 --NUM_EPOCHS 20 --LEARNING_RATE 1e-4
```
To run Unet model with a resnet18 backbone with PSNR loss - 
```
python main.py --net resnet-18 --dataset <your_path_to_dataset> --GAN_Mode PSNR --cpt_dir <path_to_store_training_checkpoints> --vis_dir <path_to_store_visualizations> --op_dir <path_to_store_output_logs> --BATCH_SIZE 8 --NUM_EPOCHS 20 --LEARNING_RATE 1e-4
```
To load models with weights from previous checkpoints
```
python main.py --net resnet-50 --dataset <your_path_to_dataset> --GAN_Mode PSNR --cpt_dir <path_to_store_training_checkpoints> --vis_dir <path_to_store_visualizations> --op_dir <path_to_store_output_logs> --BATCH_SIZE 16 --NUM_EPOCHS 20 --LEARNING_RATE 1e-4 --pPrev <path_to_GAN_weights> --path_net_g  <path_to_Resnet-18_weights>
```
Using the command line arguments,
```--GAN_Mode``` controls the GAN criterion, can hold the following values - If \"vanilla\", criterion is BCEwithLogits, if \“lsgan\”, criterion is \"MSE\", if \”PSNR\”, criterion is \"PSNR\". Default is vanilla.
``` --net``` controls the Generator network architecture, it should be either of baseline, resnet-18, vgg-16, inception. Default is baseline.
 
### For running the script for inference


python infer.py --pathGAN <path_to_model> --pathImg < path_to_test_image> --pathOP <path_to_output_directory>


