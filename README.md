# Image Colourization using Conditional Generative Adversarial Neural Networks.

### Desciption -

### Repository and code structure

#### Dependencies to be added. - 
1.  Torch (version '1.9.1+cu102') - 
    conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch 
    or 
    pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
2. Fastai (version 2.4) 
    pip install fastai==2.4
3. GPUtil - 
    pip install GPUtil

#### To execute UNet GAN with resnet18 backbone, Run the following command - 

1. to run with a pretrained resnet18 - 
python main.py --net resnet-18 --dataset <path_to_dataset>  --cpt_dir Results/checkpoints --vis_dir Results/visualizations --op_dir Results/outputs --BATCH_SIZE 16 --NUM_EPOCHS 2 --LEARNING_RATE 1e-4 --path_net_g <path_to_state_dict_of_resnet_model>

2. to pretrain resnet 18 from scratch - 
python main.py --net resnet-18 --dataset <path_to_dataset>  --cpt_dir Results/checkpoints --vis_dir Results/visualizations --op_dir Results/outputs --BATCH_SIZE 16 --NUM_EPOCHS 2 --LEARNING_RATE 1e-4

3. to load and retrain/continue to train an existing GAN model
python main.py --net resnet-18 --dataset <path_to_dataset>  --cpt_dir Results/checkpoints --vis_dir Results/visualizations --op_dir Results/outputs --BATCH_SIZE 16 --NUM_EPOCHS 2 --LEARNING_RATE 1e-4 --Retrain True --pPrev <path_To_GAN_state_dict>

#### to run infer.py

python infer.py --pathGAN <path_to_model> --pathImg < path_to_test_image> --pathOP <path_to_output_directory>


