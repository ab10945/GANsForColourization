# GANsForColourization

#### Dependencies to be added. - 
1. torchmetrics. -  pip install torchmetrics or conda install -c conda-forge torchmetrics
2. GPUtil - pip install GPUtil

#### To execute UNet GAN with resnet18 backbone, Run the following command - 

1. to run with a pretrained resnet18 - 
python main.py --net resnet-18 --dataset <path_to_dataset>  --cpt_dir Results/checkpoints --vis_dir Results/visualizations --op_dir Results/outputs --BATCH_SIZE 16 --NUM_EPOCHS 2 --LEARNING_RATE 1e-4 --path_net_g <path_to_state_dict_of_resnet_model>

2. to pretrain resnet 18 from scratch - 
python main.py --net resnet-18 --dataset <path_to_dataset>  --cpt_dir Results/checkpoints --vis_dir Results/visualizations --op_dir Results/outputs --BATCH_SIZE 16 --NUM_EPOCHS 2 --LEARNING_RATE 1e-4

3. to load and retrain/continue to train an existing GAN model
python main.py --net resnet-18 --dataset <path_to_dataset>  --cpt_dir Results/checkpoints --vis_dir Results/visualizations --op_dir Results/outputs --BATCH_SIZE 16 --NUM_EPOCHS 2 --LEARNING_RATE 1e-4 --Retrain True --pPrev <path_To_GAN_state_dict>

#### to run infer.py

python infer.py --pathGAN <path_to_model> --pathImg < path_to_test_image> --pathOP <path_to_output_directory>


