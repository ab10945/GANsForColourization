# GANsForColourization

To execute UNet GAN with resnet18 backbone, Run the following command - 

python main.py --net resnet-18 --dataset /scratch/ab10945/Final_Project/data/coco/images/train2017  --cpt_dir Results/checkpoints --vis_dir Results/visualizations --op_dir Results/outputs --BATCH_SIZE 16 --NUM_EPOCHS 2 --LEARNING_RATE 1e-4
