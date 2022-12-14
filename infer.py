import argparse
import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
import matplotlib.image

from models import MainModel
from utils import lab_to_rgb
import time

parser = argparse.ArgumentParser(description="Image Colourization with GANs.")
parser.add_argument('--pathGAN', default="", type=str, help="path to the final model") ## Path to the model.
parser.add_argument('--pathImg', default="", type=str, help="path to the test image") ## Path to the test image.
parser.add_argument('--pathOP', default="", type=str, help="path to the output directory") ## Path to the Output directory.
args = parser.parse_args()

if __name__ == '__main__':
    model = MainModel()
    # You first need to download the final_model_weights.pt file from my drive
    # using the command: gdown --id 1lR6DcS4m5InSbZ5y59zkH2mHt_4RQ2KV
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(
            args.pathGAN,
            map_location=device
        )
    )
    path = args.pathImg
    img = PIL.Image.open(path)
    img = img.resize((256, 256))
    # to make it between -1 and 1
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]

    plt.imshow(colorized)

    matplotlib.image.imsave(args.pathOP+f"/inference_{time.time()}.png", colorized)
    #plt.savefig(args.pathOP+f"inference_{time.time()}.png")
