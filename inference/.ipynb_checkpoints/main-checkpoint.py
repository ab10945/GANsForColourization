## Update: 8th Jan, 2022
## this file is supposed to give you a general idea on how to
## use the pre-trained model for colorizing B&W images. This
## file still needs development.
import matplotlib.image
import flask 
import PIL
import torch
import glob
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from models import MainModel
from utils import lab_to_rgb
import numpy as np


def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

app = flask.Flask(__name__)


@app.route("/",methods=['GET', 'POST'])
def predict():
    if flask.request.method == "POST":
        img = (flask.request.form.get("image"))
        print(img)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
        model = MainModel(net_G=net_G)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(
            torch.load(
                "./200ep_resnet-18Final_gan.pt",
                map_location=device
            )
        )
        name = img[:-4]
        path = "./test_set/"
        # paths = glob.glob(path + "/*.jpg")
        image = path + img
        # print(image_)
        img = PIL.Image.open(image)
        img = img.resize((256, 256))
        original = np.asarray(img)
        img = transforms.ToTensor()(img)[:1] * 2. - 1.
        model.eval()
        with torch.no_grad():
            preds = model.net_G(img.unsqueeze(0).to(device))
        colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
        output = np.zeros((256, 512, 3))
        c = np.asarray(colorized)
        original = original[:, :,  np.newaxis]
        output[:, :256, :] = original / 255
        output[:, 256:, :] = c 
        matplotlib.image.imsave("./results/"+f"{name}.png", output)
        filename = "./results/"+f"{name}.png"
        return flask.send_file(filename, mimetype='./results/"+f"{name}.png')
    else:
        return flask.render_template("form.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005)

