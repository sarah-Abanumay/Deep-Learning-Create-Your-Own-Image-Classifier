 import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import torchvision 
from torchvision import datasets, transforms, models
from PIL import *
import os
import json
def parse_args():
    parser=argparse.ArgumentParser(description="Load model from checkpoint and make prediction on image")
    parser.add_argument('--img_path',type=str,help='set the image path')
    parser.add_argument('--topk',default=5,type=int,help='set no. of topk')
    parser.add_argument('--gpu',default=False,type=bool,help='set the gpu mode')
    parser.add_argument('--checkpoint',default='my_point.pth',type=str,help='set path of checkpoit')
    args=parser.parse.args()
    return args
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
        
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088, 500)),
                          ('relu',nn.ReLU()),
                          ('fc2',nn.Linear(500, 102)),
                          ('output',nn.LogSoftmax(dim=1)),
                          ]))    
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
        with open(file_name,'r') as json_file:
                input_conf = json.load(json_file)
                config.update(input_conf)
                LoadConfigJsonFile.setup_config(namespace, config)
        except Exception as ex:
            raise argparse.ArgumentTypeError(
                "file:{0} is not a valid json file. Read error: {1}".format(file_name, ex))
# function that loads a checkpoint and rebuilds the model
model.class_to_idx=checkpoint['class_to_idx']
from PIL import *

def process_image(image):
    proc_img = Image.open(image)
   
    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
        
    model_img = prepoceess_img(proc_img)
    return model_img
def predict(image_path, model, top_k=5):
 
    model.to("GPU")
    model.eval();

    # Convert numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")
#Result log scale
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all 
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers
def main():
    args = parse_args()
    img_path=args.img_path
    gpu=args.gpu
    top_k=args.top_k
    checkpoint=args.checkpointfilepath
    print('='*10+'Params'+'='*10)
    print('Image path:       {}'.format(img_path))
    print('Load model from:  {}'.format(ckpt_path))
    print('GPU mode:         {}'.format(gpu_mode))
    print('TopK:             {}'.format(top_k))
    
    # Load the model
  model, __, __ = load_checkpoint(ckpt_path)
    class_names = model.class_names
if gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    print('Current device: {}'.format(device))
    model.to(device)
    
 # Predict
    print('='*10+'Predict'+'='*10)
    probs, classes = predict(img_path, model, device, topk)
    flower_names = [cat_to_name[class_names[e]] for e in classes]
    for prob, flower_name in zip(probs, flower_names):
        print('{:20}: {:.4f}'.format(flower_name, prob))
    
    
if __name__ == '__main__':
    main()    
    
