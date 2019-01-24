import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import numpy as np
import json
import os
import random
from PIL import Image
from basic_utils import load_checkpoint, load_cat_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default=None)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=True)
    return parser.parse_args()

def process_image(image):
    new_size = [0, 0]

    if image.size[0] > image.size[1]:
        new_size = [image.size[0], 256]
    else:
        new_size = [256, image.size[1]]
    
    image.thumbnail(new_size, Image.ANTIALIAS)
    width, height = image.size  

    left = (256 - 224)/2
    top = (256 - 224)/2
    right = (256 + 224)/2
    bottom = (256 + 224)/2

    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    image = image/255.
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    image = np.transpose(image, (2, 0, 1))
    
    return image

def predict(image_path, model, topk, gpu):
    model.eval()
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model = model.cuda()
    else:
        model = model.cpu()
        
    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
    
    if gpu and cuda:
        inputs = Variable(tensor.float().cuda())
    else:       
        inputs = Variable(tensor)
        
    inputs = inputs.unsqueeze(0)
    output = model.forward(inputs)
    
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])  
        
    return probabilities.numpy()[0], mapped_classes

def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    if args.filepath == None:
        img_num = random.randint(1, 102)
        image = random.choice(os.listdir('./flowers/test/' + str(img_num) + '/'))
        img_path = './flowers/test/' + str(img_num) + '/' + image
        prob, classes = predict(img_path, model, int(args.top_k), gpu)
        print('Image selected: ' + str(cat_to_name[str(img_num)]))
    else:
        img_path = args.filepath
        prob, classes = predict(img_path, model, int(args.top_k), gpu)
        print('File selected: ' + img_path)
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])

if __name__ == "__main__":
    main()