import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from PIL import Image

import time
import argparse

from basic_utils import save_checkpoint, load_checkpoint
from workspace_utils import active_session



def parse_args():

    parser = argparse.ArgumentParser(description="Training Phase")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg19', choices=['vgg13', 'vgg19'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='6')
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()

with active_session():

    
    def train(model, criterion, optimizer, dataloaders, epochs, gpu):
        cuda = torch.cuda.is_available()
        if gpu and cuda:
            model.cuda()
        else:
            model.cpu()
        print_every = 5
        steps = 0

      
        for e in range(epochs):
            running_loss = 0
            for inputs, labels in dataloaders[0]:
                steps += 1

                if gpu and cuda:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:

                    model.eval()
                    validation_loss = 0
                    accuracy=0

                    for inputs2, labels2 in dataloaders[1]:
                        optimizer.zero_grad()

                        if gpu and cuda:
                            inputs2, labels2 = Variable(inputs2.cuda()), Variable(labels2.cuda())
                        else:
                            inputs2, labels2 = Variable(inputs2), Variable(labels2)
                        
                        with torch.no_grad():
                            outputs2 = model.forward(inputs2)
                            validation_loss = criterion(outputs2,labels2).item()
                            ps = torch.exp(outputs2)
                            equality = (labels2.data == ps.max(dim=1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                    validation_loss = validation_loss / len(dataloaders[1])
                    accuracy = accuracy /len(dataloaders[1])



                    print("Epoch: {}/{}... ".format(e+1, epochs),
                            "Training Loss: {:.4f}".format(running_loss/print_every),
                            "Validation Loss {:.4f}".format(validation_loss),
                             "Accuracy: {:.4f}".format(accuracy))


                    running_loss = 0
                    model.train()
            

            
def main():
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    validataion_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 
    image_datasets = [ImageFolder(train_dir, transform=train_transforms),
                      ImageFolder(valid_dir, transform=validataion_transforms),
                      ImageFolder(test_dir, transform=test_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    

    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, int(args.hidden_units))),
                                  ('relu', nn.ReLU()),
                                  ('fc3', nn.Linear(int(args.hidden_units), 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "vgg19":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, int(args.hidden_units))),
                                  ('relu', nn.ReLU()),
                                  ('fc3', nn.Linear(int(args.hidden_units), 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=float(args.learning_rate))
   
    epochs = int(args.epochs)
    
    gpu = args.gpu
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = image_datasets[0].class_to_idx
    save_checkpoint(model, optimizer, args, classifier)


if __name__ == "__main__":
    main()
