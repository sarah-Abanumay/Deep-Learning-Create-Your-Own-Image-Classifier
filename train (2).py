import torch
from torch import nn
from torch import optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import numpy as np
import torchvision 
import time
import os
import copy
import json
def parse_args():
    parser=argparse.ArgumentParser(description="Trains a dataset of image and saves model to checkpoint")
    parser.add_argument('--data_dir',default='flowers',type=str,help='set the path')
    parser.add_argument('--arch',default='resent',type=str,help=' the model architect')
    parser.add_argument('--lr',default=0.001,type=float,help='learning rate')
    parser.add_argument('--hidden_units',default=non,nargs='+',type=int,help='list of int , size of hidden')
    parser.add_argument('--epochs',default=5,type=int,help='no. of training epoch')
    parser.add_argument('--gpu',default=False,type=bool,help='set the gpu mode')
    parser.add_argument('--saved_model',default='my_point.pth',type=str,help=' path save to checkpoit')
    parser = argparse.ArgumentParser(description='Flower Classifcation trainer')

    return args

    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                save_network(model, epoch)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    save_network(model, 'best')
    return model
with open(file_name,'r') as json_file:
                input_conf = json.load(json_file)
                config.update(input_conf)
                LoadConfigJsonFile.setup_config(namespace, config)
def main():
    args = parse_args()
    data_dir=args.data_dir
    gpu=args.data_gpu
    arch=args.arch
    lr=args.lr
    hidden_units=args.hidden_units
    epochs=args.epochs
    saved_model=args.saved_model
    print('='*10+'params'+'='*10)
    print('data_dir:   {}'.format(data_dir))
    print('arch {}'.format(arch))
    print(' lr  {}'.format(lr))
    print('  hidden_units  {}'.format(hidden_units))
    print('     epochs  {}'.format(epochs))
    print('     saved_model  {}'.format(saved_model))
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
 # Set the GPU
    if gpu_mode and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Current device: {}'.format(device))
    
    
    # Freeze parameters so we don't backprop through them
    dataloaders, image_datasets = cook_data(args)

    #  Load a pre-trained 
    if args.arch == 'vgg': 
        model = models.vgg16(pretrained=True)
    elif args.arch == 'densenet':
        model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[0].in_features
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, 512)),
                              ('relu', nn.ReLU()),
                              ('drpot', nn.Dropout(p=0.5)),
                              ('hidden', nn.Linear(512, args.hidden_units)),                       
                              ('fc2', nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1)),
                              ]))

    # Reserve for final layer: ('output', nn.LogSoftmax(dim=1))
        
    model.classifier = classifier
    
        pass
        
    print('='*10 + ' Architecture ' + '='*10)
    print('The classifier architecture:')
    print(classifier)
    
 if args.gpu:
        if use_gpu:
            model = model.cuda()
            print ("Using GPU: "+ str(use_gpu))
        else:
            print("Using CPU since GPU is not available/configured")


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model = train_model(args, model, criterion, optimizer, exp_lr_scheduler,num_epochs=args.epochs)

    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.epochs = args.epochs
    {'input_size': [3, 224, 224],
                  'batch_size': dataloaders['train'].batch_size,
                  'output_size': 102,
                  'arch': args.arch,
                  'state_dict': model.state_dict(),
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
    torch.save(checkpoint, args.saved_model)

    
if __name__ == '__main__':
    main()




  