import numpy as np
import torch

from torch import nn,optim
from collections import OrderedDict
from torchvision import datasets,transforms,models

#定义神经网络
'''
class NetWork(nn.Module):
    def __init__(self,input_size,output_size,hidden_units,drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size,hidden_units[0])])
        layer_sizes = zip(hidden_units[:-1],hidden_units[1:])

        #input_size:输入尺寸
        #output_size:输出尺寸
        #hidden_units:定义隐藏层
        #drop_p:丢弃概率

        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layer_sizes])

    def return_hidden_layers(self):
        return self.hidden_layers
'''
#定义命令行参数
def get_input_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir',help='train_data_dir',type=str)
    parser.add_argument('--valid_data_dir',help='valid_data_dir',type=str)
    parser.add_argument('--test_data_dir',help='test_data_dir',type=str)
    parser.add_argument('--arch',help='arch',type=str)
    parser.add_argument('--learning_rate',help='learning rate',type=float)
    parser.add_argument('--input_size',help='input_size',type=int)
    parser.add_argument('--output_size',help='output_size',type=int)
    parser.add_argument('--hidden_units1',help='hidden',type=int)
    parser.add_argument('--hidden_units2',help='hidden',type=int)
    parser.add_argument('--epochs',help='epochs',type=int)
    parser.add_argument('--gpu',help='gpu',type=str)
    parser.add_argument('--models',help='vgg11 or vgg16',type=str)
    parser.add_argument('--filepath',help='checkpoint save file',type=str)
    parser.add_argument('--modelpath',help='save model file',type=str)

    return parser.parse_args()

#读取数据
def load_data(train_dir,valid_dir,test_dir):
    #处理训练数据
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
    #处理验证数据
    valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
    #处理测试数据
    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])

    #用ImageFolder读取图片数据
    train_dataset = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir,transform=test_transforms)

    #批量获取数据
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset,batch_size=64,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)

    return trainloader,validloader,testloader,train_dataset

#定义训练模型
def Definition_mode(lr,choose_model,input_size,hidden_units1,hidden_units2,output_size):
    if choose_model == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif choose_model == 'vgg16':
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1',nn.Linear(input_size,hidden_units1)),
                              ('relu1',nn.ReLU()),
                              ('dropout1',nn.Dropout(0.5)),
                              ('fc2',nn.Linear(hidden_units1,hidden_units2)),
                              ('relu2',nn.ReLU()),
                              ('dropout2',nn.Dropout(0.5)),
                              ('fc3',nn.Linear(hidden_units2,output_size)),
                              ('output',nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(),lr=lr)
    hidden_layers = [hidden_units1,hidden_units2]
    return model,criterion,optimizer,classifier,hidden_layers

#定义训练过程
def train(model, trainloader, validloader, criterion, optimizer, epochs,
          device,log_interval = 20):
    steps = 0
    running_loss = 0
    model.train()
    if device == 'cuda':
        if torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()
    else:
        model.cpu()

    for e in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % log_interval == 0:
                #在eval模式下验证模型
                model.eval()
                with torch.no_grad():
                    valid_loss, valid_accuracy = validate(model, validloader, criterion,device)
                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / log_interval),
                          "Valid Loss: {:.3f}.. ".format(valid_loss / len(validloader)),
                          "Valid Accuracy: {:.3f}".format(valid_accuracy / len(validloader)))
                    running_loss = 0
                    running_accu = 0
                    #重新开始训练
                    model.train()
#定义验证过程
def validate(model, validloader, criterion, device):
    loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return loss, accuracy

#存储训练好的模型
def save_model(arch,model,input_size,output_size,hidden_layers,learning_rate,classifier,optimizer,train_dataset,filepath,modelpath):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'arch': arch,
                  'class_to_idx': model.class_to_idx,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_units': hidden_layers,
                  'epochs': 3,
                  'log_interval': 32,
                  'learning_rate': learning_rate,
                  'classifier': classifier,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict()}

    torch.save(checkpoint,filepath)
    torch.save(model,modelpath)

#主函数
def main():
    in_args = get_input_args()
    train_data_dir = in_args.train_data_dir
    valid_data_dir = in_args.valid_data_dir
    test_data_dir = in_args.test_data_dir
    arch = in_args.arch
    lr = in_args.learning_rate
    hidden_units1 = in_args.hidden_units1
    hidden_units2 = in_args.hidden_units2
    epochs = in_args.epochs
    gpu = in_args.gpu
    choose_model = in_args.models
    input_size = in_args.input_size
    output_size = in_args.output_size
    filepath = in_args.filepath
    modelpath = in_args.modelpath

    trainloader,validloader,testloader,train_dataset = load_data(train_data_dir,valid_data_dir,test_data_dir)
    model,criterion,optimizer,classifier,hidden_layers = Definition_mode(lr,choose_model,input_size,hidden_units1,hidden_units2,output_size)
    train(model,trainloader,validloader,criterion,optimizer,epochs,gpu)
    save_model(arch,model,input_size,output_size,hidden_layers,lr,classifier,optimizer,train_dataset,filepath,modelpath)

if __name__ == "__main__":
    main()
