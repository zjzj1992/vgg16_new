from PIL import Image
from torch import nn,optim
from torchvision import datasets,transforms,models

import json
import torch
import numpy as np

'''
class MyNetWork(nn.Module):
    def __init__(self,input_size,output_size,hidden_units,drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size,hidden_units[0])])
        layer_sizes = zip(hidden_units[:-1],hidden_units[1:])

        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layer_sizes])

    def return_hidden_layers(self):
        return self.hidden_layers
'''
#定义模型
def Definition_mode(lr,choose_model,hidden_units):
    if choose_model == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif choose_model == 'vgg16':
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.ModuleList()
    for i,r in enumerate(hidden_units):
        if i == len(hidden_units) - 1:
            classifier.append(r)
            classifier.append(nn.LogSoftmax(dim=1))
        else:
            classifier.append(r)
            classifier.append(nn.ReLU())
            classifier.append(nn.Dropout())

    model.classifier = classifier
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(),lr=lr)
    return model,criterion,optimizer,classifier

#定义命令行参数
def get_input_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data',help='image_data',type=str)
    #parser.add_argument('--checkpoint',help='checkpoint',type=str)
    parser.add_argument('--category_names',help='category_names',type=str)
    parser.add_argument('--topk',help='topk',type=int)
    parser.add_argument('--gpu',help='gpu',type=str)
    parser.add_argument('--modelpath',help='modelpath',type=str)

    return parser.parse_args()

#读取文件，文件用于映射到类别
def load_data(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

#读取保存好的模型
def load_model(modelpath):
    #checkpoint= torch.load(filepath)
    #model.classifier = checkpoint['classifier']
    #model.load_state_dict(checkpoint['model_state_dict'])
    #model.class_to_idx = checkpoint['class_to_idx']
    model = torch.load(modelpath)

    return model

#处理图像数据
def process_image(image):
    pil_img = Image.open(image)
    #调整图片大小
    if pil_img.size[0] > pil_img.size[1]:
        pil_img.thumbnail((256 + 1, 256))
    else:
        pil_img.thumbnail((256, 256 + 1))
    #中心裁剪
    left = (pil_img.width - 224) / 2
    bottom = (pil_img.height - 224) / 2
    right = left + 224
    top = bottom + 224
    pil_img = pil_img.crop((left, bottom, right, top))
    #数据标准化
    np_img = np.array(pil_img)
    np_img = np_img / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    np_img = np_img.transpose((2, 0, 1))
    np_img = torch.FloatTensor(np_img)

    return np_img

#图像类别预测
def predict(image_path, model, topk, gpu_mode):
    #获取处理后的图像
    img = process_image(image_path
    #根据设置选择使用CPU或GPU来处理图像
    if gpu_mode:
        if torch.cuda.is_available():
            model.cuda()
            img = img.cuda()
        else:
            model.cpu()
            img = img.cpu()
    #进入模型验证状态
    model.eval()
    #前向处理
    with torch.no_grad():
        outputs = model.forward(img.unsqueeze(0))
    #获取对应的预测类别和概率
    probs, class_indices = outputs.topk(topk)
    probs = probs.exp().cpu().numpy()[0]
    class_indices = class_indices.cpu().numpy()[0]
    #将得到的结果映射为对应的类别
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in class_indices]

    return probs, classes

#主函数
def main():
    in_args = get_input_args()
    input_data = in_args.input_data
    #checkpoint = in_args.checkpoint
    category_names = in_args.category_names
    topk = in_args.topk
    gpu = in_args.gpu
    modelpath = in_args.modelpath

    #mynetwork = MyNetWork(25088,102,[4096,1000,102])
    model = load_model(modelpath)
    cat_to_name = load_data(category_names)
    probs,classes = predict(input_data,model,topk,gpu)
    print("K corresponding probs: ",probs)
    print("K corresponding classes: ",classes)

if __name__ == "__main__":
    main()
