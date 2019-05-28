from network import alexnet
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import time
import os
img_to_tensor = transforms.ToTensor()

def make_model():
    model=alexnet(pretrained=True)
    print(model)
    model.cuda()
    return model

def inference(model,imgpath):
    model.eval()
    img=Image.open(imgpath)  
    img=img.resize((224,224))  
    tensor=img_to_tensor(img)  
      
    tensor=tensor.resize_(1,3,224,224)  
    tensor=tensor.cuda()#将数据发送到GPU，数据和模型在同一个设备上运行
    result=model(tensor)
    result_npy=result.data.cpu().numpy()#将结果传到CPU，并转换为numpy格式  
    max_index=np.argmax(result_npy[0])  
      
    return max_index

if __name__=='__main__':
    start=time.time()
    model=make_model()
    end=time.time()
    print('模型初始化耗时:'+str(end-start))
    count=0
    start=time.time()
    sub_path=os.listdir("./data/00")
    for i in range(len(sub_path)):
        img_path='./data/00/'+sub_path[i]
        index=inference(model,img_path)
        print(index)
        if index==0:
            count=count+1
    end=time.time()
    print('测试共耗时:'+str(end-start))
    print('正确数量为:'+str(count))
    print("总的检测数量为:"+str(len(sub_path)))
    print("正确率为:"+str(float(count)/len(sub_path)))
