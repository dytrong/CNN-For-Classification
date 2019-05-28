import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
def load_data(data_path):
    dataset=torchvision.datasets.ImageFolder(data_path,
                                             transform=transforms.Compose([
                                                transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])
                                            )
    ###dataset.classes -用一个list保存类名
    ###dataset.class_to_idx -类名对应的索引
    ###dataset.imgs -保存（img_path,class)tuple的list

    loader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,num_workers=15,drop_last=False) ####数据加载器
    #for data,target in loader:   ####data是图像tensor,target是图像的标签
        #print(target)
    return loader,dataset

