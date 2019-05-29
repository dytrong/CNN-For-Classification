import torch
from make_data import load_data
import torch.nn as nn
import torch.nn.functional as F 

device=torch.device('cuda:0') #调用gpu:0
torch.manual_seed(1)

class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False,**kwargs):
    model=AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load('./model/alexnet_model_para.pkl'))
    return model

model = AlexNet().cuda(device)
#print(model)
train_load,train_data=load_data('./data/')

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9,dampening=0,weight_decay=0.0001)

loss_func = torch.nn.CrossEntropyLoss()

train_epoch=100

def train(epoch):  
    model.train()
 
    train_loss=0.

    train_corrects=0.

    for batch_idx, (data, target) in enumerate(train_load):  

        data, target = data.cuda(device),target.cuda(device) 

        #将所有梯度值重新设置为0
        optimizer.zero_grad()  

        #训练模型，输出结果  
        output = model(data) 

        #torch.max(input, dim, max=None, max_indices=None)
        _,preds=torch.max(output.data,1) ###第二个参数1代表dim的意思,也就是取每一行的最大值,第三个参数表示最大值,第四个就是最大值的索引
        
        #在数据集上预测loss
        loss = loss_func(output, target)

        #反向传播调整参数pytorch直接可以用loss  
        #计算总的loss和预测正确的个数
        train_loss=train_loss+loss.data

        train_corrects=train_corrects+torch.sum(preds==target.data)  
        
        loss.backward()
  
        #SGD刷新进步  
        optimizer.step()

    #计算总的loss和预测正确的个数
    ####计算平均loss和accuracy
    epoch_loss=train_loss/len(train_load.dataset)
    
    epoch_acc=train_corrects/len(train_load.dataset)
    
    torch.save(model.state_dict(),'./model/alexnet_model_para.pkl')
    
    print('The {} epoch result: Average loss: {:.6f},Accuracy:{}/{} ({:.2f}%)\n'.
            format(epoch,epoch_loss,train_corrects,len(train_load.dataset),100.*float(train_corrects)/float(len(train_load.dataset))))

if __name__ == "__main__":
    for epoch in range(1, train_epoch+ 1):  
        train(epoch) 
