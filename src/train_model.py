import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import *
import os
import time


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device("cuda:0")
writer = SummaryWriter("../logs/vege15/{}".format(time.strftime("%Y%m%d_%H%M%S", time.localtime())))
train_data = torchvision.datasets.ImageFolder("../dataset/vegetable_classification15/train/",
                                              transform=transforms.Compose([
                                                  transforms.Resize((64, 64)), transforms.ToTensor()]))
validation_data = torchvision.datasets.ImageFolder("../dataset/vegetable_classification15/validation/",
                                              transform=transforms.Compose([
                                                  transforms.Resize((64, 64)), transforms.ToTensor()]))


train_data_size = len(train_data)
validation_data_size = len(validation_data)
print(train_data.class_to_idx)
print("The number of Train Data: {}".format(train_data_size))
print("The number of Validation Data: {}".format(validation_data_size))

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False)

vege15 = Vege15()
vege15 = torch.nn.DataParallel(vege15)
vege15 = vege15.cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

learning_rate = 1e-2
optimizer = torch.optim.SGD(vege15.parameters(), lr=learning_rate)

total_train_step = 0
total_valid_step = 0
epoch = 50
for i in range(epoch):
    since = time.time()
    print("-------{} Epoch/{} Epochs--------".format(i+1, epoch))
    vege15.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        writer.add_images("images",imgs,global_step = i)
        targets = targets.cuda()
        outputs = vege15(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 50 ==0:
            print("The Training Times: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    vege15.eval()
    total_valid_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in validation_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = vege15(imgs)
            loss = loss_fn(outputs, targets)
            total_valid_loss = total_valid_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("The accuracy on the all validation set: {}".format(total_accuracy/validation_data_size))
    total_valid_step = total_valid_step + 1
    writer.add_scalar("validation_loss", total_valid_loss, total_valid_step)
    writer.add_scalar("validation_accuracy", total_accuracy/validation_data_size, total_valid_step)
    if i == epoch-1:
        path = "./model/{}".format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        os.makedirs(path) 
        torch.save(vege15, path+"/vege15_{}.pth".format(i))
        print("Model Saved Successfully!")

writer.close()