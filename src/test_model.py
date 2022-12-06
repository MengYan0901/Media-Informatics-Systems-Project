import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn

device = torch.device("cuda:0")
test_data = torchvision.datasets.ImageFolder("../dataset/vegetable_classification15/test/",
                                              transform=transforms.Compose([
                                                  transforms.Resize((64, 64)), transforms.ToTensor()]))
test_data_size = len(test_data)
print(test_data.class_to_idx)

print("The number of Test Data: {}".format(test_data_size))
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
    
model = torch.load("./model/vege15_49.pth")
print(model)
model = model.to(device)

model.eval()
total_test_loss = 0
total_accuracy = 0
with torch.no_grad():
    for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
print("The accuracy on the all test set: {}".format(total_accuracy/test_data_size))
