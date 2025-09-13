import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

data_transforms = {
    'train': transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    'val': transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]),
}

train_ds = datasets.ImageFolder("data/train", transform=data_transforms['train'])
val_ds   = datasets.ImageFolder("data/val", transform=data_transforms['val'])
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 класса
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# тренируем пару эпох (скелет)
for epoch in range(3):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print("Epoch", epoch, "done")

# инференс на одном изображении
from PIL import Image
import torchvision.transforms.functional as TF
img = Image.open("test.jpg").convert("RGB")
input_t = TF.resize(img, (224,224))
input_t = TF.to_tensor(input_t).unsqueeze(0).to(device)
model.eval()
out = model(input_t)
pred = out.argmax(dim=1).item()
print("машина" if pred==train_ds.class_to_idx['car'] else "не машина")
