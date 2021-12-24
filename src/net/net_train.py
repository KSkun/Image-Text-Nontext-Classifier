# Get cpu or gpu device for training.
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from google.colab import drive
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# root_dir = '/content/drive/MyDrive/TextDis benchmark/'
root_dir = 'E:\\University\\21-22S1\\software_project\\TextDis benchmark\\'


# from https://www.cnblogs.com/denny402/p/7520063.html
def default_loader(path):
    return Image.open(root_dir + path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[1], int(words[2])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    # drive.mount('/content/drive')

    gc.collect()
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=2, bias=True)  # 2 classes
    model = model.to(device)
    print(model)

    train_data = MyDataset(txt=root_dir + 'trainList.txt', transform=t)
    test_data = MyDataset(txt=root_dir + 'testList.txt', transform=t)
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(30):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        train_tp, train_tn, train_fp, train_fn = 0., 0., 0., 0.
        model.train()
        for imgs, labels in train_loader:
            imgs = Variable(imgs.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == labels).sum()
            train_acc += train_correct.item()

            train_tp += (labels * pred).sum()
            train_tn += ((1 - labels) * (1 - pred)).sum()
            train_fp += ((1 - labels) * pred).sum()
            train_fn += (labels * (1 - pred)).sum()

        epsilon = 1e-7
        train_precision = train_tp / (train_tp + train_fp + epsilon)
        train_recall = train_tp / (train_tp + train_fn + epsilon)
        train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall + epsilon)
        print('Train Loss: {:.6f}, Acc: {:.6f}, Recall: {:.6f}, F1: {:.6f}'.format(train_loss / (len(
            train_data)), train_acc / (len(train_data)), train_recall, train_f1))

        model_path = '/content/drive/MyDrive/trained/20211224-3-epoch%d.pth' % (epoch + 1)
        torch.save(model, model_path)
        print('Model saved at %s' % model_path)

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        eval_tp, eval_tn, eval_fp, eval_fn = 0., 0., 0., 0.
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = Variable(imgs.to(device))
                labels = Variable(labels.to(device))

                out = model(imgs)

                loss = loss_func(out, labels)
                eval_loss += loss.item()
                pred = torch.max(out, 1)[1]
                num_correct = (pred == labels).sum()
                eval_acc += num_correct.item()

                eval_tp += (labels * pred).sum()
                eval_tn += ((1 - labels) * (1 - pred)).sum()
                eval_fp += ((1 - labels) * pred).sum()
                eval_fn += (labels * (1 - pred)).sum()

        epsilon = 1e-7
        eval_precision = eval_tp / (eval_tp + eval_fp + epsilon)
        eval_recall = eval_tp / (eval_tp + eval_fn + epsilon)
        eval_f1 = 2 * (eval_precision * eval_recall) / (eval_precision + eval_recall + epsilon)
        print('Test Loss: {:.6f}, Acc: {:.6f}, Recall: {:.6f}, F1: {:.6f}'.format(eval_loss / (len(
            test_data)), eval_acc / (len(test_data)), eval_recall, eval_f1))
