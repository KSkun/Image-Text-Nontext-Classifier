import gc

import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

# detect cuda support
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

root_dir = 'E:\\University\\21-22S1\\software_project\\TextDis benchmark\\'  # dataset root


# dataset & loader
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


# image transformation
t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    gc.collect()
    # load model
    model = torch.load('20211224-3-epoch26.pth', map_location=torch.device(device))
    model = model.to(device)
    print(model)

    # load test data
    test_data = MyDataset(txt=root_dir + 'testList.txt', transform=t)
    test_loader = DataLoader(dataset=test_data, batch_size=8)

    # set loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # evaluate the model
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    eval_tp, eval_tn, eval_fp, eval_fn = 0., 0., 0., 0.
    with torch.no_grad():
        for imgs, labels in test_loader:
            # get a batch of images
            imgs = Variable(imgs.to(device))
            labels = Variable(labels.to(device))

            # predict the images
            out = model(imgs)

            # calculate loss & accuracy
            loss = loss_func(out, labels)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == labels).sum()
            eval_acc += num_correct.item()

            # calculate f-measure
            eval_tp += (labels * pred).sum()
            eval_tn += ((1 - labels) * (1 - pred)).sum()
            eval_fp += ((1 - labels) * pred).sum()
            eval_fn += (labels * (1 - pred)).sum()

    # calculate f-measure
    epsilon = 1e-7
    eval_precision = eval_tp / (eval_tp + eval_fp + epsilon)
    eval_recall = eval_tp / (eval_tp + eval_fn + epsilon)
    eval_f1 = 2 * (eval_precision * eval_recall) / (eval_precision + eval_recall + epsilon)

    print('Test Loss: {:.6f}, Acc: {:.6f}, Precision: {:.6f}, Recall: {:.6f}, F1: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data)), eval_precision, eval_recall, eval_f1))
