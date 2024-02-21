import argparse
import json
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F

def main():
    args = parse_args()

    device = torch.device("cuda" if args.GPU == 'GPU' else "cpu")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(f"{args.data_dir}/{x}", data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
                   for x in ['train', 'valid', 'test']}

    model, _ = load_model(args.arch, args.hidden_units, args.GPU)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lrn if args.lrn else 0.001)

    train_model(model, criterion, optimizer, dataloaders, device, args.epochs)

    save_checkpoint(model, args.save_dir, args.arch, image_datasets['train'].class_to_idx)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for image classifier")
    parser.add_argument('data_dir', type=str, help='Data directory containing training, validation, and test sets')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='alexnet', help='Model architecture: vgg13 or alexnet')
    parser.add_argument('--lrn', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=2048, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=7, help='Number of epochs to train for')
    parser.add_argument('--GPU', type=str, help='Use GPU for training if available')
    return parser.parse_args()

def load_model(arch='alexnet', hidden_units=2048, use_gpu=False):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = 25088
    else:
        model = models.alexnet(pretrained=True)
        input_size = 9216

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(input_size, 4096),
                               nn.ReLU(),
                               nn.Dropout(0.3),
                               nn.Linear(4096, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.3),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    return model, arch

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=7):
    model.to(device)
    steps = 0
    print_every = 40

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = validate_model(model, dataloaders['valid', criterion)
                print(f"Epoch {epoch+1}/{num_epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()

def validate_model(model, dataloader, criterion):
    model.eval()
    accuracy = 0
    valid_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valid_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

def save_checkpoint(model, save_dir, architecture, class_to_idx):
    checkpoint = {
        'architecture': architecture,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }
    filename = save_dir + '/checkpoint.pth' if save_dir else 'checkpoint.pth'
    torch.save(checkpoint, filename)

if __name__ == "__main__":
    main()
