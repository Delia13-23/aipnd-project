import torch
from torchvision import models, transforms
from PIL import Image
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class of an image")
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()
    image = process_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p[0].tolist(), top_class[0].tolist()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint_path)
    probs, classes = predict(args.image_path, model, args.top_k, device)
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(cls)] for cls in classes]
    for i in range(args.top_k):
        print(f"Prediction {i+1}: {classes[i]} with a probability of {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()
