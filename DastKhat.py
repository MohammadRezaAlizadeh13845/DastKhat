import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import easyocr

TARGET_HEIGHT = 32
TARGET_WIDTH = 128
DATASET_PATH = "DATASET"
OUTPUT_DIR = "PREPROCESSED"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# مرحله 1: پیش‌پردازش تصاویر
def preprocess_and_save(image_path, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype(np.float32) / 255.0

    h, w = img.shape
    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    padded_img = np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
    top = (TARGET_HEIGHT - new_h) // 2
    left = (TARGET_WIDTH - new_w) // 2
    padded_img[top:top + new_h, left:left + new_w] = resized_img

    cv2.imwrite(save_path, (padded_img * 255).astype(np.uint8))
    tensor_img = torch.tensor(padded_img).unsqueeze(0)
    return tensor_img

results = []
for filename in sorted(os.listdir(DATASET_PATH)):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(DATASET_PATH, filename)
        save_path = os.path.join(OUTPUT_DIR, filename)
        tensor = preprocess_and_save(img_path, save_path)
        results.append((filename, tensor))

reader = easyocr.Reader(['fa'], gpu=torch.cuda.is_available())
labels = {}
for filename, _ in results:
    image_path = os.path.join(OUTPUT_DIR, filename)
    result = reader.readtext(image_path, detail=0)
    labels[filename] = result[0] if result else ""

all_text = "".join(labels.values())
char_set = sorted(set(all_text))
char2idx = {c: i+1 for i, c in enumerate(char_set)}  
idx2char = {i: c for c, i in char2idx.items()}

def encode_text(text):
    return [char2idx[c] for c in text if c in char2idx]

def decode_tensor(tensor):
    return ''.join([idx2char.get(i, '') for i in tensor])

class CRNNDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, image = self.data[idx]
        label = labels[filename]
        label_encoded = torch.tensor(encode_text(label), dtype=torch.long)
        return image, label_encoded, filename

class CRNN(nn.Module):
    def __init__(self, img_h, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * (img_h // 4), 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes + 1)  

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  
        x = x.reshape(b, w, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

dataset = CRNNDataset(results)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)

model = CRNN(TARGET_HEIGHT, len(char2idx)).to(DEVICE)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  
    model.train()
    total_loss = 0
    for batch in dataloader:
        images, targets, _ = zip(*batch)
        images = torch.stack(images).to(DEVICE)
        targets = torch.cat(targets).to(DEVICE)

        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        input_lengths = torch.full((len(images),), images.shape[3] // 4, dtype=torch.long)

        outputs = model(images)
        outputs = outputs.log_softmax(2)
        outputs = outputs.permute(1, 0, 2)

        optimizer.zero_grad()
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


model.eval()
with open("predictions.txt", "w", encoding="utf-8") as f:
    for filename, tensor in results:
        image = tensor.unsqueeze(0).to(DEVICE)
        output = model(image)
        output = output.log_softmax(2)
        _, pred = output.max(2)
        pred = pred.squeeze().detach().cpu().numpy()

        prev = -1
        decoded = []
        for p in pred:
            if p != prev and p != 0:
                decoded.append(p)
            prev = p
        pred_text = decode_tensor(decoded)
        f.write(f"{filename}\t{pred_text}\n")
