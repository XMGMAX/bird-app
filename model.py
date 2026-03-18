import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import json
import io
import os
from huggingface_hub import hf_hub_download

HF_REPO = os.getenv("HF_REPO", "xmgmax0/bird-classifier")
device  = torch.device("cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model():
    print("Downloading class names...")
    classes_path = hf_hub_download(repo_id=HF_REPO, filename="class_names.json")
    with open(classes_path) as f:
        class_names = json.load(f)

    print("Downloading model weights...")
    model_path = hf_hub_download(repo_id=HF_REPO, filename="bird_vit_b16_final.pth")

    print("Loading model...")
    model = timm.create_model("vit_base_patch16_224",
                               pretrained=False, num_classes=0)
    embed_dim = model.embed_dim
    model.head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(p=0.3),
        nn.Linear(embed_dim, 512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, len(class_names))
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model ready! ({len(class_names)} classes)")
    return model, class_names

def predict(model, class_names, image_bytes, top_k=5):
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = val_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    top_probs, top_idxs = probs.topk(top_k)
    return [
        {
            "rank":       i + 1,
            "label":      class_names[idx].replace("_", " "),
            "raw_label":  class_names[idx],
            "confidence": round(float(prob) * 100, 2)
        }
        for i, (prob, idx) in enumerate(
            zip(top_probs.numpy(), top_idxs.numpy())
        )
    ]