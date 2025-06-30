import torch
from transformers import ViTImageProcessor, ViTModel
import matplotlib.pyplot as plt
from PIL import Image
import requests
import numpy as np

# 1. Cargar modelo y preprocesador
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name, output_attentions=True)

# 2. Cargar imagen de ejemplo
img_url = "https://images.unsplash.com/photo-1534081333815-ae5019106622"  # Puedes poner tu imagen aquí
image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# 3. Preprocesar imagen
inputs = processor(images=image, return_tensors="pt")
# 4. Pasar imagen por el modelo
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions  # Lista: (num_layers, batch, num_heads, num_tokens, num_tokens)
layer = -1  # última capa
head = 0    # primer head

attention = attentions[layer][0, head]  # [num_tokens, num_tokens]
cls_attn = attention[0, 1:]  # Atención del [CLS] a todos los patches
num_patches = int(len(cls_attn) ** 0.5)

head_len = attentions[layer].shape[1]  # Número de heads
layers = len(attentions)  # Todas las capas
print(f"Total de capas: {len(attentions)}, Total de heads: {head_len}")

for layer in range(layers):
    for head in range(head_len):
        # Visualizar mapa de atención
        cls_attn = attentions[layer][0, head][0, 1:]  # [CLS] token
        attn_map = cls_attn.reshape(num_patches, num_patches).cpu().numpy()
        attn_map_resized = np.array(Image.fromarray(attn_map).resize(image.size, resample=Image.BILINEAR))

        plt.figure(figsize=(6, 4))
        plt.imshow(image)
        plt.imshow(attn_map_resized, cmap='jet', alpha=0.5)
        plt.title(f"Mapa de atención - Capa {layer}, Head {head}")
        plt.axis("off")
        plt.savefig(f"attention_map_layer{layer}_head{head}.png", bbox_inches='tight')
        plt.close()
