import os
import uuid
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from torchvision import transforms
from model import HybridCNNViT

# ---------- Setup ----------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNNViT().to(device)
model.load_state_dict(torch.load("model_checkpoint_epoch10.pth", map_location=device))
model.eval()

class_labels = ['Normal', 'Pneumonia']

# ---------- CLAHE ----------
class CLAHETransform:
    def __call__(self, img):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_img = clahe.apply(img_cv)
        return Image.fromarray(cl_img).convert('RGB')

transform = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def is_xray_image(pil_img):
    img_np = np.array(pil_img)

    # If the image is grayscale or close to grayscale
    if len(img_np.shape) == 2:
        return True  # Grayscale image (likely X-ray)

    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        diff_rg = np.mean(np.abs(r - g))
        diff_gb = np.mean(np.abs(g - b))
        diff_rb = np.mean(np.abs(r - b))

        color_diff = diff_rg + diff_gb + diff_rb
        if color_diff < 10:  # Low color variance implies grayscale-like
            return True

    return False  # Color image with high variance – likely not an X-ray


# ---------- Grad-CAM ----------
def generate_gradcam(model, image_tensor, class_idx, output_path):
    image_tensor = image_tensor.to(device)
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.cnn_features[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    grads = gradients[0].detach().cpu()
    acts = activations[0].detach().cpu()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()
    cam = cam.numpy()

    orig_img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    orig_img = np.clip(orig_img * [0.229, 0.224, 0.225] + 
                       [0.485, 0.456, 0.406], 0, 1)

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(orig_img * 255), 0.5, heatmap, 0.5, 0)

    gradcam_filename = f"{uuid.uuid4().hex}_gradcam.jpg"
    gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
    cv2.imwrite(gradcam_path, overlay)

    forward_handle.remove()
    backward_handle.remove()
    return gradcam_path

# ---------- ViT Attention ----------
def extract_vit_attention(model, img_tensor, save_path='static/vit_attention.png'):
    vit = model.vit
    with torch.no_grad():
        outputs = vit(pixel_values=img_tensor.unsqueeze(0), output_attentions=True)
        attn = outputs.attentions[-1] if outputs.attentions else None

    if attn is None:
        print("❌ ViT attention not available.")
        return None

    attn_map = attn[0, :, 0, 1:]  # (heads, cls -> patch)
    attn_map = attn_map.mean(0).reshape(14, 14).cpu().numpy()
    attn_map = cv2.resize(attn_map, (224, 224))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
              np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.4 * heatmap + 0.6 * img_np

    plt.imsave(save_path, overlay)
    return save_path

# ---------- Routes ----------
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/diagnose')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file).convert('RGB')

    # X-ray validation
    if not is_xray_image(img):
        return jsonify({'error': 'Uploaded image is not recognized as a chest X-ray'}), 400

    img_tensor = transform(img).to(device)

    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
        pred_class = pred_class.item()
        confidence = confidence.item()
        prediction = class_labels[pred_class] if pred_class < len(class_labels) else "Unknown"
        conf_score = f"{(100*confidence):.4f}"

    gradcam_path = generate_gradcam(model, img_tensor.unsqueeze(0), pred_class, "static/gradcam.png")
    attention_path = extract_vit_attention(model, img_tensor, "static/vit_attention.png")

    return jsonify({
        'label': prediction,
        'confidence': conf_score,
        'gradcam_url': gradcam_path,
        'attention_url': attention_path
    })

if __name__ == '__main__':
    app.run(debug=True)
