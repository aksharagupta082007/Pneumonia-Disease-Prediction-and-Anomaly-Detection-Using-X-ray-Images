# Pneumonia-Disease-Prediction-and-Anomaly-Detection-Using-X-ray-Images
Hybrid Pneumonia Detection Project (DenseNet-120 + ViT-B16 + Age Feature)
=======================================================================

1. Project Overview

Goal: Develop a hybrid deep learning model that detects Pneumonia using both chest X-ray images and patient age.
- Model Type: Hybrid CNN + Transformer model.
- Architecture: DenseNet-121 (CNN) for local image features + ViT-B16 (Vision Transformer) for global image features.

2. Dataset Details

   We collect dataset of X-ray images from kaggle and other sources. 

3. Data Preprocessing

* **CLAHETransform()** – Enhances **local contrast** in images to highlight subtle features, making patterns more visible. Applied to **both Train & Test** datasets.
* **Resize(224×224)** – Standardizes all images to a **fixed size** so they can be fed into models like CNNs and ViTs.  Applied to **both Train & Test** datasets.
* **RandomHorizontalFlip** – Performs **data augmentation** by flipping images horizontally, improving model generalization.  Applied to **Train only**.
* **RandomRotation(10°)** – Adds **data augmentation** by rotating images up to ±10°, simulating variations in image capture.  Applied to **Train only**.
* **ToTensor()** – Converts images from **PIL/numpy** format to PyTorch tensors and scales pixel values to **$0, 1$**.  Applied to **both Train & Test** datasets.
* **Normalize(mean, std)** – Standardizes image channels using dataset-specific **mean** and **standard deviation** to stabilize training.  Applied to **both Train & Test** datasets.

4. Model Architecture

* **Architecture Purpose** – This model is a **Hybrid CNN + ViT** architecture that merges the strengths of convolutional neural networks (CNNs) and vision transformers (ViTs) for more accurate image classification, especially in X-ray analysis.

* **CNN Backbone (DenseNet-121)** – Uses the **`features`** part of a pre-trained DenseNet-121 from torchvision to extract **local spatial and texture-based features** from the input image.

  * DenseNet’s dense connectivity pattern ensures efficient feature reuse and reduces parameters.
  * The output feature maps are passed through **Adaptive Average Pooling** to produce a fixed-size feature vector `(batch, 1024)` regardless of input size.

* **Vision Transformer (ViT)** – Loads **`vit-base-patch16-224`** from Hugging Face’s `transformers` library, pretrained on large-scale datasets.

  * Processes the image as **16×16 patches** and uses transformer layers to capture **global relationships** across the image.
  * All ViT parameters are **frozen** (`requires_grad=False`) to retain pretrained knowledge and reduce training cost.
  * The **\[CLS] token** from the ViT output (`last_hidden_state[:, 0, :]`) is used as a **global representation** of the entire image, producing a `(batch, 768)` vector.

* **Feature Fusion** – The CNN output vector `(batch, 1024)` and the ViT output vector `(batch, 768)` are concatenated along the feature dimension to create a **joint feature vector** `(batch, 1792)`.

* **Classification Head** –

  * First **Linear layer** reduces 1792 features to 512.
  * **ReLU activation** introduces non-linearity.
  * **Dropout(0.3)** helps prevent overfitting.
  * Final **Linear layer** maps 512 features to **2 output classes** (binary classification).

* **Forward Pass Flow** –

  1. Input image passes through DenseNet to extract local features.
  2. Parallelly, the same image is fed to ViT to extract global context features.
  3. CNN and ViT outputs are concatenated into a single vector.
  4. This combined representation is passed through the fully connected layers to produce the final prediction.

* **Strength of the Approach** –

  * **CNN branch** → excels at **local feature extraction** (edges, textures, small patterns).
  * **ViT branch** → excels at **global context understanding** (relationships between distant image parts).
  * Combining them provides a **richer, more comprehensive representation**, improving classification accuracy for complex medical imaging tasks.



5. Training Pipeline
* **Model Initialization**

  * Creates an instance of `HybridCNNViT`.
  * Moves the model to the chosen device (**GPU or CPU**) for computation.

* **Loss Function (Criterion)**

  * Uses **CrossEntropyLoss**, ideal for classification tasks.
  * Measures the difference between predicted probabilities and actual class labels.

* **Optimizer**

  * Uses **Adam** optimizer for faster and more stable convergence.
  * **Learning rate (lr)** = `1e-4` (controls step size for weight updates).
  * **Weight decay** = `1e-5` (L2 regularization to prevent overfitting).

* **Learning Rate Scheduler**

  * **ReduceLROnPlateau**: Automatically lowers the learning rate when validation loss stops improving.
  * **Mode** = `'min'`: Watches a metric that should decrease (e.g., validation loss).
  * **Factor** = `0.5`: Halves the learning rate when triggered.
  * **Patience** = `2`: Waits 2 epochs without improvement before reducing the learning rate.
  * **Verbose** = `True`: Displays messages when learning rate changes.



