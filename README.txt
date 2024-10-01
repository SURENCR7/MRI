
# Brain MRI Metastasis Segmentation

## 1. Introduction

This project implements and compares two architectures for brain MRI metastasis segmentation: **Nested U-Net (U-Net++)** and **Attention U-Net**. The models are used to segment metastasis regions from brain MRI images, and the best-performing model is deployed in a Streamlit web application.

## 2. Architectures Overview

### 2.1 Nested U-Net (U-Net++)
The Nested U-Net, also known as U-Net++, is an extension of the U-Net architecture. It introduces **nested skip connections** between the encoder and decoder at different levels. This allows the model to capture finer details and contextual information during the segmentation process, which is crucial for detecting small and irregular metastasis regions.

#### Key Features:
- Multiple skip connections to enhance feature reuse.
- Allows better gradient flow, improving model convergence.
- Effective in segmenting small metastasis regions.

### 2.2 Attention U-Net
The Attention U-Net enhances the original U-Net by introducing **attention gates** in the decoder. These gates focus the model's attention on the relevant areas of the image, thus improving segmentation accuracy by filtering out irrelevant features.

#### Key Features:
- Focuses on the most important regions in an image (e.g., metastasis regions).
- Reduces unnecessary computation by suppressing irrelevant background features.
- Effective when metastasis regions are small and difficult to distinguish from surrounding tissues.

### Application to Metastasis Segmentation
Both architectures are well-suited for brain metastasis segmentation due to their ability to capture detailed, localized information in the MRI images. These models can handle the challenges of segmenting small and irregularly shaped metastases, which often vary across different MRI sequences.

## 3. Streamlit Web Application

The Streamlit UI provides an easy-to-use interface for users to upload MRI images and view the metastasis segmentation results. The UI allows users to visualize how the model detects metastases in real-time.

## 4. Video Demonstration
A video demonstration of the Streamlit UI in action, showing metastasis segmentation results, can be found in the following link:

[Demo Video Link](#) *(Insert video link after upload)*

The video shows how users can upload an MRI image, and the system will display the predicted metastasis segmentation.

## 5. Instructions for Setting Up and Running the Code

### 5.1 Requirements
- Python 3.7+
- TensorFlow / PyTorch
- Streamlit
- Pillow (for image handling)
- NumPy and Pandas
- scikit-image (for image processing)

### 5.2 Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. Access the application in your browser at `http://localhost:8501`.

### 5.3 Running Model Training and Evaluation
The model training scripts can be found in the `training/` folder. You can run the training process with the following command:
   ```bash
   python train_model.py --model {nested_unet/attention_unet}
   ```
The pre-trained models are saved in the `models/` folder.

## 6. Challenges in Brain Metastasis Segmentation

Brain metastasis segmentation presents several challenges:
- **Irregular Shape**: Metastases often have irregular shapes and can appear in various sizes, making them difficult to segment accurately.
- **Small Lesions**: The metastasis regions are sometimes very small, making it harder to distinguish from the surrounding tissues.
- **Multiple MRI Sequences**: Different MRI sequences (pre-contrast, post-contrast, and FLAIR) highlight different aspects of the brain tissues, and the model must learn to combine this information effectively.

### How the Implementations Address These Challenges:
- **Nested U-Net** captures multi-scale features using dense connections, which helps in segmenting metastases of different sizes.
- **Attention U-Net** focuses on the metastasis regions through attention gates, which improves segmentation accuracy, especially for small lesions.
- **Data Augmentation** techniques are applied to increase the variability in the training set, helping the models generalize better.

## 7. Conclusion

This project demonstrates how deep learning architectures like Nested U-Net and Attention U-Net can be applied to the challenging task of brain metastasis segmentation. The models have been evaluated and deployed in a user-friendly web application using Streamlit.

