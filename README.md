🐠 Fish Species Classification using Deep Learning

This project builds a multi-class fish classifier using Convolutional Neural Networks and Transfer Learning.
A simple Streamlit interface allows users to upload an image and instantly view the predicted fish category along with confidence scores.

🌟 What This Project Does

Accepts image uploads (jpg, jpeg, png)
Processes the image and predicts the fish species
Shows:
🎯 Predicted class
📊 Confidence percentage
📈 Bar chart showing probability for all classes

Uses the best-performing model selected after comparing multiple architectures

🔧 Workflow Overview
1. Preparing the Dataset
Images resized to 224×224
Normalized pixel values
Data augmentation applied for robustness:
Random rotations
  Zoom
  Horizontal flips
Dataset split into:
Training | Validation | Testing

2. Building & Training Models

implemented: A basic CNN model

Transfer learning models:

VGG16
ResNet50
MobileNet
InceptionV3
EfficientNetB0

Each model was:

Trained on the fish dataset
Fine-tuned
Evaluated on the test split
➡ VGG16 Fine-Tuned achieved the highest accuracy and was selected for deployment.

3. Model Evaluation

For every model you checked:
Accuracy on test data
Confusion matrix
Classification report
Training/validation curves
Comparison table for all models

This allowed selecting the most reliable model.

4. Streamlit Deployment

A lightweight Streamlit app was developed where users can:
Upload an image
Preview the uploaded image
View the predicted category
View a probability chart for all classes

Run the app using:

streamlit run app.py

📂 Project Structure
app.py                  → Streamlit interface
fish_best_model.h5      → Best trained model (VGG16 Fine-Tuned)
class_labels.json       → List of fish categories
model_comparison.csv    → Accuracy results of all models
training_notebook.ipynb → Entire model training workflow

🛠️ Tools & Libraries

Python
TensorFlow / Keras
NumPy / Pandas
Matplotlib / Seaborn / Plotly
Streamlit

✅ Conclusion

This project successfully demonstrates how deep learning can accurately classify multiple fish species using both custom CNNs and transfer learning models.
By evaluating several architectures, the VGG16 Fine-Tuned model proved to be the most reliable for real-world prediction.
The Streamlit app provides an easy and interactive way for users to upload images and instantly view classification results.
