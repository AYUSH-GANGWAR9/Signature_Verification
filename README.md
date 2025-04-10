# Offline Signature Verification

This project implements an offline signature verification system using a hybrid approach combining Convolutional Neural Networks (CNN) and Histogram of Oriented Gradients (HOG) for feature extraction. The system uses RandomForest for feature selection and implements multiple classifiers including LSTM, SVM, and KNN for signature verification.
<br><hr>
## 🎯 Project Overview <br><br>

The system is designed to verify the authenticity of handwritten signatures by comparing them against known genuine signatures. It uses a hybrid approach that combines: <br>

- Deep learning features (CNN) <br>
- Traditional computer vision features (HOG) <br>
- Machine learning classifiers (LSTM, SVM, KNN) <br>
<br>
<hr>
## 🚀 Requirements <br> <br>

- Python 3.8+ <br>
- PyTorch <br>
- scikit-learn <br>
- OpenCV <br>
- NumPy <br>
- Other dependencies listed in `requirements.txt` <br>

To install dependencies:
<br>
```bash <br>
pip install -r requirements.txt
```
<br>
<hr>
## 💻 Usage<br>

### 1. Data Preparation <br> <br>

- Place your signature datasets in the `data/` directory <br>
- Supported datasets: CEDAR and UTSig <br>
<br>
### 2. Training <br>
<br>
```bash<br>
python main.py  <br>
```
<br>
### 3. Testing <br>

```bash <br>
python test_signature.py --image_path path/to/signature.png <br>
```
<br><hr>
## 🏗️ Model Architecture <br><br>

The system uses a hybrid architecture: <br>

### 1. Feature Extraction <br>

- **CNN**: Deep learning features from signature images <br>
- **HOG**: Traditional computer vision features <br>
   
### 2. Feature Selection <br><br>

- RandomForest-based feature selection <br>

### 3. Classification <br><br>

- **LSTM**: For sequence-based verification <br>
- **SVM**: For binary classification <br>
- **KNN**: For similarity-based verification <br>

## 📊 Performance <br><br>

The system is evaluated using: <br><br>

- False Acceptance Rate (FAR) <br>
- False Rejection Rate (FRR) <br>
- Equal Error Rate (EER) <br>
- Accuracy 
<br><hr> 
## 📚 References <br><br>

This implementation is based on the paper: <br>

"A Hybrid Method of Feature Extraction for Signatures Verification Using CNN and HOG: a Multi-Classification Approach" 
<br><hr>
## 📄 License <br>

MIT License 
<br><hr>
## 👥 Contributors <br>

- Ayush gangwar
<br><hr>
## 📫 Contact <br>
 
For questions or feedback, please [open an issue](https://github.com/AYUSH-GANGWAR9/Signature-verification/issues) on GitHub. <br>

#   S i g n a t u r e _ V e r i f i c a t i o n 
 
 #   S i g n a t u r e _ V e r i f i c a t i o n 
 
 
