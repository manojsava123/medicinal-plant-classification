Medicinal Plant Identification using Deep Learning & Machine Learning

📌 Project Overview

Medicinal plants play a vital role in healthcare, but traditional identification by experts is slow, subjective, and error-prone. This project presents an automated system that accurately identifies medicinal plants using deep learning and machine learning techniques.
A cascaded model is developed where ResNet50 extracts deep visual features from plant images and Particle Swarm Optimization (PSO) optimizes these features. The optimized features are classified using multiple machine learning algorithms to achieve high accuracy.

🎯 Objectives

Automate medicinal plant identification
Reduce human error and manual effort
Improve classification accuracy using optimization techniques
Compare multiple machine learning classifiers

🧠 Methodology

Image Input
Plant leaf images are taken from the Kaggle Indian Medicinal Plant Image Dataset.

Feature Extraction

ResNet50, a deep CNN model, extracts important visual features from the images.

Feature Optimization

Particle Swarm Optimization (PSO) selects the best features to improve classification performance.

Classification

The optimized features are classified using:
Support Vector Machine (SVM)
Random Forest
Decision Tree
K-Nearest Neighbors (KNN)
XGBoost
Naïve Bayes
Logistic Regression
Model Tuning
SVM hyperparameters are tuned to achieve the best performance.

📊 Results

SVM (Tuned) achieved the highest accuracy of 99.75%
Logistic Regression achieved 99.18%
The proposed cascaded architecture proved highly reliable and accurate

🛠 Technologies Used

Python
TensorFlow / Keras
Scikit-learn
OpenCV
NumPy, Pandas
Kaggle Dataset

📂 Dataset

Dataset used: Indian Medicinal Plant Image Dataset (Kaggle)
Contains images of 7 medicinal plant species.

🚀 Advantages

Fully automated feature extraction
High classification accuracy
Scalable for larger plant datasets
Reduces human error
Useful for botanical, agricultural, and pharmaceutical applications

🔮 Future Scope

Extend to more plant species
Mobile app for real-time identification
Integration with herbal medicine databases
Deployment using cloud or IoT

👨‍💻 Author

Manoj Sava

Final Year Engineering Project
