# 🌾 Rice Type Classification using PyTorch

This project presents an end-to-end **machine learning solution** for classifying rice grain types using **PyTorch**.  
The model is trained on numerical grain features to differentiate between two rice varieties, achieving a high level of accuracy and generalization.  

---

## 📘 Project Overview

The aim of this project is to build a **binary classification model** that identifies rice types based on their physical and geometric features.  
The dataset contains essential measurements of rice grains such as area, perimeter, eccentricity, and aspect ratio.  
By applying deep learning techniques, this model automates the process of identifying rice varieties efficiently.

The model is designed, trained, and tested in **PyTorch**, showcasing a practical application of neural networks on **structured tabular data**.

---

## 🎯 Objectives

- To preprocess and normalize rice grain data for deep learning.  
- To design a feedforward neural network capable of binary classification.  
- To analyze model performance using training, validation, and testing metrics.  
- To visualize learning progress with accuracy and loss plots.  
- To perform inference and predict the rice type based on new data.

---

## 🧠 Methodology

1. **Data Preprocessing**  
   The dataset is cleaned by removing missing values and irrelevant columns.  
   All numerical features are normalized to ensure uniform scaling, improving convergence during training.

2. **Dataset Splitting**  
   The dataset is divided into three subsets:  
   - **70% Training Data** — for learning model parameters.  
   - **15% Validation Data** — for tuning and preventing overfitting.  
   - **15% Testing Data** — for final performance evaluation.

3. **Model Design**  
   A **fully connected feedforward neural network** is used.  
   It consists of an input layer corresponding to the number of features, one hidden layer with ReLU activation, and an output layer with a Sigmoid activation function for binary classification.

4. **Training Phase**  
   The model is trained using the **Binary Cross-Entropy Loss (BCELoss)** and optimized with the **Adam Optimizer**.  
   Each epoch updates the weights through backpropagation, gradually minimizing the loss function.

5. **Validation Phase**  
   During training, the model’s performance on the validation dataset is continuously monitored to prevent overfitting and ensure generalization.

6. **Testing and Evaluation**  
   The trained model is tested on unseen data to evaluate its accuracy, precision, and reliability.  
   The final model achieves an impressive **testing accuracy of around 98.9%**.

7. **Visualization**  
   Training and validation loss/accuracy are plotted over epochs, providing visual insight into the model’s learning behavior and convergence.

---

## 🧩 Dataset Description

The dataset used for this project contains geometric and morphological features of rice grains.  
Each feature helps describe the physical characteristics that distinguish one type of rice from another.

| Feature | Description |
|----------|-------------|
| Area | Total pixel area of the rice grain |
| MajorAxisLength | Length of the major axis of the fitted ellipse |
| MinorAxisLength | Length of the minor axis of the fitted ellipse |
| Eccentricity | Measure of elongation or deviation from circular shape |
| ConvexArea | Area of the convex hull surrounding the grain |
| EquivDiameter | Diameter of a circle with equivalent area |
| Extent | Ratio of grain area to bounding box area |
| Perimeter | Boundary length of the grain |
| Roundness | Degree of shape compactness |
| AspectRation | Ratio of major axis to minor axis length |
| Class | Output label (0 or 1) representing rice type |

---

## ⚙️ Model Architecture Summary

- **Input Layer:** 10 neurons (corresponding to the 10 input features)  
- **Hidden Layer:** 10 neurons with ReLU activation  
- **Output Layer:** 1 neuron with Sigmoid activation  
- **Loss Function:** Binary Cross-Entropy Loss (BCELoss)  
- **Optimizer:** Adam (learning rate = 0.001)  
- **Batch Size:** 8  
- **Epochs:** 10  

This configuration provides a simple yet powerful baseline model for structured data classification using PyTorch.

---

## 📊 Performance Metrics

After the training process, the model achieved the following results:

| Metric | Accuracy (%) |
|---------|---------------|
| Training Accuracy | 98.5 |
| Validation Accuracy | 98.5 |
| Testing Accuracy | **98.97** |

These results indicate that the model generalizes well to unseen data with minimal overfitting.

---

## 📈 Results Interpretation

- The **training and validation loss curves** decrease steadily, indicating effective learning.  
- The **accuracy curves** plateau near 99%, confirming model stability.  
- The model demonstrates consistent predictions across all phases (train, validation, test).  

Overall, this project shows how a properly normalized dataset and a simple neural network can yield high-accuracy results for binary classification tasks.

---

## 🔍 Inference Process

The trained model can classify a new rice grain by inputting its geometric features (such as area, perimeter, and aspect ratio).  
After normalization, these values are passed to the model, which outputs a probability between 0 and 1.  
The value is then rounded to the nearest class label to predict the rice type.

---

## 💡 Future Enhancements

- Introduce **Dropout** or **Batch Normalization** layers for further regularization.  
- Implement **hyperparameter tuning** to optimize performance.  
- Experiment with **deep architectures** like MLPs or CNNs for improved representation.  
- Integrate the model into a **Flask or Streamlit** web application for real-time prediction.  
- Extend the dataset to **multi-class classification** for more rice varieties.

---

## 🧾 Key Learnings

- Deep learning models can effectively classify structured data, not just images.  
- Proper normalization and balanced dataset splitting are essential for high accuracy.  
- Even simple architectures can outperform complex models if the data quality is high.  
- Visualization of accuracy and loss is crucial for understanding model behavior.

---

## 🏁 Conclusion

This project demonstrates that **PyTorch** is not limited to image or text-based deep learning — it can efficiently handle **tabular data** classification as well.  
The model achieved nearly **99% accuracy**, highlighting the effectiveness of neural networks for structured data when combined with thoughtful preprocessing and validation.  
The approach provides a foundation for developing advanced models for **agricultural analytics**, **grain quality assessment**, and **food industry automation**.

---

## 📜 References
- PyTorch Official Documentation  
- Scikit-learn API Reference  
- Rice Classification Dataset (Kaggle)  
- Goodfellow, I., Bengio, Y., & Courville, A. — *Deep Learning* (MIT Press, 2016)

---

## ✨ Author
**Developed by:** Sagar Shrivas  
**Tools Used:** Python, PyTorch, Scikit-learn, Pandas, Matplotlib  
**Year:** 2025  
