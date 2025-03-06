# **Neural Network Implementation on UCI Adult Dataset**  

## **Overview**  
This project implements a **multi-layer perceptron (MLP)** to perform binary classification on the **Adult dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult). The dataset is commonly used to predict whether an individual's income exceeds $50K per year based on census data.  

## **Key Features**  
- **Preprocessing**: Data cleaning, feature selection, one-hot encoding, and standardization.  
- **Neural Network Architecture**:  
  - 3 layers: **Input → 64 → 32 → Output**  
  - Activation functions: **ReLU** for hidden layers, **sigmoid** for the output layer  
  - Binary Cross-Entropy Loss (**BCELoss**) and **Adam optimizer**  
- **Training**: Runs for **5000 epochs** with progress tracking every 200 epochs.  
- **Performance Evaluation**: Tracks loss and accuracy over epochs, with visualizations using Matplotlib.  
- **Inference**: Provides a function to make predictions on test samples.  

## **Current Challenge**  
The model's improvement slows down after **2000 epochs**, and I'm currently investigating why. Suggestions and insights are welcome!  

## **Installation & Usage**  
### **Dependencies**  
Ensure you have Python and the following libraries installed:  
```bash
pip install torch pandas numpy scikit-learn matplotlib ucimlrepo
```
### **Running the Model**  
Clone the repository and execute the script:  
```bash
python train.py
```
### **Example Prediction**  
The script includes a function to test the model on a sample instance:  
```python
predict_example(model, t_X_test, t_y_test, index=5)
```

## **Results & Visualizations**  
The model's accuracy and loss trends are plotted during training. Example visualization:  

![Loss and Accuracy Graph](https://github.com/user-attachments/assets/12324df7-0bd4-4cb2-b69c-2ad91dd352b5)

