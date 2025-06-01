# 🌿 Kernel Ridge Regression — Nonlinear Regression from Scratch

This repository contains a minimal yet powerful implementation of **Kernel Ridge Regression (KRR)** — a method that combines the **kernel trick** with **ridge regression** to handle nonlinear data with elegance and mathematical rigor.

Developed using pure **Python** and **NumPy**, with support for data visualization through **matplotlib**, this project is ideal for understanding the fundamentals of nonlinear regression without relying on high-level libraries like scikit-learn.

---

## 📁 Project Structure

```

├── datasets/
│   └── df\_regresion\_nolineal\_3.csv       # Sample dataset with nonlinear patterns
│
├── notebooks/
│   └── ...                               # Jupyter notebooks for experimentation
│
├── src/
│   └── ...                               # Core implementation of Kernel Ridge Regression
│
├── .gitignore
├── README.md

````

---

## 🧠 Technologies & Libraries

- **Python 3.x**
- **NumPy** — for efficient numerical operations
- **Matplotlib** — for insightful data visualizations
- Core Python tools and data structures

---

## 🚀 What is Kernel Ridge Regression?

**Kernel Ridge Regression** solves the problem of nonlinear regression by:
- Mapping input data into a high-dimensional space using **kernels**
- Applying **Ridge Regression** (L2 regularization) in that space

This allows for modeling complex, nonlinear relationships without explicitly computing the feature transformation — thanks to the **kernel trick**.

---

## 💡 Example: How to Use

Here's a minimal working example of using the implemented model:

```python
import numpy as np
import matplotlib.pyplot as plt
from src.kernel_ridge import KernelRidge

# Load your dataset
data = np.genfromtxt('datasets/df_regresion_nolineal_3.csv', delimiter=',', skip_header=1)
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

# Fit Kernel Ridge Regression
model = KernelRidge(kernel='rbf', alpha=0.1, gamma=15)
model.fit(X, y)

# Predict
X_pred = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_pred = model.predict(X_pred)

# Visualize
plt.scatter(X, y, color='navy', label='Data')
plt.plot(X_pred, y_pred, color='crimson', label='Kernel Ridge Prediction')
plt.title('Kernel Ridge Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
````

---

## 🧪 Features

* ✅ Implementation of Kernel Ridge with support for **RBF**, **polynomial**, and **linear** kernels
* ✅ Works on 1D or multidimensional data
* ✅ Easy to visualize and interpret
* ✅ Designed for educational clarity and extensibility

---

## 🔧 Installation & Setup

```bash
git clone https://github.com/your-username/kernel-ridge-regression.git
cd kernel-ridge-regression
pip install -r requirements.txt
```

---

## 📌 To-Do

* [ ] Add scikit-learn benchmark comparisons
* [ ] Include more datasets
* [ ] Extend to classification

---

## 📚 References

* Hastie, Tibshirani, and Friedman — *The Elements of Statistical Learning*
* Kernel Methods documentation: [Wikipedia](https://en.wikipedia.org/wiki/Kernel_method)
* Ridge Regression and Regularization Theory

---

## 📝 License

This project is licensed under the MIT License — feel free to use, share, and contribute.

---

> *Crafted with precision, for learners who value the math behind the models.*

