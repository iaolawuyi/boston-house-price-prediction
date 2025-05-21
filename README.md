# Boston House Price Prediction 🏠📈

This project is a regression model trained to predict house prices using the Boston Housing dataset from Keras. It explores a deep learning approach with k-fold cross-validation to mitigate overfitting on this relatively small dataset.

## 📁 Project Structure

```
boston-price-prediction/
│
├── src/
│   ├── train.py          # Training script with k-fold cross-validation
│   ├── model.py          # Model definition
│   ├── normalize.py      # Custom normalization utility
│   ├── plot.py           # Visualization of training metrics
│
├── saved_models/         # (Optional) Folder for storing trained models
├── graphs/               # Output graphs (loss, MAE over epochs)
├── README.md             # Project documentation
├── .gitignore
```

## 🚀 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/boston-price-prediction.git
cd boston-price-prediction
```

2. **(Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run training**

```bash
python src/train.py
```

Graphs and results will be printed and optionally saved to the `graphs/` directory.

## 📊 Techniques Used

- Keras Sequential Model
- ReLU activations
- Mean Squared Error (MSE) loss
- Mean Absolute Error (MAE) as evaluation metric
- K-Fold Cross Validation (k=4)
- Manual normalization of data

## 📌 Notes

- Because the dataset is small (404 training samples), k-fold cross-validation is used to ensure robust evaluation.
- Consider adding dropout or regularization if overfitting is observed.

## 🧠 Future Improvements

- Add dropout layers for regularization
- Compare with `sklearn` models (e.g. Linear Regression, Random Forest)
- Automate hyperparameter tuning

---

*This project is part of my deep learning portfolio showcasing applications of neural networks on classical machine learning problems.*