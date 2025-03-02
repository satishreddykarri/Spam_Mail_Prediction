# Spam Mail Prediction

## Overview
This project is a machine learning model that predicts whether an email is spam or not using a logistic regression classifier. The dataset used for training and testing is preprocessed to ensure optimal performance. The model efficiently classifies emails by analyzing their text content, making it useful for filtering unwanted messages.

## Features
- Data collection and preprocessing
- Feature extraction using TF-IDF vectorization
- Model training using Logistic Regression
- Accuracy evaluation of the trained model
- Supports classification of new emails

## Dataset
The dataset used is `mail_data.csv`, which contains labeled emails categorized as `spam` or `ham`. The dataset includes a variety of spam messages, such as promotional emails and phishing attempts, as well as legitimate messages.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/SpamMailPrediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd SpamMailPrediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the Jupyter Notebook to execute the spam mail prediction steps:
```bash
jupyter notebook SpamMailPrediction.ipynb
```

You can also use the trained model to classify new email messages by providing text input.

## Dependencies
Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `jupyter`

You can install them using:
```bash
pip install numpy pandas scikit-learn jupyter
```

## Model Training and Evaluation
1. The dataset is loaded and preprocessed by removing null values.
2. TF-IDF vectorization is applied to transform text data into numerical features.
3. A Logistic Regression model is trained on the dataset.
4. Model accuracy is evaluated using `accuracy_score`.
5. The trained model can be tested with new email samples.

## Future Improvements
- Implementing deep learning models such as LSTMs for better accuracy.
- Expanding the dataset to include more diverse spam messages.
- Deploying the model as a web application for real-time spam detection.

## Contributing
Feel free to fork this repository and contribute by submitting a pull request.

