# predict.py

from joblib import load
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def predict(input_data, model_path='trained_model.joblib'):
    # Load the trained model
    model = load(model_path)

    # Assuming X is the original dataset used for scaling during training
    # If you have a separate scaler, load it similarly to the model
    data = pd.read_excel('Raisin_Dataset.xlsx')  # Load or define your original dataset for scaling
    data['Class'] = data['Class'].str.replace('Kecimen', '0')
    data['Class'] = data['Class'].str.replace('Besni', '1')

    cols = data.columns
    data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')
    X = data
    X = X.drop(['Class'], axis=1)


    # Standardize the input data using the same scaler
    sc = StandardScaler()
    sc.fit(X)
    scaled_input = sc.transform(input_data.reshape(1, -1))

    # Make predictions using the loaded model
    predictions = model.predict(scaled_input)

    return predictions

# if __name__ == "__main__":
#     # Example usage:
#     sample_input = np.array([79058, 454.4372156, 236.9642525, 0.853284574, 82555, 0.578255972, 1175.034])
#     result = predict(sample_input)
#     print("Predictions:", result)
