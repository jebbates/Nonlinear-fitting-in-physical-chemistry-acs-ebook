import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import sys

# Sigmoidal function definition
def sigmoid(x, c0, c1, c2, c3):
    return c0 + c1 * np.tanh((x - c2) / c3)

def process_data(file_path):
    # Step i: Import the data and convert it to a pandas dataframe
    data = pd.read_csv(file_path, header=None, names=['temp', 'trial1', 'trial2'])

    # Step ii: Average the two trials together to create a new column
    data['average'] = data[['trial1', 'trial2']].mean(axis=1)

    # Step iii: Perform a nonlinear fit using the sigmoidal function on each subset of the average data
    fit_results = []
    for i in range(0, len(data), 10):  # Assuming 10 points per subset
        subset = data.iloc[i:i+10]
        
        # Initial guesses for c0, c1, c2, c3
        initial_guesses = [0.5, 1.0, 30.0, 1.0]  # Here, c2 is initially guessed to be 30

        # Perform curve fitting
        popt, pcov = curve_fit(sigmoid, subset['temp'], subset['average'], p0=initial_guesses, maxfev=10000)
        
        # Extracting c2 and its uncertainty
        c2 = popt[2]
        c2_uncertainty = np.sqrt(np.diag(pcov))[2]
        fit_results.append((c2, c2_uncertainty))

    # Step iv: Summarize the values of the fit constant c2 and its uncertainty for the user
    for i, (c2, uncertainty) in enumerate(fit_results):
        print(f"Group {i+1}: c2 = {c2:.4f} ± {uncertainty:.4f}")

if __name__ == "__main__":
    # Use sys.argv to take the input CSV file path
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    process_data(file_path)

