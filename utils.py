import numpy as np
import pandas as pd
from typing import List, Tuple


def get_autocorr_coeff(data: np.ndarray, max_lag: int = 10) -> List:
    data_pd = pd.Series(data)
    autocorr_coeffs = []
    
    for i in range(max_lag+1):
        autocorr_coeffs.append(data_pd.autocorr(i))

    return autocorr_coeffs

# Removes empty coefficients added in calculate_levinson_durbin to keep indexing from 1
def clean_coeff_list(coeff_list: List) -> List:
    del coeff_list[0]
    return [coeffs[1:] for coeffs in coeff_list]

# Pads signal with zeros at beginning and end
def pad_signal(data: np.ndarray, padding: int) -> np.ndarray:
    return np.pad(data, (padding, padding), 'constant', constant_values=(0, 0))

# Given reflection coefficients returns True if AR model is stable
def check_stability(refl_coeffs: List) -> bool:
    for refl_coeff in refl_coeffs:
        if np.abs(refl_coeff) >= 1:
            return False
    return True

def calculate_levinson_durbin(data: np.ndarray, r_max: int = 10) -> Tuple:
    p_list = get_autocorr_coeff(data, r_max)

    sigmas = []
    sigmas.append(p_list[0])    # Add sigma_0

    a_list = [[0] for _ in range(0, r_max+1)]   # Adding 0 to every list to start indexing from 1 possible later
    k_list = []     # Reflection coefficients
    for i in range(1, r_max+1):
        p_i = p_list[i]

        z = 0   # Sum from j=1 to i-1 a_j,i-1(N)*p(i-j)(N)

        for j in range(1, i):
            z += a_list[i-1][j]*p_list[i-j]
        ki = (p_i - z) / sigmas[i-1]
        k_list.append(ki)

        for j in range(1, i+1):
            if j == i:
                a_list[i].append(ki)
            else:
                a_list[i].append(a_list[i-1][j]-ki*a_list[i-1][i-j])

        sigma = (1-ki**2)*sigmas[i-1]
        sigmas.append(sigma)

    a_list = clean_coeff_list(a_list)
    return a_list, k_list