import numpy as np
from ar import ARModel


class Receiver:
    def __init__(self, ar_order: int = 10, segment_width: int = 256, overlap: int = 10) -> None:
        self.segment_width = segment_width
        self.overlap = overlap
        self.ar_model = ARModel(order=ar_order)

    def forward(self, errors_indices: np.ndarray, max_vals: np.ndarray, coeffs: np.ndarray, init_data: np.ndarray, n_bits: int = 2) -> np.ndarray:
        reconstructed_data = []
        reconstructed_data.extend(init_data.tolist())    # Add initial data to the beginning

        for i, (segment_errors_idx, ar_model_coeffs) in enumerate(zip(errors_indices, coeffs)):    
            segment_max_val = max_vals[i]
            segment_errors = self._indices_to_errors(segment_errors_idx, segment_max_val, n_bits=n_bits)

            # Set AR model coefficients for current segment
            self.ar_model.set_coeffs(np.flip(np.array(ar_model_coeffs, dtype=np.float32)))

            for j in range(len(segment_errors)):
                curr_data = np.array(reconstructed_data[-self.ar_model.order:], dtype=np.float32)
                prediction = self.ar_model._predict(curr_data)
                reconstructed_value = prediction + segment_errors[j]
                reconstructed_data.append(reconstructed_value)

        return np.array(reconstructed_data)
    
    def _indices_to_errors(self, errors_indices: np.ndarray, max_val: float, n_bits: int) -> np.ndarray:
        n_unique_values = 2**n_bits
        uniq_vals = np.linspace(-max_val, max_val, n_unique_values)

        index_to_value = {i: v for i, v in enumerate(uniq_vals)}

        # Vectorized mapping using list comprehension
        errors = np.array([index_to_value[idx] for idx in errors_indices])

        return errors