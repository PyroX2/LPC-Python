import numpy as np
from typing import Tuple
from ar import ARModel
from utils import calculate_levinson_durbin, pad_signal


class Transmitter:
    def __init__(self, ar_order: int = 10, segment_width: int = 256, overlap: int = 10) -> None:
        self.segment_width = segment_width
        self.overlap = overlap
        self.ar_model = ARModel(order=ar_order)

    def forward(self, data: np.ndarray) -> Tuple:
        segmented_data = self._prepare_data(data)

        errors_list = []
        coeffs_list = []
        targets_list = []
        predictions_list = []
        for segment in segmented_data:
            ar_model_coeffs, refl_coeffs = calculate_levinson_durbin(pad_signal(self._flatten_segment_ends(segment), self.ar_model.order))
            ar_model_coeffs = ar_model_coeffs[-1]
            self.ar_model.set_coeffs(np.flip(np.array(ar_model_coeffs, dtype=np.float32)))
            
            targets = []
            predictions = []
            segment_errors = []
            for i in range(self.ar_model.order, len(segment)):
                curr_data = segment[i-self.ar_model.order:i]
                prediction = self.ar_model._predict(curr_data)
                target = segment[i]
                error = target - prediction
                segment_errors.append(error)
                predictions.append(prediction)
                targets.append(target)
            
            errors_list.append(segment_errors)
            coeffs_list.append(ar_model_coeffs)
            predictions_list.append(predictions)
            targets_list.append(targets)

        errors_list = np.array(errors_list)
        coeffs_list = np.array(coeffs_list)
        predictions_list = np.array(predictions_list)
        targets_list = np.array(targets_list)

        return coeffs_list, errors_list, refl_coeffs

    # Given data cuts it into segments with specified overlap
    def _prepare_data(self, data: np.ndarray) -> np.ndarray:
        n_segments = len(data) // (self.segment_width - self.overlap)

        segments = []
        for i in range(n_segments):
            start_index = i*self.segment_width-i*self.overlap
            end_index = start_index+self.segment_width
            segment = data[start_index:end_index]

            # Drop last segment if it's length is not equal to segment width
            if i == n_segments-1 and len(segment) != self.segment_width:
                break

            segments.append(segment)

        return np.array(segments)
    
    def _flatten_segment_ends(self, segment: np.ndarray) -> np.ndarray:
        cos_arg = (2*np.pi / (len(segment)+1))*np.arange(1, len(segment)+1)
        return 0.5*(1 - np.cos(cos_arg))*segment