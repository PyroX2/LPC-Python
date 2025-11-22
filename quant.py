import numpy as np


def pack_bits(bit_string):
    # Pad the string so length is multiple of 8
    length = len(bit_string)
    padded_length = ((length + 7) // 8) * 8  # next multiple of 8
    bit_string = bit_string.rjust(padded_length, '0')

    packed_bytes = bytearray()
    for i in range(0, padded_length, 8):
        byte = bit_string[i:i+8]
        packed_bytes.append(int(byte, 2))
    return packed_bytes

def unpack_bits(packed_bytes):
    bits = ''.join(f'{byte:08b}' for byte in packed_bytes)
    return bits


def quantize_to_levels(arr, levels):
    flat = arr.ravel()
    idx = np.abs(flat[:, None] - levels[None, :]).argmin(axis=1)
    return levels[idx].reshape(arr.shape)


def quantize_n_bit(arr, n, max_val):
    n_unique_values = 2**n

    # Generate quantization levels between -max_val and max_val
    uniq_vals = np.linspace(-max_val, max_val, n_unique_values)

    # Quantize the array to the nearest levels
    arr_quant = quantize_to_levels(arr, uniq_vals)

    # Create mapping from value to index
    value_to_index = {v: i for i, v in enumerate(uniq_vals)}

    # Vectorized mapping using list comprehension
    indices = np.array([value_to_index[val] for val in arr_quant])

    
    return arr_quant, indices