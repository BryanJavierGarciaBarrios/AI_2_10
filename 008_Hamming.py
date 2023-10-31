def hamming_code_encode(data):
    # Calculate the number of parity bits needed (m)
    m = 0
    while 2 ** m < len(data) + m + 1:
        m += 1

    # Create a list for the encoded data with placeholders for parity bits
    encoded_data = [0] * (len(data) + m)
    j = 0  # Index for the encoded data

    # Fill in data bits and placeholders for parity bits
    for i in range(1, len(encoded_data) + 1):
        if i & (i - 1) == 0:
            encoded_data[i - 1] = 0
        else:
            encoded_data[i - 1] = data[j]
            j += 1

    # Calculate parity bits
    for i in range(m):
        index = 2 ** i - 1
        parity = 0
        for j in range(index, len(encoded_data), 2 ** (i + 1)):
            for k in range(2 ** i):
                if j + k < len(encoded_data):
                    parity ^= encoded_data[j + k]
        encoded_data[index] = parity

    return encoded_data

def hamming_code_decode(encoded_data):
    m = 0
    while 2 ** m < len(encoded_data):
        m += 1

    error_position = 0
    for i in range(m):
        index = 2 ** i - 1
        parity = 0
        for j in range(index, len(encoded_data), 2 ** (i + 1)):
            for k in range(2 ** i):
                if j + k < len(encoded_data):
                    parity ^= encoded_data[j + k]
        if parity != encoded_data[index]:
            error_position += index + 1

    if error_position == 0:
        return encoded_data[:len(encoded_data) - m]
    else:
        print(f"Error detected at position {error_position}. Correcting...")
        encoded_data[error_position - 1] = 1 - encoded_data[error_position - 1]
        return encoded_data[:len(encoded_data) - m]

# Example
data = [1, 0, 1, 0]
encoded_data = hamming_code_encode(data)
print(f"Encoded data: {encoded_data}")
decoded_data = hamming_code_decode(encoded_data)
print(f"Decoded data: {decoded_data}")
