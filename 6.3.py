import random
import zlib
import sys

random_string = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', k=1000))

compressed_string = zlib.compress(random_string.encode())

compression_ratio = len(random_string.encode()) / len(compressed_string)

original_size = sys.getsizeof(random_string)
compressed_size = sys.getsizeof(compressed_string)

print("Compressed string (in bytes representation):", compressed_string)
print("Compressed size (bytes):", compressed_size)
print("Compression ratio:", compression_ratio)

# 6.3 b)
# For a string, we expect a 1.25-1.3 compression ratio since are only using the alphanumeric characters in our string which is 62/256 bytes which is 
# around 24% of the full byte range. Therefore, there are likely more repeatable characters that can be compressed. If our data used 
# the full capability of the byte range it would be a 1 ratio as mentioned in the book