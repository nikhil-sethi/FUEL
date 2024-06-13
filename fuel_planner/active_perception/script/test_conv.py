import numpy as np

size = 5
sigma = 2

arr = np.zeros((size, size, size))
depth = int(size/2)
norm = 0;
scale = np.exp(-(3*depth*depth)/(2*sigma*sigma))

for i in range(-depth, depth+1):
     for j in range(-depth, depth+1):
             for k in range(-depth, depth+1):
                     weight = np.exp(-(i*i + j*j + k*k)/(2*sigma*sigma))
                     arr[i+depth][j+depth][k+depth] = weight/scale
                     norm +=weight
# print(arr/arr.min())

print(arr)
print(arr/norm)
print(norm)