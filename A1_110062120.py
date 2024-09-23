import scipy.linalg.blas as blas
import scipy.linalg.lapack as lapack
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time 

# function to check if the solution is equal in lapack and numpy function
def check_equal(n, A, B):
    for i in range(n):
        if (A[i]!=B[i]):
            return 0
    return 1

# evaluate Ax = b
result = []
sizes = [10, 100, 1000, 2000, 3000, 4000, 5000] #different matrix size
for n in sizes:
    #random number put in the matrix
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    start = time.time()
    #calculate AC=b by lapack function
    lu, piv, C, info = lapack.dgesv(A, b)
    end = time.time()
    #calculate AC=b by numpy function
    D = np.linalg.solve(A,b)
    if (check_equal(n, C, D)!=1):
        print('there are some errors for n = ', n) 
    #calculate the time spend in calculating lapack function
    result.append(end-start)
        
plt.plot(sizes, result) #matrix size, calculation time
plt.title("Execution Time for AX=b") # title
plt.ylabel("Execution time (seconds)") # y label
plt.xlabel("Matrix size") # x label
plt.show