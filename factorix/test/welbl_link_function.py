# this code shows facts about the link function used by Johannes Welbl in his KB factorization experiments


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10., 4., 0.01)
eta = 0.1

def softplus(xx):
    return np.log(1.0 + np.exp(xx))

def sigmoid(xx):
    return 1.0 / (1.0 + np.exp(-xx))

s_pos = np.exp(softplus(x))
s_neg = np.exp( - 1.0 / eta * softplus(x))
s = s_pos + s_neg
p_pos = s_pos / s
p_neg = s_neg / s

# red dashes, blue squares and green triangles
[line_true, line_false] = plt.plot(x, p_pos, 'b-', x, p_neg, 'r-')
plt.ylabel('Probability')
plt.xlabel('Score Xf')
plt.legend([line_true, line_false], ['P(Yf=T|Xf)', 'P(Yf=F|Xf)'])

plt.show()
