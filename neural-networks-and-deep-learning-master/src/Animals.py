import pandas as pd

import numpy as np

from sklearn import linear_model
import matplotlib.pyplot as plt

datafram = pd.read_fwf('brain_body.txt')

x_values = datafram[['Brain']]
y_values = datafram[['Body']]

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#plt.scatter(x_values, y_values)

a = 10

for i in range(10):
    print(i)

return    

x = [i for i in range(10)]
y = [np.power(2, i) for i in range(10)]
plt.scatter(x, y)
plt.plot(np.polyfit(x, y, 4))

#plt.plot(x_values, body_reg.predict(x_values))

plt.show()
