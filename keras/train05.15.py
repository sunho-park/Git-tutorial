import numpy as np
x=np.array(range(1, 101))
y=np.array(range(101, 201))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.4)

print(x_train)
print(x_test)
