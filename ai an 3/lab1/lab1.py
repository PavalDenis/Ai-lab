import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss="mean_squared_error")

xs = np.array([1,2,3,4,5,6,7,8], dtype=float)
ys = np.array([4,7,10,13,16,19,22,25])

model.fit(xs,ys,epochs=100)
print(model.predict(np.array([14.0])))
predicted_ys = model.predict(xs)
plt.plot(xs,ys,'b',label = 'Date reale')
plt.plot(xs,predicted_ys, 'r', label = "predictii")
plt.legend()
plt. show()