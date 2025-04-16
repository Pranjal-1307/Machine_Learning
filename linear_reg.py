#USING ARRAY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.array([500,800,1200,1500,1800]).reshape(-1,1)
y=np.array([50000,80000,120000,150000,180000])

model=LinearRegression()
model.fit(x,y)
new_size=np.array([[1000]])
predicted_p=model.predict(new_size)

print(f"{predicted_p[0]}")

plt.scatter(x,y,color='blue',label='data')
plt.plot(x,model.predict(x),color='red',label='registration')
plt.scatter(new_size,predicted_p,color='green',label=f'{predicted_p[0]}')

plt.xlabel("size")
plt.ylabel("price")
plt.title("House")
plt.legend()
plt.grid(True)
plt.show()



#USING CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.linear.model import LinearRegression
from sklearn.model_selection import train_test_split


df=pd.read("file.csv")
x=df[['hours']]
y=df.[['score']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

model=LinearRegression()
model.fit(x_train,y_train)
predicted_p=model.predict(x_test)

plt.figure(figsize(8,6))
plt.scatter(x,y,color='blue')
plt.plot(x,model.predict(x),color='yellow',linesize='dotted')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("House")
plt.legend()
plt.grid(True)
plt.show()

new_s=np.array([[1.5]])
predicted_p=model.predict(new_s)
print(f"{predicted_p[0]:2f}")

