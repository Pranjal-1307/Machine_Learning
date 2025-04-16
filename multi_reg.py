#USING ARRAY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x=np.array([[1000,3,10],[2000,5,12],[3000,7,18]])
y=np.array([3000,4000,5000])

model=LinearRegression()
model.fit(x,y)

new_house=np.array([[2000,4,10]])
predicted_pr= model.predict(new_house)
print(f"{predicted_pr[0]}")

predicted_p=model.predict(x)
plt.figure(figsize=(8,6))
plt.scatter(y,predicted_p,color='blue',label='A vs P')
plt.plot([min(y),max(y)],[min(y),max(y)],color='red',linestyle='dashed',label='fit')
plt.legend()
plt.xlabel('X($)')
plt.ylabel('Y($)')
plt.title('house')
plt.grid(True)
plt.show()



#USING CSV 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read("file.csv")

x=df[['']]
y=df[['']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

new_c=([[220,22,4]])
pred=model.predict(new_c)
print(f"{predicted_pr[0]}")

pred=model.predict(x_test)

plt.figure(figsize=(8,6))
plt.scatter(y,pred,color="",label="")
plt.plot([min(y) max(y)],[min(y) max(y)],color="",linestyle="",label="")
plt.legend()
plt.xlabel('X($)')
plt.ylabel('Y($)')
plt.title('house')
plt.grid(True)
plt.show()