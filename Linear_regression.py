import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#load data
df = pd.read_csv("linear_regression_dataset.csv")

def show_data (data,a,b) :
    #body
    sns.scatterplot(x='x', y='y', data=data)
    #priamka
    x_vals,y_vals = [],[]
    for i in range (100) :
        if i == 0 :
            x_vals.append(int(data["x"].min()) + ((int(data["x"].max() - data["x"].min())) / 100))
        else :
            x_vals.append (x_vals[i-1]+((int(data["x"].max()-data["x"].min()))/100))
        y_vals.append (a * x_vals[i] + b)
    plt.plot(x_vals, y_vals, color='red', linewidth=2, label=f'y = {a}x + {b}')
    #vykreslenie
    plt.title('Linear Regression Dataset (Seaborn)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
#SS(mean)
mean_y = df["y"].mean()
show_data(df,0,mean_y)

