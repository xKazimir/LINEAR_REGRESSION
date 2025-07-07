import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#for dataset_1 this learning rate is more optimal, though 0.0003 works fine as well
#learning_rate = 0.0006
#for dataset_2
learning_rate = 0.0003

#plotting data with line
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
    plt.title('Linear Regression Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def SSR_der (data,a,b,is_for_intercept) :
    ssr_der_val = 0.0
    for i in range(len(data)) :
        if is_for_intercept == 0 :
            ssr_der_val += -2* (data["y"][i] - (a * data["x"][i] + b))
        else :
            ssr_der_val += -2* data["x"][i]* (data["y"][i] - (a * data["x"][i] + b))
    return ssr_der_val

def SSR (data,a,b) :
    ssr_val = 0.0
    for i in range(len(data)):
        ssr_val +=(data["y"][i] - (a * data["x"][i] + b)) ** 2
    return ssr_val


def fit_line_to_data (data) :
    eps = 1e-10
    a,b = 1,0
    ssr_der = 1.0
    count = 0
    prev_cost = float('inf')

    while count < 10000:
        b -= SSR_der(data,a,b,0) * learning_rate
        a -= SSR_der(data, a, b, 1) * learning_rate
        print ("Iteration :" + str(count) + " A:"+ str(a) + " B:"+ str(b) )
        if count % 100 == 0 :
            show_data (data,a,b)
        count += 1
        ssr = SSR(data, a, b)
        if abs(prev_cost - ssr) < eps :
            break
        prev_cost = ssr
    return a,b

def var (data,a,b) :
    var_val = 0.0
    for i in range (len(data)) :
        var_val += (data["y"][i] - (a * data["x"][i] + b))**2
    return var_val/len(data)

def main():
    # load data
    #df = pd.read_csv("linear_regression_dataset.csv")
    df = pd.read_csv("linear_regression_dataset_2.csv")
    #df = pd.read_csv("linear_regression_dataset_3.csv")


    #show mean and data
    mean_y = df["y"].mean()
    show_data(df, 0, mean_y)
    #fitting line to data
    a,b = fit_line_to_data(df)

    #numpy fitting for checking
    x = df["x"].values
    y = df["y"].values
    # deg=1 znamená lineárny tvar y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    print(f"NumPy polyfit: y = {a:.4f}x + {b:.4f}")

    #R**2
    print ("variance for mean : " + str(var(df,0,mean_y)))
    print ("variance for fitted line : " + str(var(df,a,b)))
    R_squared = (var(df,0,mean_y)-var(df,a,b))/var(df,0,mean_y)
    print("R**2: " + str(R_squared))
    #other way of checking R**2
    #print ((SSR(df,0,mean_y) - SSR(df,a,b)) / SSR(df,0,mean_y))

    #p-value
    F = ((SSR(df,0,mean_y) - SSR(df,a,b))/ (2-1) ) / (SSR(df,a,b)/(len(df)-2))
    print("F: " + str(F))

main ()