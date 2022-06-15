# 1. List Tuple Set Dictionary : 
NUMPY, PANDAS, Matplotlib
# 2. Pearson Correlation Coefficient. 


```
# %%
# WAP to determine Pearson Correlation Coefficient   
import pandas
import math 
import matplotlib.pyplot as plt
import json

# %%
# data
x = [7,6,8,5,6,9]
y = [12,8,12,10,11,13]

input_len = len(x)
plt.title("Age vs Weight scatter Graph:  ")
plt.xlabel("Age ")
plt.ylabel("Weight ")
plt.scatter(x, y)
plt.show()

print(input_len)

# %%
# calculating values a/c to formula. 
x_sum = sum(x)
y_sum = sum(y)
x_sqsum = 0
y_sqsum = 0
xy_sum = 0

x_sq_list =[]
y_sq_list = []
xy_list = []
for i,j in zip(x,y):
    # print(i,j)
    xy_sum += i*j
    x_sqsum += i*i
    y_sqsum += j*j

    x_sq_list.append(i**2)
    y_sq_list.append(j**2)
    xy_list.append(i*j)

n = input_len * xy_sum - x_sum*y_sum

d = (input_len * x_sqsum - x_sum**2 ) * (input_len * y_sqsum - y_sum**2 )
# final coff.
r = n/(math.sqrt(d))


# %%
# printing table : 
print(pandas.DataFrame({'X': x, 'Y' : y, 'X_sq' : x_sq_list, 'Y_sq': y_sq_list, 'XY' : xy_list }) )
print('Sum table : ')
sum_dict = {'X_sum': x_sum, 'Y_sum' : y_sum, 'X_sq_sum' : x_sqsum, 'Y_sq_sum': y_sqsum, 'XY_sum' : xy_sum }

print(json.dumps(sum_dict, indent=4) )

print("Value of Pearson Correlation Coefficient : ", r)
if (r == 0):
    print("No Corelation")
elif(r >= 0.5 ):
    print("Strong : Positive Co-relation")
elif(r <= -0.5 ):
    print("Strong : Negative Co-relation")
elif(r < 0 ):
    print("Weak : Negative Co-relation")
elif(r > 0 ):
    print("Weak : Positive Co-relation")


```

# 3. Simple regression without any inbuilt library.

```
# %%
import pandas as pd
import matplotlib.pyplot as plt
 
# read csv
path= 'data.csv'
df = pd.read_csv(path)

xy_list =[]
xsq_list =[]
ysq_list =[]
 
x_sum = 0
y_sum = 0
y_sqsum = 0
x_sqsum= 0
xy_sum= 0
# for x,y in zip(df[1],df[2]):
for x,y in zip(df['age'],df['gcose']):
    x_sum += x 
    y_sum += y 
    y_sqsum += y*y 
    x_sqsum += x*x
    xy_sum += x*y
 
    xy_list.append(x*y)
    ysq_list.append(y*y)
    xsq_list.append(x*x)
 
n = 6
 
A = (y_sum * x_sqsum - x_sum * xy_sum) / (n * x_sqsum - x_sum **2)
 
B = (n * xy_sum - x_sum*y_sum) / (n * x_sqsum - x_sum**2)
# age = int(input("Enter Age (to predict glucose level ): "))
# a = 30
# glucose = A + B*a
# print(f"Your Glucose level is {glucose}")

# %%
print(f"Value of constant A : {A} \nValue of constant B : {B}")
print(pd.DataFrame({'Age (X)': x, 'Glucose(Y)' : y, 'X_sq' : xsq_list, 'Y_sq': ysq_list, 'XY' : xy_list }) )

predicted =[]
print("Age -------- Glucose ---- Predicted(glucose level)")
for i,j in zip(df['age'],df['gcose']):
    predicted.append(A+B*i)
    print(f"{i}  --------   {j}    ---- {round(A+B*i,2)}")

# %%
# plotting graph
plt.title(f"Age vs glucose")
plt.xlabel('Age')
plt.ylabel('Glucose')
df['predicted'] = predicted

plt.scatter(df['age'], df['gcose'])
plt.plot(df['age'], df['predicted'], color = "r", marker = "x", alpha = 0.6)
plt.show()

```

# 4. Simple regression using different library. 


```
# Simple Linear Regression With scikit-learn.  
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read csv
path= 'data.csv'
df = pd.read_csv(path)

# get data
x = np.array(df['age']).reshape(-1,1) 
# make 2d array coz , sk learn needs it

y = np.array(df['gcose'])

model = LinearRegression()
# calculate the optimal values of the weights A(intercept) and B(slope)
model.fit(x,y) 

#  coefficient of determination, 𝑅², with .score()  
# r_sq = model.score(x,y)
# print(f"coefficient of determination: {r_sq}")
print(f"Value of constant A : {model.intercept_}")
print(f"Value of constant B : {model.coef_}")

predicted = model.predict(x)

# print table to compare. 
print("Age -------- Glucose ---- Predicted(glucose level)")
for i,j,k in zip(df['age'],df['gcose'],predicted):
    print(f"{i}  --------   {j}    ---- {round(k,2)}")

# plotting graph
plt.title(f"Age vs glucose")
plt.xlabel('Age')
plt.ylabel('Glucose')
df['predicted'] = predicted

plt.scatter(df['age'], df['gcose'])
plt.plot(df['age'], df['predicted'], color = "r", marker = "x", alpha = 0.6)
plt.show()



```
# 5. Multiple regression using different library. 

```
# Multiple Linear Regression With scikit-learn.  
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# get data 
x = [ [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34],
     [60, 35] ]

y = [4, 5, 20, 14, 32, 22, 38, 43]

# maiking numpy array for processing
x, y = np.array(x), np.array(y)

model = LinearRegression().fit(x,y) 

print(f"Value of intercept (b0): {model.intercept_}")
print(f"Value of coefficients[b1,b2,...bn] : {model.coef_}")

predicted = model.predict(x)

# print(x[:, 0], x[:, 1],sep='\n')
# print tables to compare data
df = pd.DataFrame({'X1': x[:,0], 'X2' : x[:,1], 'Y(Original)': y, 'Y(Predicted)':predicted })
print(df) 


```
# 6. Find association rules from given dataset using Naïve algorithm. 


```
from itertools import chain, combinations

def all_combo(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

# list of items in the transactions
mylist = [
    ['Bread', 'Cheese'],
    ['Bread', 'Cheese', 'Juice'],
    ['Bread', 'Milk'],
    ['Cheese', 'Juice', 'Milk']
]

# finding unique items
items = set()
for i in mylist:
    for j in i:
        items.add(j)

items = list(items)
print(f"Unique items : \n\t{items}\n")

support = 0.5
min_freq = support * len(items)
min_confidence = 0.75

# finding powerset of all the unique items
pwset_items = list(" ".join(i) for i in all_combo(items))
# print(pwset_items)

# creating dictionary from powerset 
# that have itemsets and their frequency
freq_items = dict.fromkeys(pwset_items, 0)
for k in freq_items.keys():
    for j in mylist:
        s1 = set(k.split(" "))
        s2 = set(j)
        if s1.issubset(s2):
            freq_items[k] = freq_items[k] + 1

print(f"Itemset Combination and their Frequency : ")
for k,v in freq_items.items():
    print(f"\t{k} : {v}")
print("")

# removing itemsets that have less than required minimum frequency
freq_items2 = dict()
for k,v in freq_items.items():
    if v >= min_freq:
        freq_items2[k] = v

print(f"Itemset after checking minimum support : ")
for k,v in freq_items2.items():
    print(f"\t{k} : {v}")
print("")

# removing itemsets that are alone
freq_items3 = dict()
for k,v in freq_items2.items():
    if len(k.split(" ")) > 1:
        freq_items3[k] = v

print(f"Itemset after removing alone items : ")
for k,v in freq_items3.items():
    print(f"\t{k} : {v}")
print("")

#making pairs from both sides
pairs = []
for k,v in freq_items3.items():
    temp_mylist = k.split(" ")
    # print(temp_mylist)
    for i in temp_mylist:
        pair_mylist = []
        pair_mylist.append(i)
        temp_mylist2 = list(temp_mylist)
        temp_mylist2.remove(i)
        pair_mylist.append(" ".join(temp_mylist2))
        # print(pair_mylist)
        pairs.append(" -> ".join(pair_mylist))

print(f"Unique pairs : \n\t{pairs}\n")

# dictionary for holding pairs and their confidence
freq_items_conf = dict.fromkeys(pairs, 0)

for k,v in freq_items_conf.items():
    t_mylist = k.split(" -> ")
    nt_mylist = []
    for i in t_mylist:
        nt_mylist.extend(i.split(" "))

    freq = 0
    for k2,v2 in freq_items3.items():
        if set(nt_mylist) == (set(k2.split(" "))):
            freq = v2
            break
    
    count = 0
    for x in mylist:
        if set([nt_mylist[0]]).issubset(set(x)):
            count += 1 
    
    freq_items_conf[k] = round(freq/count,8)

print(f"Pairs and their Confidence : ")
for k,v in freq_items_conf.items():
    print(f"\t{k} : {v}")
print("")

# removing pairs that have less than required minimum confidence
freq_items_conf_final = dict()
for k,v in freq_items_conf.items():
    if v >= min_confidence:
        freq_items_conf_final[k] = v

print(f"Pairs after checking minimum confidence : ")
for k,v in freq_items_conf_final.items():
    print(f"\t{k} : {v}")
print("")



```


# 7. Find association rules from given dataset using Apriori algorithm. (weka)

```
Weka

```




# 8. WAP to implement single Neural Network.

```
import numpy as np 

np.random.seed(101) #fixing the seed 
inputs= np.random.rand(3) #input array of 1x3 with values b/w 0-1

weights=np.random.rand(3) #weights array of 1x3 with values b/w 0-1

weighted_input= np.dot(inputs,weights) ¬¬¬
#dot product of inputs and weights i.e. x1w1+x2w2+x3w3

def activation_func(weighted_input):  #activation function definition

    if(weighted_input <= 0):   #step function condition
        return 0
    else:
        return 1

output=activation_func(weighted_input)  #final output

print("Inputs->",inputs)
print("weights->",weights)
print("weighted_input dot product->",weighted_input) 
print("output->",output)

```
