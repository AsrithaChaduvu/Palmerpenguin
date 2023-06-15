import pandas as pd

df = pd.read_csv("penguins_size.csv")

island = "Dream"
cul_len = 46.6
cul_depth = 17.9
flip_length = 193
body_mass = 3340
sex = "FEMALE"


print("############ PALMER PENGUINS ####################")
print("1 : Torgersen")
print("2 : Biscoe")
print("3 : Dream")
flag = int(input("Choose Island: "))
if (flag == 1):
    island = 'Torgersen'
elif flag == 2:
    island = 'Biscoe'
else:
    island = 'Dream'

cul_len = float(input("Enter Culmen Length in mm : "))
cul_depth = float(input("Enter Culmen Depth in mm : "))
flip_length = float(input("Enter Flipper Length in mm : "))
body_mass = float(input("Enter Body_mass in g : "))

print("1 : MALE")
print("2 : FEMALE")
flag = int(input("Choose Gender: "))
if (flag == 1):
    sex = 'MALE'
else:
    sex = 'FEMALE'


df.loc[len(df)] = ["", island, cul_len, cul_depth, flip_length, body_mass, sex]
df = df.fillna(0)
df = pd.get_dummies(df, columns=['sex', 'island'], drop_first=True)



from sklearn import preprocessing as pre

scale = pre.StandardScaler().fit(df.drop(columns=['species']))
transformed = scale.transform(df.drop(columns=['species']))
df_scaled = pd.DataFrame(transformed, columns=df.columns[1:])
INP = df_scaled[-1:]
X = df_scaled[:-1]
Y = df['species'][:-1]

from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X, Y) 
output = knn.predict(INP)
print("Species:", output[0])
