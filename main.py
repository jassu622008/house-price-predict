import pandas as pd

from sklearn.linear_model import LinearRegression

bed_rooms = int(input("Enter Bedrooms: "))

bath_rooms = int(input("Enter Bathrooms: "))

feet = int(input("Enter Square Feet: "))

df = pd.read_csv("data.csv")

model = LinearRegression()

X = df[["bedrooms","bathrooms", "sqft_living"]]

Y = df[["price"]]

model.fit(X, Y)

res = model.predict(bed_rooms, bath_rooms, feet)

print(res)
