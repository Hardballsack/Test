#

import tensorflow as tf
import numpy as np
import pandas as pd
import keras

#

df = pd.read_csv(r"C:\Users\Lofti\Downloads\Balls\gender_submission.csv")
#print(df[['Survived']])
print("Survived:", df.iloc[:,1].sum(axis=0), "\nTotal passengers:", df.index[-1]) # Survivors presumed all Women

train = pd.read_csv(r"C:\Users\Lofti\Downloads\train.csv")
test = pd.read_csv(r"c:\Users\Lofti\Downloads\test.csv")
print(train.columns)

# Lets check how many women and men survived 
women = train.loc[train.Sex == 'female']['Survived']
rate_women = sum(women) / len(women)
print('percent of women who survived', rate_women)

men = train.loc[train.Sex == 'male']['Survived']
rate_men = sum(men) / len(men)
print('percent of men who survived', rate_men)

y = train.Survived
features = ["Pclass", "Sex", "SibSp", "Parch"]
x = pd.get_dummies(train[features])
x_test = pd.get_dummies(test[features])
x = x.astype('float32')
x_test = x_test.astype('float32')

def get_model():
    I = keras.Input(shape=(5,))
    x = keras.layers.Flatten()(I)
    O = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=I, outputs=O)
    model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.mean_absolute_error])
    return model
model = get_model()
model.fit(x, y, epochs=1)
predictions = model.predict(x_test)

