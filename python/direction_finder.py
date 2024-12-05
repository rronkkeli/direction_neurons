import numpy as np
from numpy import genfromtxt
import keras
from keras import layers
from custom_layers import middle_point, maximinimize

# Suunnat ovat eri järjestyksessä tietokannassa, kuin niiden tulisi olla lopputuloksessa
# korjatkaamme tämä ongelma luomalla taulu suunnista
directions = np.array([5, 4, 2, 3, 1, 0], dtype=np.float32)

# Luetaan CSV-tiedosto suoraan numpy-tauluksi ottaen mukaan myös suuntatieto opetusta varten
data = genfromtxt("measurements.csv", np.float32, delimiter=",", skip_header=1, usecols=(1, 2, 3, 4))
np.random.shuffle(data)

# Korjataan opetusdatan suunnat käyttämällä aiemmin luotua directions taulua
# Käydään siis läpi joka ikinen rivi ja muutetaan sen 'direction' luku
row_count = len(data)
i = 0

for row in data:
    # Jos row[3], eli rivin 'direction' kenttä on tietokannassa esimerkiksi 0,
    # haemme taulumuuttujasta directions elementin indeksistä 0 eli directions[0],
    # joka on 5
    ix = int(row[3])
    row[3] = directions[ix]

# Erotetaan datasta oikeat vastaukset opetusta varten..
labels = keras.utils.to_categorical(data[100:, 3], 6)
training_ds = data[100:, 0:3]
test_ds = data[:100, :3]
test_val = keras.utils.to_categorical(data[:100, 3], 6)

for i in range(len(training_ds)):
    training_ds[i] = middle_point(training_ds[i])

for i in range(len(test_ds)):
    test_ds[i] = middle_point(test_ds[i])

print("Preprocessing done!")

# Määritetään mallin rakenne. Syötekerros ja tiivistyskerros. Tuloskerrosta tässä ei merkitä, koska se on se mitä
# tiivistyskerros tuottaa
model = keras.Sequential([
       layers.Input(shape=(3,)),
    #    layers.Dense(6, activation="sigmoid"),
       layers.Dense(6, activation="softmax")
   ])

# Näytetään tiivistelmä neuroverkon rakenteesta
model.summary()

# Kootaan malli
model.compile(optimizer=keras.optimizers.Adam(), metrics=["accuracy"], loss=keras.losses.binary_crossentropy)

# Koulutetaan malli
hist = model.fit(training_ds, labels, batch_size=200, epochs=5, validation_split=0.2, shuffle=True, validation_steps=10, callbacks=keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2))

# ja viedään
# model.export("direction.model")
model.save("direction.keras")
falses = 0
trues = 0
prediction = model.predict(test_ds,  verbose=0)

for i in range(len(test_ds)):
    prediction[i] = maximinimize(prediction[i])
    correct = test_val[i]

    print(f"Tulos: {prediction[i]}, Oikea: {correct}")

    if np.array_equal(prediction[i], correct): trues += 1
    else: falses += 1

if falses == 100:
    print("Johtopäätös: algoritmi on paska")
else:   
    print(f"Osumia: {trues}, Huteja: {falses}")