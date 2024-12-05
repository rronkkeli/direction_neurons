import keras
from keras import layers
import tensorflow as tf
import numpy as np
# Pukkaa neuroverkon C-kielen otsaketiedostoon (h-tiedosto)

model = keras.saving.load_model("direction.keras")
model.summary()
writer_buffer_h = "// Tämä on python-skriptin tuottama h-tiedosto\n\n#ifndef NEURAL_NET_H\n#define NEURAL_NET_H\n\n"
writer_buffer_c = "// Tämä on python-skriptin tuottama c-tiedosto\n\n#include \"neural_net.h\"\n\n"

for layer in model.layers:
    weights = layer.get_weights()

    # Jos kerroksella on painoarvoja
    if weights:
        wr, wc = weights[0].shape # painot: rivit ja sarakkeet
        br = weights[1].shape[0] # bias-arvot: rivit ja sarakkeet

        warr_name = "w_" + layer.name
        barr_name = "b_" + layer.name

        # Aloitetaan taulu
        # declaration_buffer kirjoitetaan h-tiedostoon
        # array_buffer kirjoitetaan c-tiedostoon
        declaration_buffer = "float {}[{}][{}] ".format(warr_name, wr, wc) 
        array_buffer = declaration_buffer + "= {\n"
        declaration_buffer = "extern " + declaration_buffer + ";\n"
        
        # Pusketaan painoarvot puskurimuistiin
        # ax ja bx ovat vain laskureita indexeille
        for ax in range(wr):
            row = weights[0][ax]
            row_buffer = "    {"

            # Lisätään yksitellen taulun rivin elementit
            for bx in range(wc):
                # Lisätään elementin eteen pilkku, jos kyse ei ole ensimmäisestä sarakkeesta,..
                if bx > 0:
                    row_buffer += ", " + str(row[bx])

                # ..muuten ei lisätä pilkkua
                else:
                    row_buffer += str(row[bx])

            # Mennään seuraavalle taulun riville..
            if ax + 1 < wr:
                row_buffer += "},\n"

            # ..tai jos kyseessä on viimeinen rivi, lopetetaan taulu
            else:
                row_buffer += "}\n};\n\n"

            # Lisätään taulupuskuriin rivipuskurin sisältö ennen sen nollaamista,
            # ettei turha työ mene hukkaan
            array_buffer += row_buffer

        # Kirjoitetaan bias-arvot puskurimuistiin
        array_buffer += "float {}[{}][1] ".format(barr_name, br) + "= {\n"
        declaration_buffer += "extern float {}[{}][1];\n".format(barr_name, br)

        # Toistetaan sama operaatio bias arvoille kuin yllä painoarvoille
        for ax in range(br):
            row = weights[1][ax]
            row_buffer = "    {"

            # Lisätään taulun rivin elementti (bias arvoja on vain yksi per rivi)
            # Lisätään elementin eteen pilkku, jos kyse ei ole ensimmäisestä sarakkeesta,..
            row_buffer += str(row)
            
            # Mennään seuraavalle taulun riville..
            if ax + 1 < br:
                row_buffer += "},\n"

            # ..tai jos kyseessä on viimeinen rivi, lopetetaan taulu
            else:
                row_buffer += "}\n};\n\n"

            # Lisätään taulupuskuriin rivipuskurin sisältö ennen sen nollaamista,
            # ettei turha työ mene hukkaan
            array_buffer += row_buffer

        # Päivitetää kirjoituspuskuri
        writer_buffer_c += array_buffer
        writer_buffer_h += declaration_buffer

writer_buffer_h += "\n#endif\n"
# Kirjoitetaan kirjoistuspuskuri h-tiedostoon
file = open("neural_net.h", "wb")
file.write(writer_buffer_h.encode("utf-8"))
file.close()

file = open("neural_net.c", "wb")
file.write(writer_buffer_c.encode("utf-8"))