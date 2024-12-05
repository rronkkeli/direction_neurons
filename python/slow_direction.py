# Kokoelma neuroverkon kerrosfunktioita, eli pääasiassa input ja dense
import keras
import numpy as np

def dense(values_in: np.ndarray, weights: np.ndarray, biases: np.ndarray) -> np.ndarray:
    '''
    Tämä funktio on siis käytännössä äärimmäisen hidas dense-funktio, joka suorittaa matriisilaskutoimitukset solu kerrallaan.
    Valmistaudu vanhenemaan odotellessasi tulosta.
    '''

    # Otetaan talteen korkeus ja leveys tulostaulua varten
    # Painotaulu on muotoa (korkeus, leveys) ja tulostaulu on sen leveyden pituinen
    # eli käytetään muuttujaa muuttujaa h tulostaulun korkeutta
    # Jos painojen muoto on (3, 64), niin..
    w, h = weights.shape

    # ..tulostaulun muoto on (64)
    # Luodaan tulostaulu joka on painojen leveys kertaa painojen korkeus
    output = np.zeros((1, h), dtype=float)

    # Käydään solu solulta laskennat läpi
    # Jokaista saraketta kohti..
    for c in range(h):
        # ..jokaisen rivin alkio..
        for r in range(w):
            # ..kerrotaan ja summataan yhteen
            output[0, c] += values_in[0, r] * weights[r, c]

        # Ja lopulta lisätään bias-arvo
        output[0, c] += biases[c]

    # Palautetaan tuloksena syntynyt taulu
    return output

def maximum(values_in: np.ndarray) -> str:
    # Määritellään suunnat ihmisille luettaviin muotoihin
    directions = [
        "x-alas",
        "x-ylös",
        "y-alas",
        "y-ylös",
        "z-alas",
        "z-ylös"
    ]

    # Asetetaan maksimiindeksi alussa nollaksi ja maksimiarvo ensimmäisen kentän arvoksi
    maximum_index = 0
    maximum_value = values_in[0, 0]

    for i in range(1, len(values_in[0])):
        if maximum_value < values_in[0, i]:
            maximum_value = values_in[0, i]
            maximum_index = i

    # Palautetaan ihmisluettava suunta
    return directions[maximum_index]

def get_direction(point: np.ndarray) -> str:
    '''
    Palauttaa neuroverkon arvaaman suunnan. Ottaa parametrina pisteen, eli x, y ja z koordinaatit.
    '''

    # Luetaan malli
    model = keras.saving.load_model("direction.keras")

    # Tiivistyskerros
    l = model.get_layer("tiivistys")
    w = l.weights[0]
    b = l.weights[1]
    out = dense(point, w, b)

    # Ulostuskerros
    l = model.get_layer("tulos")
    w = l.weights[0]
    b = l.weights[1]
    out = dense(out, w, b)

    # Palautetaan korkein kerroksen arvo
    return maximum(out)
