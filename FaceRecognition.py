
import cv2


class FaceRecognition:
    m_cascade = None

    def __init__(self):
        # Load Haar cascade classifier:
        self.m_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default_b.xml")

    def findFace(self, frame):


# Cargamos la imagen y la convertimos a grises:
img = cv2.imread('imagen_input.jpg')
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Nota: la imagen de ejemplo que hemos utilizado para el tutorial ya est� en blanco y negro,
# por lo que no ser�a necesario convertirla. Lo he hecho igualmente por si m�s adelante quer�is
# probar con una imagen en color.


# Buscamos los rostros:
coordenadas_rostros = cascada_rostro.detectMultiScale(img_gris, 1.3, 5)
# Nota 1: la funci�n detectMultiScale() requiere una imagen en escala de grises. Esta es la raz�n
# por la que hemos hecho la conversi�n de BGR a Grayscale.
# Nota 2: '1.3' y '5' son par�metros est�ndar para esta funci�n. El primero es el factor de escala ('scaleFactor'): la
# funci�n intentar� encontrar rostros escalando la imagen varias veces, y este factor indica en cu�nto se reduce la imagen
# cada vez. El segundo par�metro se llama 'minNeighbours' e indica la calidad de las detecciones: un valor elevado
# resulta en menos detecciones pero con m�s fiabilidad.


# Ahora recorremos el array 'coordenadas_rostros' y dibujamos los rect�ngulos sobre la imagen original:
for (x, y, ancho, alto) in coordenadas_rostros:
    cv2.rectangle(img, (x, y), (x + ancho, y + alto), (0, 255, 0), 3)

# Abrimos una ventana con el resultado:
cv2.imshow('Output', img)
print("\nMostrando resultado. Pulsa cualquier tecla para salir.\n")
cv2.waitKey(0)
cv2.destroyAllWindows()