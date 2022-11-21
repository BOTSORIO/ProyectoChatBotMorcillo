from tkinter import *
from tkinter import messagebox
import nltk #Para procesamiento de lenguaje natural
from nltk.stem.lancaster import LancasterStemmer #Transformar palabras - quitar letras demas
stemmer = LancasterStemmer()
import numpy 
import tflearn
import tensorflow
import json
import random
import pickle # Guardar el modelo json

#nltk.download('punkt') #Validación - Descarga paquete en caso de que moleste al ejecutarlo

with open('./data/contenido/contenido.json', encoding='utf-8') as archivo:
    datos = json.load(archivo)

try:
    with open('./data/archivos/variables.pickle','rb') as archivoPickle:
        palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
except:

    palabras = []
    tags = []
    auxX = []
    auxY = []

    for contenido in datos['contenido']:
        for patrones in contenido['patrones']:
            auxPalabra = nltk.word_tokenize(patrones) #Toma la frase y la separa en palabras o tokens
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido['tag']) 

            if contenido['tag'] not in tags: #Se almacenan todos los tags mientras no esten dentro
                tags.append(contenido['tag'])


    palabras = [stemmer.stem(w.lower()) for w in palabras if w!='?'] #Casteo de palabra
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = []
    salida = []
    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)

        filaSalida = salidaVacia[:] 
        filaSalida[tags.index(auxY[x])] = 1 #Contenido de y que hay en indice y asigna un 1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    with open('./data/archivos/variables.pickle','wb') as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)

tensorflow.compat.v1.reset_default_graph()

red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,10)
red = tflearn.fully_connected(red,len(salida[0]),activation='softmax') # Predicciones
red = tflearn.regression(red)

modelo = tflearn.DNN(red)

modelo.fit(entrenamiento, salida, n_epoch= 1000, batch_size=10, show_metric=True) #Va a ver la información 1000 veces y cuantas entradas
modelo.save('./data/archivos/modelo.tflearn')

def mainBot(texto):   
    if texto!= '' and texto != ' ':
        chatLog.insert(END, "    ")
        chatLog.image_create(END, image = imgRU, pady=20)
        chatLog.insert(END, "  "+texto + '\n')

        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(texto)
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]

        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1

        resultados = modelo.predict([numpy.array(cubeta)]) #Predict probabilidad de 0 a 1
        resultadosIndices = numpy.argmax(resultados)
        tag = tags[resultadosIndices]

        for tagAux in datos['contenido']:
            if tagAux['tag'] == tag:
                respuesta = tagAux['respuestas']

        chatLog.insert(END, "    ")
        chatLog.image_create(END, image = imgRB)
        chatLog.insert(END, "  "+random.choice(respuesta) + '\n\n')

        if texto == 'Adios' or texto == 'Hasta la proxima' or texto == 'Chao' or texto == 'adios' or texto == 'chao' or texto == 'hasta la proxima':
            root.destroy()
    else:
        messagebox.showerror("Error", "Por favor ingrese un texto valido")

# Interfaz grafica
root = Tk()
root.title("ChatBot")
root.geometry("600x600")
root.resizable(width=False, height=False)

img = PhotoImage(file='./data/img/morcillo.png')
imgRB = PhotoImage(file='./data/img/morcillo_respuesta.png')
imgRU = PhotoImage(file='./data/img/user.png')

chatLog = Text(root, bd=0, bg="white", height="10", width="50", font="Helvetica", pady=10, )
scrollbar = Scrollbar(root, command=chatLog.yview, cursor="heart")  
chatLog['yscrollcommand'] = scrollbar.set
chatLog.config(foreground="#2C3E50", font=("Verdana", 12 ))
chatLog.pack(expand= 1, fill= BOTH)
frame = Frame(root)
frame.pack(fill="x")
Label(frame, image=img).pack(side=TOP, pady=10)
Label(frame, text="Comenta lo que necesitas al bot", font=("Helvetica", 12, "bold")).pack(anchor="center", pady=10)
texto = StringVar()
entradaTexto = Entry(frame)
entradaTexto.config(bd=5, font=("Verdana", 10), width=50, textvariable=texto)
entradaTexto.pack(padx=20)
botonEnviar = Button(frame, text="Enviar", font= 'currier 13 bold', width= 15 , fg= '#17202A', bg= '#A6B2B3', highlightbackground= 'black', highlightthickness = 2)
botonEnviar.config(bd=3, font=("Helvetica", 10, "bold"), command=lambda:mainBot(texto.get()))
botonEnviar.pack(pady=10)
root.mainloop()