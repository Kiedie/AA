import numpy as np # Calculo numérico
import sympy as sp # Cálculo simbólico
import pandas as pd

import matplotlib.pyplot as plt # Graficar
from matplotlib.colors import ListedColormap
from matplotlib import lines
from utils import *
# Inicializamos semilla
np.random.seed(1)
##############################################################################
############################### EJERCICIO 1.1 ################################
##############################################################################

def simula_unif(N, dim, rango):
    return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gauss(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

##############################################################################
############################### EJERCICIO 1.2 ################################
##############################################################################

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

def signo(x):
    if x >= 0:
        return 1
    return -1

def f(x, y, a, b):
    return signo(y - a*x - b)

def f(x,y):
    return y - a*x - b


def etiquetar(muestra, f, noise = False):
    """
    Función que realiza el etiquetado de una muestra de acuerdo al signo de la función f
    
    Params:
    ----------
        - muestra: np.ndarray,
            Array multidimensional en el que cada elemento es una tupla de coordenadas
        - f:
            función que develve el signo de la distancia de los puntos de la muestra a una 
            recta, la recta se le pasa con los parametros 'a' y 'b'
        - a: int
            pendiente de la recta
        - b: int
            ordenada en el origen de la recta
        - noise: boolean, default=False
            Indica si queremos añadir o no un 10% de ruido
            
    Return:
    ----------
        - etiquetas: np.ndarray
            Etiquetado realizado
    """
    
    # Etiquetamos los ejemplos de la muestra usando la función f
    etiquetas = np.array([signo(f(x[0],x[1])) for x in muestra])
    
    # Si queremos añadir un 10% de ruido a cada etiqueta
    if noise:
        # Tomamos los indices de las muestras que tienen etiquetas 1 y -1
        pos = np.array([i for i, y in enumerate(etiquetas) if y == 1]) 
        neg = np.array([i for i, y in enumerate(etiquetas) if y == -1]) 
        
        # Barajamos aleatoriamente los vectores de indices
        np.random.shuffle(pos)
        np.random.shuffle(neg)
        
        # Calculamos el número de etiquetas que debemos de cambaiar
        npos = int(len(pos)*0.1)
        nneg = int(len(neg)*0.1)
        
        # Cambiamos las etiquetas multiplicando por -1.
        # 1    * (-1)    = -1
        # (-1) * (-1)    =  1
        if len(pos)!=0:
            etiquetas[pos[:npos]]*=(-1)
        if len(neg)!=0:
            etiquetas[neg[:nneg]]*=(-1)
    
    # Devolvemos las etiquetas
    return etiquetas


##############################################################################
################################### PLOTS ####################################
##############################################################################

def plot_labels(muestra, labels, f = f, 
                   title = 'Etiquetas de la muestra separadas por la función f', 
                   rango = [-50,50],
                   axis_labels = ['Eje X','Eje Y'], 
                   colors      = ['purple','orange'],
                   figsize=(7,7)):
    
    # Instanciamos el marco
    plt.figure(figsize=figsize)
    
    # Etiquetamos los ejes
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    
    # Declaramos el mapa de colores para dividir luego en regiones
    colormap = ListedColormap([colors[0],colors[1]])
    
    # Evaluamos la función f, para ello me creo un grid primero y luego evaluo
    xx  = np.linspace(rango[0],rango[1],100)
    yy  = np.linspace(rango[0],rango[1],100)
    X,Y = np.meshgrid(xx,yy)
    Z   = f(X,Y)
    
    
    # Mostramos la curva de nivel en Z=0 del clasificador (recta)
    plt.contour(X,Y,Z,[0],colors='black') 
    # Colorea las regiones con colores 
    # alpha indica la transparencia
    plt.contourf(X,Y,Z,0,cmap=colormap,alpha=0.4) 

    # Hasta ahora hemos mostrado la recta, y las regiones por colores
    # Ahora ponemos los puntos etiquetados por colores
    
    # Agrupamos las etiquetas en dos grupos
    pos = np.array([a for a,b in zip(muestra,labels) if b == 1])
    neg = np.array([a for a,b in zip(muestra,labels) if b == -1])

    # Visualizamos los datos, poniendo un color diferente a cada clase
    scatter1=plt.scatter(pos[:,0], pos[:,1], c="green",label='Label 1',alpha=0.75)
    scatter2=plt.scatter(neg[:,0], neg[:,1], c="crimson",label='Label -1',alpha=0.75)

    # Creamos un gráfico en 2D que no muestre nada para poder poner leyenda,
    # ya que la leyenda no admite el tipo devuelto por la función contour
    # Codigo sacado de Stackoverflow.com
    line_proxy = lines.Line2D([0],[0], linestyle="none", c='black', marker = '_')
    pos_proxy  = lines.Line2D([0],[0], linestyle="none", c='orange', marker = 's')
    neg_proxy  = lines.Line2D([0],[0], linestyle="none", c='purple', marker = 's')
    
    plt.legend([scatter1,scatter2,line_proxy, pos_proxy, neg_proxy], ['Label 1','Label -1', 
            "f(x,y)=0","f(x,y)>0","f(x,y)<0"], numpoints = 1,framealpha=0.25)
    plt.title(title)
    plt.show()
    
    
    
def plot_plain_3D(w,f,X,y,title,axis_labels,legend_labels,zlim,view,using_w = True):
    '''
    Función que visualiza en un espacio los puntos usados en el entrenamiento junto 
    con el plano generado generado por los pesos tras la regresión lineal.
    
    Nota: Función pensada para el ejercicio 2, aunque se puede generalizar 
    
    @Params:
        - w: vector de pesos
        - f: función que clasifica (gráfica)
        - x: Datos de entrenamienacco  [1,X_1,X_2]
        - y: etiquetas
        - title: título del plot
        - axis_labels: etiquetas de los ejes
        - legend_labels: etiquetas de la leyenda
        - using_w: boolean, default = True
            Especifica si la superficie la vamos crear a partir de los w
            o a partir de la gráfica f (parámetro f)
        
    '''
    # Creamos la figura y los ejes
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111,projection='3d')
    
    # Pintamos los datos, para ello
    # identificamos las coordenadas según la clase
    # pintamos los puntos de dicha clase de un color
    for j,c in enumerate(np.unique(y)):
        
        ax_x = [x[1] for i,x in enumerate(X) if y[i] == c] # eje x
        ax_y = [x[2] for i,x in enumerate(X) if y[i] == c] # eje y 
        
        ax.scatter(ax_x,ax_y,c,color=['darkorchid','black'][j],alpha=0.8)
        
    
    # etiquetamos los ejes
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    
    # Determinamos los extremos de los valores para luego construir un grid con las proporciones adecaudas
    min_int = np.min(X[:,1])
    max_int = np.max(X[:,1])
    min_sim = np.min(X[:,2])
    max_sim = np.max(X[:,2])
    
    # Declaramos un grid de puntos equiespaciados para evaluar la función
    xx = np.linspace(min_int,max_int,50)
    yy = np.linspace(min_sim,max_sim,50)
    xx, yy = np.meshgrid(xx,yy)
    
    # Evaluamos en los puntos de la malla la función lineal obtenida 
    if using_w:
        z = np.array(w[0]+xx*w[1]+yy*w[2])
    else:
        z = np.array(f(xx,yy))
    # Visualizamos el hiperplano 
    ax.plot_surface(xx,yy, z, color='aqua',alpha=0.6)
    #ax.set_zlim(zlim)
    
    # Creamos un gráfico en 2D que no muestre nada para poder poner leyenda,
    # debido a que la leyenda no admite el tipo devuelto por un scatter 3D,
    # https://stackoverflow.com/questions/20505105/add-a-legend-in-a-3d-scatterplot-with-scatter-in-matplotlib
    scatter_proxy1 = lines.Line2D([0],[0], linestyle="none", c='black', marker = 'o')
    scatter_proxy2 = lines.Line2D([0],[0], linestyle="none", c='darkorchid', marker = 'o')
    plane_proxy = lines.Line2D([0],[0], linestyle="none", c='aqua', marker = '_')
    ax.legend([scatter_proxy1,scatter_proxy2, plane_proxy], legend_labels, numpoints = 1, loc='upper left')
    
    #Cambiamos la posición del gráfico para que se vean mejor los datos y el hiperplano
    #https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib
    ax.view_init(view[0],view[1]) 
    
    plt.title(title)
    plt.show()