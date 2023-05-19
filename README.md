# Algoritmo Genetico - Flappy Bird

En este proyecto, exploramos la poderosa aplicación de un algoritmo genético en la resolución del desafiante juego Flappy Bird. Mediante la representación de las aves como individuos en una población y la aplicación de operadores genéticos como selección, cruzamiento y mutación, buscamos encontrar la combinación óptima de características y comportamientos que maximicen la habilidad de evitar obstáculos y obtener puntuaciones altas. A través de la optimización iterativa basada en la evaluación de fitness y la evolución generacional, el algoritmo genético converge hacia soluciones cada vez mejores. Descubre cómo esta técnica matemática nos permite abordar problemas complejos y lograr resultados impresionantes en el contexto del Flappy Bird.

<img src="https://github.com/JavierAM01/Machine-Learnig-in-Games/blob/main/images/ai/flappybird.gif" height="400">

## Ejecución del programa

 - Jugar al juego

```
python main.py -play
```

 - Ver a la IA jugar.

```
python main.py -ai
```

 - Entrenar un modelo

```
python main.py -ai -train
```

## Red Neuronal

La red neuronal recibirá 5 valores de entrada como información disponible:

 1) Distancia al punto más alto (top) de entrada entre las dos columnas.
 2) Distancia al punto más bajo (bottom) de entrada entre las dos columnas.
 3) Distancia horizontal a la columna.
 4) Altura (eje y) con respecto al suelo.
 5) Velocidad.

Una vez aportados estos datos serán procesados por un red de tipo perceptrón de una sola capa. La estructura de creación en pytorch es la siguiente:

```python
        # architecture 
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 1)
```

Podemos ver que la red es muy sencilla, pero obtiene muy buenos resultados con los pesos apropiados.

## Algoritmo genético

Esta parte está completamente desarrollada en el script **genetic_algorithm.py**. Para la realización del [algoritmo genético](https://es.wikipedia.org/wiki/Algoritmo_gen%C3%A9tico) usamos la red neuronal ya mencionada. Apartir de esta, generamos distintas generaciones de pájaros y nos quedamos con las mejores versiones. Una vez las tenemos guardadas, en cada generación nueva las mezclamos con el fin de conseguir mejores resultados.

Por ejemplo uno de los posibles cambios es el siguiente:

```python
        def change2(params):
            if len(index) == 1:
                a = index[0]
                params[key][a]      *= (1 + (0.01 * (-1)**random.randint(0,1)))
            elif len(index) == 2:
                a, b = index[0], index[1]
                params[key][a,b]    *= (1 + (0.01 * (-1)**random.randint(0,1)))
            elif len(index) == 3:
                a, b, c = index[0], index[1], index[2]
                params[key][a,b,c]  *= (1 + (0.01 * (-1)**random.randint(0,1)))
```

en el cual tratamos de mover un cierto rango los valores actuales, con el fin de hacercanos a máximos locales. Sería como un uso de la idea del descenso del gradiente, pero en este caso nos movemos en drecciones aleatorias.

Otra posible modificación es la mutaciones de 2 redes. En este caso cambios un conjunto de parámetros con los de otra red:

```python
        def change(params, new_params):
            if len(index) == 1:
                a = index[0]
                params[key][a] = new_params[key][a]
            elif len(index) == 2:
                a, b = index[0], index[1]
                params[key][a,b] = new_params[key][a,b]
            elif len(index) == 3:
                a, b, c = index[0], index[1], index[2]
                params[key][a,b,c] = new_params[key][a,b,c]
```

## Pygame

Para la creación del juego se ha usado pygame, en este repositorio nos centramos en la creación del algoritmo genético, por lo que si quieren más información sobre esta librería pueden acceder a la documentación [aquí](https://www.pygame.org/docs/). Para contenido similar de algortimos y juegos pueden verlo [aquí](https://github.com/JavierAM01/Machine-Learnig-in-Games).  
