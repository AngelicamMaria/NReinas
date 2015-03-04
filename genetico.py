#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
genetico.py
------------

Este modulo incluye el algoritmo genérico para algoritmos genéticos, así como un
algoritmo genético adaptado a problemas de permutaciones, como el problema de las
n-reinas o el agente viajero.

Como tarea se pide desarrollar otro algoritmo genético con el fin de probar otro tipo
de métodos internos, así como ajustar ambos algortmos para que funcionen de la mejor
manera posible.


Para que funcione, este modulo debe de encontrarse en la misma carpeta que blocales.py
y nreinas.py vistas en clase.

"""

__author__ = 'Escribe aquí tu nombre'

import nreinas
import random
import time
from itertools import permutations
from itertools import combinations

class Genetico:
    """
    Clase genérica para un algoritmo genético.
    Contiene el algoritmo genético general y las clases abstractas.

    """

    def busqueda(self, problema, n_poblacion=10, n_generaciones=30, elitismo=True):
        """
        Algoritmo genético general

        @param problema: Un objeto de la clase blocal.problema
        @param n_poblacion: Entero con el tamaño de la población
        @param n_generaciones: Número de generaciones a simular
        @param elitismo: Booleano, para aplicar o no el elitismo

        @return: Un estado del problema

        """
        poblacion = [problema.estado_aleatorio() for _ in range(n_poblacion)]

        for _ in range(n_generaciones):

            aptitud = [self.calcula_aptitud(individuo, problema.costo) for individuo in poblacion]

            elite = min(poblacion, key=problema.costo) if elitismo else None

            padres, madres = self.seleccion(poblacion, aptitud)

            poblacion = self.mutacion(self.cruza_listas(padres, madres))

            poblacion = poblacion[:n_poblacion]

            if elitismo:
                poblacion.append(elite)

        e = min(poblacion, key=problema.costo)
        return e

    def calcula_aptitud(self, individuo, costo=None):
        """
        Calcula la adaptación de un individuo al medio, mientras más adaptado mejor, por default
        es inversamente proporcionl al costo (mayor costo, menor adaptción).

        @param individuo: Un estado el problema
        @param costo: Una función de costo (recibe un estado y devuelve un número)

        @return un número con la adaptación del individuo
        """
        # return max(0, len(individuo) - costo(individuo))
        return 1.0 / (1.0 + costo(individuo))

    def seleccion(self, poblacion, aptitud):
        """
        Seleccion de estados

        @param poblacion: Una lista de individuos

        @return: Dos listas, una con los padres y otra con las madres.
        estas listas tienen una dimensión int(len(poblacion)/2)

        """
        raise NotImplementedError("¡Este metodo debe ser implementado por la subclase!")

    def cruza_listas(self, padres, madres):
        """
        Cruza una lista de padres con una lista de madres, cada pareja da dos hijos

        @param padres: Una lista de individuos
        @param madres: Una lista de individuos

        @return: Una lista de individuos

        """
        hijos = []
        for (padre, madre) in zip(padres, madres):
            hijos.extend(self.cruza(padre, madre))
        return hijos

    def cruza(self, padre, madre):
        """
        Cruza a un padre con una madre y devuelve una lista de hijos, mínimo 2
        """
        raise NotImplementedError("¡Este metodo debe ser implementado por la subclase!")

    def mutacion(self, poblacion):
        """
        Mutación de una población. Devuelve una población mutada

        """
        raise NotImplementedError("¡Este metodo debe ser implementado por la subclase!")


class GeneticoPermutaciones1(Genetico):
    """
    Clase con un algoritmo genético adaptado a problemas de permutaciones

    """

    def __init__(self, prob_muta=0.01):
        """
        @param prob_muta : Probabilidad de mutación de un cromosoma (0.01 por defualt)

        """
        self.prob_muta = prob_muta
        self.nombre = 'propuesto por el profesor con prob. de mutación ' + str(prob_muta)

    def seleccion(self, poblacion, aptitud):
        """
        Selección por torneo.


        """
        padres = []
        baraja = range(len(poblacion))
        random.shuffle(baraja)
        for (ind1, ind2) in [(baraja[i], baraja[i + 1]) for i in range(0, len(poblacion) - 1, 2)]:
            ganador = ind1 if aptitud[ind1] > aptitud[ind2] else ind2
            padres.append(poblacion[ganador])

        madres = []
        random.shuffle(baraja)
        for (ind1, ind2) in [(baraja[i], baraja[i + 1]) for i in range(0, len(poblacion) - 1, 2)]:
            ganador = ind1 if aptitud[ind1] > aptitud[ind2] else ind2
            madres.append(poblacion[ganador])

        return padres, madres

    def cruza(self, padre, madre):
        """
        Cruza especial para problemas de permutaciones

        @param padre: Una tupla con un individuo
        @param madre: Una tupla con otro individuo

        @return: Dos individuos resultado de cruzar padre y madre con permutaciones

        """
        hijo1, hijo2 = list(padre), list(madre)
        corte1 = random.randint(0, len(padre) - 1)
        corte2 = random.randint(corte1 + 1, len(padre))
        for i in range(len(padre)):
            if i < corte1 or i >= corte2:
                hijo1[i], hijo2[i] = hijo2[i], hijo1[i]
                while hijo1[i] in padre[corte1:corte2]:
                    hijo1[i] = madre[padre.index(hijo1[i])]
                while hijo2[i] in madre[corte1:corte2]:
                    hijo2[i] = padre[madre.index(hijo2[i])]
        return [tuple(hijo1), tuple(hijo2)]

    def mutacion(self, poblacion):
        """
        Mutación para individus con permutaciones. Utiliza la variable local self.prob_muta

        @param poblacion: Una lista de individuos (tuplas).

        @return: Los individuos mutados

        """
        poblacion_mutada = []
        for individuo in poblacion:
            individuo = list(individuo)
            for i in range(len(individuo)):
                if random.random() < self.prob_muta:
                    k = random.randint(0, len(individuo) - 1)
                    individuo[i], individuo[k] = individuo[k], individuo[i]
            poblacion_mutada.append(tuple(individuo))
        return poblacion_mutada


################################################################################################
# AQUI EMPIEZA LO QUE HAY QUE HACER CON LA TAREA
################################################################################################

class GeneticoPermutaciones2(Genetico):
    """
    Clase con un algoritmo genético adaptado a problemas de permutaciones

    """

    def __init__(self, prob_muta=0.01):
        """
        Aqui puedes poner algunos de los parámetros que quieras utilizar en tu clase

        """
        self.nombre = 'Angelica Maldonado'
        #
        # ------ IMPLEMENTA AQUI TU CÓDIGO ------------------------------------------------------------------------
        #
        self.prob_muta = prob_muta
        self.nombre = 'propuesto por el profesor con prob. de mutación ' + str(prob_muta)

    def calcula_aptitud(self, individuo, costo=None):
        """
        Desarrolla un método específico de medición de aptitud.

        """
    ####################################################################
    #                          20 PUNTOS
    ####################################################################


        matriz = []
        reinas = len(individuo)
        for i in range(reinas):
            matriz.append([])
            for j in range(reinas):
                matriz[i].append(None)
        i = 0
        j = 0
        t = 0
        for i in 8:
            for j in 8:
                matriz[i][j] = individuo[t]
                t += 1
        i = 0
        for i in reinas:
            x = 0
            y = 0
            for x in reinas:
                for y in reinas:
                    if matriz[x][y] == i:
                        #arriba y abajo
                        if x > 0 and x < reinas:
                            j = x
                            while j >= 0:
                                if 0 <= matriz[j][y] < reinas and matriz[j][y] != i:
                                    costo += 1
                                    j -= 1

                            j = x
                            while j < reinas:
                                if 0 <= matriz[j][y] < reinas and matriz[j][y] != i:
                                    costo += 1
                                    j += 1
                        if x == 0:
                            j = x
                            while j < reinas:
                                if 0 <= matriz[j][y] < reinas and matriz[j][y] != i:
                                    costo += 1
                                    j += 1
                        if x == 7:
                            j = x
                            while j >= 0:
                                if 0 <= matriz[j][y] < reinas and matriz[j][y] != i:
                                    costo += 1
                                    j -= 1

                        # izquierda Derecha
                        if y == 0:
                            j = y
                            while j < reinas:
                                if 0 <= matriz[x][j] < reinas and matriz[x][j] != i:
                                    costo += 1
                                    j += 1
                        if y == reinas:
                            j = y
                            while j >= 0:
                                if 0 <= matriz[x][j] < reinas and matriz[x][j] != i:
                                    costo += 1
                                    j -= 1
                        if y > 0 and y < reinas:
                            j = y
                            while j >= 0:
                                if 0 <= matriz[x][j] < reinas and matriz[x][j] != i:
                                    costo += 1
                                    j -= 1

                            j = y
                            while j < reinas:
                                if 0 <= matriz[x][j] < reinas and matriz[x][j] != i:
                                    costo += 1
                                    j += 1
                        #Diagonales
                        j = x
                        t = y
                        while j >= 0:
                            if 0 <= matriz[j][t] < reinas and matriz[j][t] != i:
                                costo += 1
                                j -= 1
                                t -= 1
                                if t == 0:
                                    j = -1
                        j = x
                        t = y
                        while j >= 0:
                            if 0 <= matriz[j][t] < reinas and matriz[j][t] != i:
                                costo += 1
                                j -= 1
                                t += 1
                                if t == reinas:
                                    j = -1
                        j = x
                        t = y
                        while j < reinas:
                            if 0 <= matriz[j][t] < reinas and matriz[j][t] != i:
                                costo += 1
                                j += 1
                                t -= 1
                                if t == 0:
                                    j = 9
                        j = x
                        t = y
                        while j < reinas:
                            if 0 <= matriz[j][t] < reinas and matriz[j][t] != i:
                                costo += 1
                                j += 1
                                t += 1
                                if t == reinas:
                                    j = 9
        if costo > 0:
            return costo
        return 0
        #raise NotImplementedError("¡Este metodo debe ser implementado!")

    def seleccion(self, poblacion, aptitud):
        """
        Desarrolla un método específico de selección.

        """
    #####################################################################
    #                          20 PUNTOS
    #####################################################################
    #
    # ------ IMPLEMENTA AQUI TU CÓDIGO ----------------------------------
    #
        tam = len(poblacion)
        torneo1 = [tam]
        torneo2 = [tam]
        vector1 = [tam]
        vector2 = [tam]
        i = 0
        for i in tam:
            torneo1[i] = -3
            torneo2[i] = -3
        i = 0
        x = 0
        for i in tam:
            for x in not range(torneo1):
                torneo1[i] = x
        i = 0
        x = 0
        for i in tam:
            for x in not range(torneo1):
                torneo2[i] = x

        x = 0
        y = 0
        pose = 0
        if tam % 2 == 0:
            for i in tam:
                if i % 2 == 0:
                    x = torneo1[i]
                    y = torneo1[i+1]
                    if aptitud[x] >= aptitud[y]:
                        vector1[pose] = x
                    else:
                        vector1[pose] = y
                    pose += 1
            for i in tam:
                if i % 2 == 0:
                    x = torneo2[i]
                    y = torneo2[i+1]
                    if aptitud[x] >= aptitud[y]:
                        vector2[pose] = x
                    else:
                        vector2[pose] = y
                    pose += 1
        if tam %2 != 0:
            for i in tam:
                if i % 2 != 0:
                    x = torneo1[i]
                    y = torneo1[i-1]
                    if aptitud[x] >= aptitud[y]:
                        vector1[pose] = x
                    else:
                        vector1[pose] = y
            for i in tam:
                if i % 2 != 0:
                    x = torneo2[i]
                    y = torneo2[i-1]
                    if aptitud[x] >= aptitud[y]:
                        vector2[pose] = x
                    else:
                        vector2[pose] = y
        cruza(self, vector1, vector2)

#    raise NotImplementedError("¡Este metodo debe ser implementado!")


    def cruza(self, padre, madre):
        """
        Cruza especial para problemas de permutaciones

        @param padre: Una tupla con un individuo
        @param madre: Una tupla con otro individuo

        @return: Dos individuos resultado de cruzar padre y madre con permutaciones

        """
        hijo1, hijo2 = list(padre), list(madre)
        corte1 = random.randint(0, len(padre) - 1)
        corte2 = random.randint(corte1 + 1, len(padre))
        for i in range(len(padre)):
            if i < corte1 or i >= corte2:
                hijo1[i], hijo2[i] = hijo2[i], hijo1[i]
                while hijo1[i] in padre[corte1:corte2]:
                    hijo1[i] = madre[padre.index(hijo1[i])]
                while hijo2[i] in madre[corte1:corte2]:
                    hijo2[i] = padre[madre.index(hijo2[i])]
        return [tuple(hijo1), tuple(hijo2)]


    def mutacion(self, poblacion):
        """
        Desarrolla un método específico de mutación.

        """

    ###################################################################
    #                          20 PUNTOS
    ###################################################################
    #
    # ------ IMPLEMENTA AQUI TU CÓDIGO --------------------------------
    # el metodo de mutacion que se ha realisado es el intercambio de 2 puntos diferentes
        poblacion_mutada = []
        for individuo in poblacion:
            individuo = list(individuo)
            for i in range(len(individuo)):
                PM = random.random()
                if PM < self.prob_muta:
                    x = random.randint(0, len(individuo) - 1)
                    y = x
                    while x == y:
                        x = random.randint(0, len(individuo) - 1)

                    aux = individuo[x]
                    individuo[x] = individuo[y]
                    individuo[y] = aux

            poblacion_mutada.append(tuple(individuo))
        return poblacion_mutada



#    raise NotImplementedError("¡Este metodo debe ser implementado!")


    def prueba_genetico_nreinas(algo_genetico, problema, n_poblacion, n_generaciones):
        tiempo_inicial = time.time()
        solucion = algo_genetico.busqueda(problema, n_poblacion, n_generaciones, elitismo=True)
        tiempo_final = time.time()
        print "\nUtilizando el algoritmo genético " + algo_genetico.nombre
        print "Con poblacion de dimensión ", n_poblacion
        print "Con ", str(n_generaciones), " generaciones"
        print "Costo de la solución encontrada: ", problema.costo(solucion)
        print "Tiempo de ejecución en segundos: ", tiempo_final - tiempo_inicial
        return solucion


if __name__ == "__main__":
    #################################################################################################
    #                          20 PUNTOS
    #################################################################################################
    # Modifica los valores de la función siguiente (o el parámetro del algo_genetico)
    # buscando que el algoritmo encuentre SIEMPRE una solución óptima, utilizando el menor tiempo
    # posible en promedio. Realiza esto para las 8, 16 y 32 reinas.
    #   -- ¿Cuales son en cada caso los mejores valores (escribelos abajo de esta lines)
    #
    #
    #   -- ¿Que reglas podrías establecer para asignar valores segun tu experiencia
    #

    #solucion = prueba_genetico_nreinas(algo_genetico=GeneticoPermutaciones1(0.05),
     #                                  problema=nreinas.ProblemaNreinas(16),
      #                                 n_poblacion=32,
       #                                n_generaciones=100)
    #print solucion

    #################################################################################################
    #                          20 PUNTOS
    #################################################################################################
    # Modifica los valores de la función siguiente (o los posibles parámetro del algo_genetico)
    # buscando que el algoritmo encuentre SIEMPRE una solución óptima, utilizando el menor tiempo
    # posible en promedio. Realiza esto para las 8, 16 y 32 reinas.
    #   -- ¿Cuales son en cada caso los mejores valores (escribelos abajo de esta lines)
    #
    #
    #   -- ¿Que reglas podrías establecer para asignar valores segun tu experiencia? Escribelo aqui
    #   abajo, utilizando tnto espacio como consideres necesario.
    #
    # Recuerda de quitar los comentarios de las lineas siguientes:

    solucion = prueba_genetico_nreinas(algo_genetico=GeneticoPermutaciones2(0.05),
                                       problema=nreinas.ProblemaNreinas(16),
                                       n_poblacion=32,
                                       n_generaciones=500)
    print solucion