# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:46:26 2021

@author: brice
"""

'''
Brute force :
On constate qu'il y'a 100 x 20 x10^4 = 2 x 10^8 combinaisasons possibles
'''

import numpy as np
import pandas as pd
import random 


def getdata():
    df = pd.read_csv('IA_tp6_data.csv', header=None)
    df.columns=['x','y']
    return df


def erreur_modele(ens:pd.DataFrame,m,b):
    y_predit = ens['x']*m+b
    mse = 1/len(ens) * sum((y_predit-ens['y'])**2)
    return mse


def viz(ens,m,b):
    ens['y_predit'] = ens['x']*m+b
    ax = ens.plot(x='x',y='y', kind='scatter')
    ens.plot(x='x',y='y_predit', kind='line', color='red', ax=ax)
    
    
def brute_force():
    df = getdata()
    allmodel = {(m,b):erreur_modele(df,m,b) for m in np.arange(-50,50,10**-1) for b in np.arange(-10,10,10**-1) }
    print('longeur :', len(allmodel))
    best_model = min(allmodel, key=lambda x: allmodel.get(x))
    m,b = best_model
    print(f'm = {m:.2f} et b = {b:.2f} pour une erreur de {erreur_modele(df,m,b)} ')
    viz(df, m, b)



def step_gradient(m,b,alpha):
    
    data = getdata()
    n = len(data)
    
    grad_m = lambda m,b: -2/n * sum(data['x']*(data['y']-(m * data['x'] + b)))
    grad_b = lambda m,b: -2/n * sum(data['y']-(m * data['x'] + b))
    
    m_maj = m - alpha*grad_m(m, b)
    b_maj = b - alpha*grad_b(m, b)
    
    return m_maj, b_maj



def gradient_descent(nbiteration = 20, alpha = 0.0001, form = True):
    
    data = getdata()
    m = random.uniform(-50,50)
    b = random.uniform(-10,10)
    new_m = 0
    new_b = 0
    
    
    if form:
        
        for i in range(nbiteration):
            print(f'm = {m:.2f} et b = {b:.2f} et erreur = {erreur_modele(data, m, b):.2f}')
            m,b = step_gradient(m, b, alpha)
            viz(data, m, b)
            
    else : 
        
        while abs(erreur_modele(data, m, b) - erreur_modele(data, new_m, new_b)) > 10**-2:
            print(f'm = {m:.2f} et b = {b:.2f} et erreur = {erreur_modele(data, m, b):.2f}')
            new_m,new_b = m,b
            m,b = step_gradient(m, b, alpha)
            viz(data, m, b)
        
    
    print("\nLe meilleur modèle / Best fit :")
    print(f'm = {m:.2f} | b = {b:.2f} avec erreur de {erreur_modele(data,m,b):.2f} ')

        
if __name__ == '__main__':
    choice = input('Brut force : 1 | Grandient descent : 2\nchoice : ')
    
    if choice == '1':
        print("With brut force : ")
        brute_force()
        
    else:
        
        print("With grandient descent : ")
        #gradient_descent(form=True)
        gradient_descent(form=False)
    
    
        