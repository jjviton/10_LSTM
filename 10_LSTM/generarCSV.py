# -*- coding: utf-8 -*-


"""
******************************************************************************
Clase para generar CSV con datos de Instrumentos 
******************************************************************************
******************************************************************************


Mejoras:    

Started on DIC/2022
Version_1: 

Objetivo: 

Author: J3Viton

"""

# J3_DEBUG__ = False  #variable global (global J3_DEBUG__ )



################################ IMPORTAMOS MODULOS A UTILIZAR.
import pandas as pd
import numpy as np
import datetime as dt
#import pandas_datareader as web
import yfinance as yf

from pykalman import KalmanFilter

import sys
sys.path.insert(0,"C:\\Users\\INNOVACION\\Documents\\J3\\100.- cursos\\Quant_udemy\\programas\\Projects\\libreria")
import quant_j3_lib as quant_j

#### Variables globales  (refereniarlas con 'global' desde el codigo
versionVersion = 1
globalVar  = True



#################################################### Clase Estrategia 

class dataGenClass:

    """CLASE ESTRATEGIA

       
    """  
    
    
    
    #Variable de CLASE
    backtesting = False  #variable de la clase, se accede con el nombre

    
    def __init__(self, instrumento= 'IBE.MC', para1=False, para2=1):
        
        #Variable de INSTANCIA
        self.para_02 = para2   #variable de la isntancia
        self.instrumento =instrumento
 
        
        globalVar = False
        
        return
    
        
    def analisis(self, instrumento, startDate, endDate, DF):
        """
        Descripcion: this method evaluates the convenience of the inversion measuring Profit and loss
        Currrent method asumes profit line three times bigger than stopLoss line
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------


        """
        pass
   
        return
    
    def creaCSV_01 (self, instrumento, startDate, endDate):
        
        try: 
            dff = yf.download(instrumento, startDate,endDate)
        except:
            logging.info('Ticker no existe'+instrumento)
            return dff    #Irá empty
        if dff.empty:
            return dff
        
        dff=self.featured_Data(dff)
        
        
        return dff

    def featured_Data (self, dff):
        """
        Descripcion: this method calculates the extra features of the eviroment observation. 
        Aqui es donde radica parte del arte de este poryecto, como diseñar la estrategia y los rewards.
        En definitiva enriquecemos el dataSet.
        
        Parameters
        ----------
            df: 
        Returns
        -------
            TYPE
                DESCRIPTION.
        """
   
        #Calculo las Exponencial Moving Average 200, 50, 30
        #dff= quant_j.ExponentialMovingAverage(dff, 200,short_=50)
        dff= quant_j.ExponentialMovingAverage(dff, 100,short_=30)
           
        # Calculo el media exponencial del Volumen de los ultimos dias
        #df_aux =dff.loc[:,['Volume','Open']] #si paso una sola hace una serie y rompe todo
        #df_aux.rename(columns={'Volume': 'Close'}, inplace=True)    #cambio el nombre porque la libreria busca 'close a piñon'   
        #df_aux= quant_j.ExponentialMovingAverage(df_aux)
        #df_aux.rename(columns={'EMA_30': 'VolEMA_30'}, inplace=True) #Coloco el buen nombre
        #del df_aux['EMA_200']
        #dff['VolEMA_30']= df_aux['VolEMA_30']  
        #j# dff.dropna(inplace=True)        
        #dff['DeltaVol_EMA']=dff['Volume']-dff['VolEMA_30']  # Calculo si el volumen está por cima de la media :-)


        # El amigo Kalman
        
        # 1.- CALCULAMOS EL FILTRO DE KALMAN
        # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [1],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=.01)
        state_meansS, _ =kf.smooth(dff.Close)
        state_meansS = pd.Series(state_meansS.flatten(), index=dff.index)
        dff['Kalman'] = state_meansS
        
        # Regresion lineal ultimas 50 sesiones
        close_ =dff.columns.get_loc("Close") 
        slice01= dff.iloc[-50:,close_]
        coef_, intercept_= quant_j.linearRegresion_J3(slice01)
               
        ################### Media Hull
        dff['hull']=quant_j.HMA(dff['Close'], 20)
        
        """
        # SUPERVISED: Un poco de supervisado  https://www.aprendemachinelearning.com/pronostico-de-series-temporales-con-redes-neuronales-en-python/
        ##dias = 1  # 1= cierre de mañana
        dff['Futuro'] = dff['Open'].shift((-1)*dias)

        dff.dropna(inplace=True)        #quito los CEROS
        """
        
        # Coloco un indice natural y incremental
        dff['position']=dff['Close']   #columna fake
        position_ =dff.columns.get_loc("position")
        for i in range(len(dff)):
            dff.iloc[i,position_] = int( i)
        # Cambio el indice del Dataframe a position
        dff.set_index('position', drop=False,inplace=True, append =True)  
        
        ## Calculo el dia de la semana Lunes=0
        dff['dia']=1
        dia_ =dff.columns.get_loc("dia")
        for i in range(len(dff)):
            temp, pos= dff.index[i]
            dff.iloc[i,dia_]=temp.dayofweek
            
        ## Calculo la media del Volumen
        ##fm["mv_avg_12"]= dfm["Open"].rolling(window=12).mean().shift(1)
        dff["MA_Vol"]= dff["Volume"].rolling(window=60).mean()   #.shift(1)
            
        #Limpio columnas 
        del dff['Open']
        del dff['High']
        del dff['Low']
        del dff['Adj Close']
        del dff['position']
        
        dff['Volume']
        #del dff['VolEMA_30']
        #del dff['DeltaVol_EMA']
        dff.dropna(inplace=True)        #quito los CEROS
       
        #Quito el valor de Close para simplificar el trabajo de la red 
        dff['Close']=0  
       
        print(dff.head())
        ##quant_j.salvarExcelTOTEST(dff, 'telefonica_01', nombreSheet="data")  
        
        return dff

    
#################################################### Clase FIN






#/******************************** FUNCION PRINCIPAL main() *********/
#     def main():   
if __name__ == '__main__':    
    """Esta parte del codigo se ejecuta cuando llamo tipo EXE
        
    Parámetros:
    a -- 
    b -- 
    c -- 
    
    Devuelve:
    Valores 

    Excepciones:
    ValueError -- Si (a == 0)
    
    """   
    print ('version: ',versionVersion) 
    
    myGenerar = dataGenClass()    #Creamos la clase
    
  
    #### FECHAS
    #start =dt.datetime(2000,1,1)
    startD =dt.datetime.today() - dt.timedelta(days=500)    #un año tiene 250 sesiones.
    #end = dt.datetime(2019,10,1)
    endD= dt.datetime.today()  - dt.timedelta(days=1)        #Quito hoy para no ver el valor al ejecutar antes del cierre
    #end = '2021-9-19'
    
    myGenerar.creaCSV_01('aapl', startD, endD)
    
    
else:
    """Esta parte del codigo se ejecuta si uso como libreria"""    
    print ('formato libreria')
    print ('version: ',versionVersion)    