# -*- coding: utf-8 -*-


"""
******************************************************************************
Clase que implementa una LSTM para evaluar la serie temporal de una secuencia
 de precios y mostrar la prevision calculada.
 
Como objetivo tenemos entrenar una red LSTM para que descubra patrones esta-
cionales que nos permitan hacer trading de exito.
El proyecto 100_backtrading, permite evaluar la estrategia antes de pasar a PROD
Luego entrenamos la red con todos los datos disponibles hasta la fecha y 
salvamos el modelos para ponerle a operar diariamente y que nos dé señales.
Al comienzo creo que haremos las entradas y el moneymangement a mano.

Todo apunta que esta formula va a dar buenos resultados, espero no equivocarme 
o que lo que vemos a feb-23 no sea un espejismo o fruto de una buena racha.
 
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
import datetime as dt
import yfinance as yf

#Import AI
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


import tensorflow as tf

#Import trading
from pykalman import KalmanFilter

import sys
sys.path.insert(0,"C:\\Users\\INNOVACION\\Documents\\J3\\100.- cursos\\Quant_udemy\\programas\\Projects\\libreria")

#Mis import
import quant_j3_lib as quant_j
import generarCSV as fileCSV


####################### LOGGING
import logging    #https://docs.python.org/3/library/logging.html
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../log/registro.log', level=logging.INFO ,force=True,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.warning('esto es una kkk')

#### Variables globales  (refereniarlas con 'global' desde el codigo
versionVersion = 1.1
globalVar  = True

###################### DATOS
from sp500 import tickers_sp500
from nasdaq import tickers_nasdaq
from ibex import tickers_ibex
from eurostoxx import tickers_eurostoxx
from comodity import tickers_comodities


pdf_flag =True
epochs_ =10

#################################################### Clase Estrategia 



class LSTMClass:

    """CLASE ESTRATEGIA

       
    """  
    
    #Variable de CLASE
    backtesting = False  #variable de la clase, se accede con el nombre
    n_past = 14  # Number of past days we want to use to predict the future.  FILAS
    flag01 =0
   
    def __init__(self, previson_a_x_days=3, Y_supervised_ = 'hull', para1=False, para2=1):
        
        #Variable de INSTANCIA
        self.para_02 = para2   #variable de la isntancia
        
        self.trainXX = []
        self.trainYY = []
        self.trainX_test=[]
        self.trainX = []
        self.trainY = []
        
        self.dfx = pd.DataFrame()
        self.cols =0
        self.Y_supervised = Y_supervised_ 

        self.n_future=previson_a_x_days
        
        self.df_previsiones_xd = pd.DataFrame(columns=['X_dias'])
        
        
        globalVar = True
        LSTMClass.flag01 =True
        
        return
    
        
    def analisis(self, instrumento, startDate, endDate, DF):
        """
        Descripcion: sample method
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------


        """
        pass
   
        return
    
    def estrategia_LSTM_01(self, instrumento_, startDate_, endDate_):
        """
        Descripcion: parto de los datos de reales y las predicciones de la LSTM, y defino una estrategia.
        Atencion que desplazo en array de previsiones para que coincida el una misma vertical el valor real con la prevision
        
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------

        """
        ##  RED
        
        #Preparo los datos
        self.dataPreparation_1(instrumento_,startDate_, endDate_)
        #creo y entreno la NET
        self.LSTM_net_2()
        # Pinto la grafica
        #self.plottingSecuence_prevision(self)        
        df_predi2= self.predicionLSTM(instrumento_) #predicciones del grupo de test
        df_gap= pd.DataFrame(columns=['X_dias'],index=range(self.n_future )) #Gap por los dias a prevision vista
        df_predi= pd.concat([df_gap,df_predi2], axis=0,ignore_index=True)
        df_predi = np.array(df_predi)
        
        ## Graficar ....................................        
        print (endDate_)
        print ('Prevision a  ', self.n_future)
        
        x = pd.DataFrame({'Números': range(1,1000)})

        plt.plot(x[:200],df_predi[-200:], color='red',label='Prediccion')
        #plt.plot(self.trainX_test[-200:,-1,self.Y_supervised], color='blue',label='Origen Prediccion')
        
        df_aux9 =df_for_training['hull'].copy()
        df_aux9=df_aux9.reset_index(drop=True)  #quito los index
        df_aux9= pd.concat([df_aux9,df_gap], axis=0,ignore_index=True)
        del df_aux9["X_dias"]
        
        plt.plot(x[:200],df_aux9[-200:], color='lightblue',label='Origen PPrediccion')
        plt.title(instrumento_ +" PREVISIONES a  "+ str(self.n_future) + ' dias. Con desplazam')
        plt.legend()
        plt.show()
        
        #Guardo los excel para cotejar los datos.
        ##quant_j.salvarExcel(df_predi, '../temp/'+instrumento_+'_predic_')
        ##quant_j.salvarExcel(df_for_training, '../temp/'+instrumento_+'_train_')
        
        
        ################################################
        #Estrategia simple: diferencia de pendientes identifican un minimo, prevision psotiva, anterior no da señal
        ##df_signal= pd.DataFrame({'signal':range(1,200)})
        df_signal= pd.DataFrame(columns=['signal'], index=range(len(self.trainX_test)))
        df_signal.fillna(0, inplace=True)
        for i in range( 10, len(self.trainX_test) ):
            #Prediccion subiendo tres dias
            if((df_predi2['X_dias'].iloc[i-4] < df_predi2['X_dias'].iloc[i-3]) and
               (df_predi2['X_dias'].iloc[i-3] < df_predi2['X_dias'].iloc[i-2]) and
               (df_predi2['X_dias'].iloc[i-2] < df_predi2['X_dias'].iloc[i-1]) and
               (df_predi2['X_dias'].iloc[i-1] < df_predi2['X_dias'].iloc[i-0])):
                df_signal['signal'].iloc[i]=1
            else:
                df_signal['signal'].iloc[i]=0
                        
   
        return df_signal, df_predi2, df_predi
    
    def predicionLSTM(self, instrumento, daysBack=200):
        """
        Descripcion: Metodo para calcular una prediccion con la red entrenada
        Voy a usar para probar a estrategia los datos del ultimo año. 
        Si la estrategia es buena, reentreno con todos los datos para predecir el futuro.
        Trabajo con los datos del array reservado para test... teoricamente para cada
        dia dado nos la la prediccion a los dias n_future definidos.

        Returns
        -------
        Devuelve un array con las ultimas previsiones, el ultimo elemento del array corresponde a la 
        prevision de hoy con el horizonte defindo para la red LSTM
        """
        #df_previsiones_xd
        
        ###################################################################
        #self.trainX_test   # Aquí guarde 250 ultimos datos +- un año
        iii=0
        #comienzo_=len(self.trainX_test) - daysBack
        comienzo_=0
        for i in range(comienzo_, len(self.trainX_test), 1):  # vamos avanzando a saltos de longuitud muestreo
            
            prediction = self.model.predict(self.trainX_test[i:i+1])
            #prediction = self.model.predict(self.trainX[0:1])
            
            #Perform inverse transformation to rescale back to original range
            #Since we used 5 variables for transform, the inverse expects same dimensions
            #Therefore, let us copy our values 5 times and discard them after inverse transform
            Y_supervised_num=self.dfx.columns.get_loc(self.Y_supervised)    #Selecciono la columna a supervisada
            prediction_copies = np.repeat(prediction, 8, axis=-1)   #df_for_training.shape[1]            
            prediction = scaler.inverse_transform(prediction_copies)[:,Y_supervised_num]  #self.Y_supervised]
            
                     
            self.df_previsiones_xd.loc[iii,'X_dias']= float(prediction)
            
            iii+=1
            
        iii
        return self.df_previsiones_xd
    
    
    def dataPreparation_1(self, instrumento='san', startD=5, endD=6):
        """
        Descripcion: Data preparation for LSTM training and prediction
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------
        """        
        
        #startD = dt.datetime(2018,1,10)
        #endD= dt.datetime.today()  - dt.timedelta(days=1)        #Quito hoy para no ver el valor al ejecutar antes del cierre
        #end = '2021-9-19'
        
        #instrumento = tickers_nasdaq[jjj]  ##'TEF.MC'
        self.instrumento = instrumento
        print(self.instrumento)
        
        mycsv = fileCSV.dataGenClass()    #Creamos la clase
        self.dfx=mycsv.creaCSV_01(instrumento, startD, endD)
        
        if self.dfx.empty:   #Error en la recogida de datos
            logging.info('No existe  {}'.format(instrumento))
            #continue
            return

        # #Read the csv file
        # #df = pd.read_csv('GE.csv')
        # df= pd.read_csv('telefonica_01.csv')    ### 'rovi_05.csv''zara_02.csv'telefonica_01.csv
        # print(df.tail()) #7 columns, including the Date. 
        # 
        
        # #Separate dates for future plotting
        # train_dates = pd.to_datetime(df['Date'])
        # print(train_dates.tail(5)) #Check last few dates. 
  
        print (self.dfx.shape)
        
        #Variables for training
        #self.cols = list(self.dfx)[0:8]  #df.columns.tolist()

        ## PARAMETRIZAR
        self.cols = list['Volume', 'EMA_100', 'EMA_30', 'Kalman', 'hull', 'dia', 'MA_Vol', 'Close']
        #Date and volume columns are not used in training. 
        print(self.cols) 

        #self.Y_supervised=self.dfx.columns.get_loc("hull")    #Selecciono la columna a supervisada
        #Y_supervised_num=self.dfx.columns.get_loc(self.Y_supervised)    #Selecciono la columna a supervisada
        
        #New dataframe with only training data - 5 columns
        global df_for_training
        df_for_training=self.dfx[self.cols].astype(float)
        #df_for_training_scaled_pre= self.dfx[self.cols].astype(float)
        
        Y_supervised_num=df_for_training.columns.get_loc(self.Y_supervised) 
    
        
        #df_for_plot=df_for_training.tail(500)
        ##j df_for_plot.plot.line()
        
        ### ESTUDIA como funciona este tema del scaler... deberia ser entre 0 y 1, no?
        
        #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
        # normalize the dataset
        #Estándariza los datos eliminando la media y escalando los datos de forma que su varianza sea igual a 1.
   
        global scaler 
        scaler = StandardScaler()
        scaler = scaler.fit(df_for_training)
        df_for_training_scaled_pre = scaler.transform(df_for_training)
        
         
        #Divido antes de prepararlos para no perder los  ultimos datos de test al hacer la preparacion supervidada
        #p_train = 0.80 # Porcentaje de train.
        #df_for_training_scaled = df_for_training_scaled_pre[:int((len(df_for_training_scaled_pre))*p_train)] 
        #df_for_training_scaled_test = df_for_training_scaled_pre[int((len(df_for_training_scaled_pre))*p_train):]
        df_for_training_scaled = df_for_training_scaled_pre[:int(-250)] 
        df_for_training_scaled_test = df_for_training_scaled_pre[int(-250):]  #dejo el ultimo año para test
   
        
        # df_data = pd.DataFrame(df_for_training_scaled ,columns = ['close','hull','50','100','30'])
        # df_data.to_excel("telefonica_scaled.xlsx", 
        #           index=True,
        #           sheet_name="data")

        #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
        #In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 
        
        #Empty lists to be populated using formatted training data
        #trainXX = []
        #trainYY = []
        #trainX_test=[]
        
        #self.n_future = previson_a_x_days   # origina=1,   Number of days we want to look into the future based on the past days.

        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        #In my example, my df_for_training_scaled has a shape (12823, 5)
        #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
        for i in range (LSTMClass.n_past, len(df_for_training_scaled) - self.n_future +1):
            ## en este caso Append añade un elemento que es un array de dos dimensiones.
            self.trainXX.append( df_for_training_scaled[ i - LSTMClass.n_past : i ,  0:df_for_training.shape[1]])  #n_past filas X 5 columnas (feautures)
            #slicing: fila desde (i-n_past) hasta i///// Columna desde 0: 5 =>(df_for_training.shape[1])
            
            self.trainYY.append(df_for_training_scaled[i + self.n_future - 1:i + self.n_future, Y_supervised_num])  ##[17:18,0] un posicoin de la fila para la columna 0
            ### el 4/Y_supervised es la caracteritica elegida Close//EMA//EMA100//
            
        for i in range (LSTMClass.n_past, len(df_for_training_scaled_test) +1):
            ## en este caso Append añade un elemento que es un array de dos dimensiones.
            self.trainX_test.append( df_for_training_scaled_test[ i - LSTMClass.n_past : i ,  0:df_for_training.shape[1]])  #n_past filas X 5 columnas (feautures)
            #slicing: fila desde (i-n_past) hasta i///// Columna desde 0: 5 =>(df_for_training.shape[1])
            

        # ## separar Training y TEST  Aqui no separo porque lo hize arriba
        self.trainX, trainX_test_kk, self.trainY, self.trainY_test  = train_test_split(self.trainXX, self.trainYY, test_size = 0.001,shuffle = False)
        
        
        # trainX_test[-1,-1, Y_supervised]
        # trainY_test[-1]
       
        self.trainX, self.trainY = np.array(self.trainX), np.array(self.trainY)
        self.trainX_test, self.trainY_test = np.array(self.trainX_test), np.array(self.trainY_test)
        
        print('trainX shape == {}.'.format(self.trainX.shape))
        print('trainY shape == {}.'.format(self.trainY.shape))
        print('trainX_test shape == {}.'.format(self.trainX_test.shape))
        
        """
        tenemos un array de 238 elementos en el que cada elemento es un array de 14x5
        """

        return


    def dataPreparation_PROD(self, instrumento='san', startD=5, endD=6):
        """
        Descripcion: Data preparation for LSTM training and prediction
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------
        """        
        
        #startD = dt.datetime(2018,1,10)
        #endD= dt.datetime.today()  - dt.timedelta(days=1)        #Quito hoy para no ver el valor al ejecutar antes del cierre
        #end = '2021-9-19'
        
        #instrumento = tickers_nasdaq[jjj]  ##'TEF.MC'
        self.instrumento = instrumento
        print(self.instrumento)
        
        mycsv = fileCSV.dataGenClass()    #Creamos la clase
        self.dfx=mycsv.creaCSV_01(instrumento, startD, endD)
        
        if self.dfx.empty:   #Error en la recogida de datos
            logging.info('No existe  {}'.format(instrumento))
            #continue
            return

        # #Read the csv file
        # #df = pd.read_csv('GE.csv')
        # df= pd.read_csv('telefonica_01.csv')    ### 'rovi_05.csv''zara_02.csv'telefonica_01.csv
        # print(df.tail()) #7 columns, including the Date. 
        # 
        
        # #Separate dates for future plotting
        # train_dates = pd.to_datetime(df['Date'])
        # print(train_dates.tail(5)) #Check last few dates. 
  
        print (self.dfx.shape)
        
        #Variables for training
        #self.cols = list(self.dfx)[0:8]  #df.columns.tolist()

        ## PARAMETRIZAR
        self.cols = list['Volume', 'EMA_100', 'EMA_30', 'Kalman', 'hull', 'dia', 'MA_Vol', 'Close']
        #Date and volume columns are not used in training. 
        print(self.cols) 

        
        #New dataframe with only training data - 5 columns
        global df_for_training
        df_for_training=self.dfx[self.cols].astype(float)
        #df_for_training_scaled_pre= self.dfx[self.cols].astype(float)
 
        
        #self.Y_supervised=self.dfx.columns.get_loc("hull")    #Selecciono la columna a supervisada
        #Y_supervised_num=self.dfx.columns.get_loc(self.Y_supervised)    #Selecciono la columna a supervisada
        Y_supervised_num=df_for_training.columns.get_loc(self.Y_supervised) 
    
        #df_for_plot=df_for_training.tail(500)
        ##j df_for_plot.plot.line()
        
        ### ESTUDIA como funciona este tema del scaler... deberia ser entre 0 y 1, no?
        
        #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
        # normalize the dataset
        #Estándariza los datos eliminando la media y escalando los datos de forma que su varianza sea igual a 1.
   
        global scaler 
        scaler = StandardScaler()
        scaler = scaler.fit(df_for_training)
        df_for_training_scaled_pre = scaler.transform(df_for_training)
        
         
        #Divido antes de prepararlos para no perder los  ultimos datos de test al hacer la preparacion supervidada
        #p_train = 0.80 # Porcentaje de train.
        #df_for_training_scaled = df_for_training_scaled_pre[:int((len(df_for_training_scaled_pre))*p_train)] 
        #df_for_training_scaled_test = df_for_training_scaled_pre[int((len(df_for_training_scaled_pre))*p_train):]
        df_for_training_scaled = df_for_training_scaled_pre[:] 
        df_for_training_scaled_test = df_for_training_scaled_pre[int(-50):]  # En produccion no dejo nada para test.
   
        
        # df_data = pd.DataFrame(df_for_training_scaled ,columns = ['close','hull','50','100','30'])
        # df_data.to_excel("telefonica_scaled.xlsx", 
        #           index=True,
        #           sheet_name="data")

        #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
        #In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 
        
        #Empty lists to be populated using formatted training data
        #trainXX = []
        #trainYY = []
        #trainX_test=[]
        
        #self.n_future = previson_a_x_days   # origina=1,   Number of days we want to look into the future based on the past days.

        #Reformat input data into a shape: (n_samples x timesteps x n_features)
        #In my example, my df_for_training_scaled has a shape (12823, 5)
        #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
        for i in range (LSTMClass.n_past, len(df_for_training_scaled) - self.n_future +1):
            ## en este caso Append añade un elemento que es un array de dos dimensiones.
            self.trainXX.append( df_for_training_scaled[ i - LSTMClass.n_past : i ,  0:df_for_training.shape[1]])  #n_past filas X 5 columnas (feautures)
            #slicing: fila desde (i-n_past) hasta i///// Columna desde 0: 5 =>(df_for_training.shape[1])
            
            self.trainYY.append(df_for_training_scaled[i + self.n_future - 1:i + self.n_future, Y_supervised_num])  ##[17:18,0] un posicoin de la fila para la columna 0
            ### el 4/Y_supervised es la caracteritica elegida Close//EMA//EMA100//
            
        for i in range (LSTMClass.n_past, len(df_for_training_scaled_test) +1):
            ## en este caso Append añade un elemento que es un array de dos dimensiones.
            self.trainX_test.append( df_for_training_scaled_test[ i - LSTMClass.n_past : i ,  0:df_for_training.shape[1]])  #n_past filas X 5 columnas (feautures)
            #slicing: fila desde (i-n_past) hasta i///// Columna desde 0: 5 =>(df_for_training.shape[1])
            

        # ## separar Training y TEST  Aqui no separo porque lo hize arriba
        self.trainX, trainX_test_kk, self.trainY, self.trainY_test  = train_test_split(self.trainXX, self.trainYY, test_size = 0.001,shuffle = False)
        
        
        # trainX_test[-1,-1, Y_supervised]
        # trainY_test[-1]
       
        self.trainX, self.trainY = np.array(self.trainX), np.array(self.trainY)
        self.trainX_test, self.trainY_test = np.array(self.trainX_test), np.array(self.trainY_test)
        
        print('trainX shape == {}.'.format(self.trainX.shape))
        print('trainY shape == {}.'.format(self.trainY.shape))
        print('trainX_test shape == {}.'.format(self.trainX_test.shape))
        
        """
        tenemos un array de 238 elementos en el que cada elemento es un array de 14x5
        """
        return

    
    def LSTM_net_2 (self, parametro1=1, parametro2=2, parametro3=3):
        """
        Descripcion: NET LSTM definition and fitting.
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------
        """
        # define the Autoencoder model
        
        self.model = Sequential()
        self.model.add(LSTM(256, activation='relu', input_shape=(self.trainX.shape[1], self.trainX.shape[2]), return_sequences=True)) #64
        self.model.add(LSTM(128, activation='relu', return_sequences=False))   #32
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.trainY.shape[1]))
        
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

        # fit the model
        #global epochs_
        history = self.model.fit(self.trainX, self.trainY, epochs=epochs_, batch_size=16, validation_split=0.15, verbose=1) #batch=16
        
        """
        plt.title("LSTM training")
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()
        """
        
        #logging.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))
        logging.info('Loss: {}'.format(self.n_future))      
        
        #Predicting...
        prediction = self.model.predict(self.trainX[0:1])
        #prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction
                                                                    # desde -n hasta el final. Cada elemento es un array bidimensional
        #Pido una predcicion para un array de fechas, me devuelve la predicion para cada una
        
        #Nos vamos n_daysforPredcition atras y calculamos la precidion a n_future (6) days despues.
        
        return

    def LSTM_net_2_PROD (self, instrumento_, parametro1=1, parametro2=2, parametro3=3):
        """
        Descripcion: esta funcion toma los datos entrena la red y la guarda en disco
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------
        """
        # define the Autoencoder model
        
        self.model = Sequential()
        self.model.add(LSTM(256, activation='relu', input_shape=(self.trainX.shape[1], self.trainX.shape[2]), return_sequences=True)) #64
        self.model.add(LSTM(128, activation='relu', return_sequences=False))   #32
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.trainY.shape[1]))
        
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

        # fit the model
        #global epochs_
        history = self.model.fit(self.trainX, self.trainY, epochs=epochs_, batch_size=16, validation_split=0.15, verbose=1) #batch=16
        
        
        plt.title("LSTM training")
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()        
        
        #logging.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))
        logging.info('Loss: {}'.format(self.n_future))      
        
        #Predicting...
        prediction = self.model.predict(self.trainX[0:1])
        #prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction
                                                                    # desde -n hasta el final. Cada elemento es un array bidimensional
        #Pido una predcicion para un array de fechas, me devuelve la predicion para cada una   
        #Nos vamos n_daysforPredcition atras y calculamos la precidion a n_future (6) days despues.
        
        #GUARDO EL MODELO PARA ESTE INSTRUMENTO
        self.model.save("../models/mod_"+str(self.n_future)+"d_"+instrumento_)
                
        return
 
    def train_and_save(self, instrumento_, startDate_, endDate_):
       """
       Descripcion: parto de los datos de reales y las predicciones de la LSTM, y defino una estrategia.
       Atencion que desplazo en array de previsiones para que coincida el una misma vertical el valor real con la prevision
       
       
       Parameters
       ----------
       beneficio : TYPE
           DESCRIPTION.

       Returns
       -------

       """
       ##  RED
       #Preparo los datos
       self.dataPreparation_PROD(instrumento_,startDate_, endDate_)
       #creo y entreno la NET
       self.LSTM_net_2_PROD(instrumento_)
                             
  
       return 
    


    def plottingSecuence_prevision(myLSTMnet):
        """
        Descripcion: sample method
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------
        
        

        """

            ## Pinto en un grafico los parametros del ejercico.
        if True:
            plt.title("Datos del ejercicio")
            #plt.text(0.1,0.6, 'Predicción a '+str(myLSTMnet.n_future)+' dias' ) 
            plt.text(0.1, 0.2, 'Ultimo dia ' + str( myLSTMnet.dfx.index[-1]))
            plt.text(0.1, 0.4, 'datos'  + str(myLSTMnet.cols))
            plt.legend()
            
            if ((pdf_flag == True) and (LSTMClass.flag01)):
                LSTMClass.flag01=False
                plt.savefig("../docs/temp/0_descricion.pdf")
            plt.show()
            
            
            ### VISUALIZAION DATOS TEST  (esto no son lo datos de test, TrainX_test deberia ser)
            
            pred_gap = np.zeros(myLSTMnet.n_future)
            muestra_gap =[]
            xx=[]
            
            muestreo=35  # minimo 20
            gapmuestras=np.zeros(muestreo-LSTMClass.n_past)
            gapprevion=np.zeros(myLSTMnet.n_future)
            gappostprevison= np.zeros(muestreo-myLSTMnet.n_future-LSTMClass.n_past)
            
            for ii in range (len(gapmuestras)):
                gapmuestras[ii] = np.nan
            for ii in range (len(gapprevion)):
                gapprevion[ii] = np.nan
            for ii in range (len(gappostprevison)):
                gappostprevison[ii] = np.nan 
            for ii in range (len(pred_gap)):
                pred_gap[ii] = np.nan      
            
            
            inicio= np.zeros(myLSTMnet.n_future+LSTMClass.n_past)    
            
            origen=LSTMClass.n_past+myLSTMnet.n_future
            
            
            for i in range(LSTMClass.n_past, len(myLSTMnet.trainX_test)-20, muestreo):  # vamos avanzando a saltos de longuitud muestreo
                
                prediction = myLSTMnet.model.predict(myLSTMnet.trainX_test[i-LSTMClass.n_past:i])
                
                #Perform inverse transformation to rescale back to original range
                #Since we used 5 variables for transform, the inverse expects same dimensions
                #Therefore, let us copy our values 5 times and discard them after inverse transform
                prediction_copies = np.repeat(prediction, 8, axis=-1)   #df_for_training.shape[1]
                prediction = scaler.inverse_transform(prediction_copies)[:,myLSTMnet.Y_supervised]
                                
                
                prediction.shape = (LSTMClass.n_past)   #me devuleve 15//n_past previsiones a 6//n_future dias vista
                pred_gap=np.concatenate((pred_gap, prediction), axis=0)  #predcicion son n_past... en un futuro de n_futre muestras
                pred_gap=np.concatenate((pred_gap, gapmuestras), axis=0)
            
                ##xx=(trainX[i-n_past:i,0,Y_supervised])
                xx=(myLSTMnet.trainX_test[i-LSTMClass.n_past:i,-1,myLSTMnet.Y_supervised])    ##yyy=trainX_test[-(n_past+n_future):,-1,Y_supervised]   
                ##xx= trainXX[-1:, -n_past:, Y_supervised]  # coje el ultimo tramo de datos 
                xx.shape = ( LSTMClass.n_past)
                muestra_gap=np.concatenate((muestra_gap, xx  ), axis=0)
                muestra_gap=np.concatenate((muestra_gap, gapmuestras), axis=0)
            
                #print(trainX[i:i+n_days_for_prediction])
                ##print(xx)
                #fake =input()
        
            ###################################################################
            # Ahora ultimos datos, me reserve 20 en el for de arriba
            try: 
                #falta empujar la curva a su sitio en la parte final de la curva.
                prediction = myLSTMnet.model.predict(myLSTMnet.trainX_test[-LSTMClass.n_past:])
                
                prediction_copies = np.repeat(prediction, 8, axis=-1)   #df_for_training.shape[1]
                prediction = scaler.inverse_transform(prediction_copies)[:,myLSTMnet.Y_supervised]
                
                prediction.shape = (LSTMClass.n_past)   #me devuleve 15//n_past previsiones a 6//n_future dias vista
            except:
                logging.info('Error en la cantidad de los datos de test')
                
                #continue  #pondría un break para salir del todo del bucle for, continue ten lleva la sigueitne iteracion
            finally:
                logging.info('se ejecuta siempre... try')
                
            #relleno hasta el final
            arr_x =np.zeros(len(myLSTMnet.trainX_test)-len(pred_gap)-LSTMClass.n_past+myLSTMnet.n_future )
            for ii in range (len(arr_x)):
                arr_x[ii] = np.nan    
            pred_gap=np.concatenate((pred_gap,arr_x),axis=0)
            pred_gap=np.concatenate((pred_gap, prediction), axis=0) 
            
            ##des Scaler
            """
            prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=0)
            predd= prediction_copies.reshape(14,8)
            y_pred_future = scaler.inverse_transform(predd)[:,0]
            del prediction
            prediction=y_pred_future
            """
            
            ## Preparo las graficas 
            #pred_gap=np.concatenate((pred_gap, gapmuestras[0:5]), axis=0)
            #pred_gap=np.concatenate((pred_gap, prediction), axis=0)  #predcicion son n_past... en un futuro de n_futre muestras
            #pred_gap=np.concatenate((pred_gap, gapmuestras), axis=0)
           
            #yyy=myLSTMnet.trainX_test[0:(len(myLSTMnet.trainX_test)),-1,myLSTMnet.Y_supervised]        ## yyy=trainXX[-1:,-n_past:,Y_supervised]   de todos los grupos de 14 observaciones, cojo la primero y la column 'hull' 
            yyy=myLSTMnet.trainX_test[0:,-1,myLSTMnet.Y_supervised]   
            yyy_copies = np.repeat(yyy, 8, axis=-1)   #df_for_training.shape[1]
            yyy=np.reshape(yyy_copies, (len(yyy), -1))            
            yyy = scaler.inverse_transform(yyy)[:,myLSTMnet.Y_supervised]   ### DESESCALAR
            yyy.shape = (len(yyy)) 
            plt.plot(yyy, label='datos REALES',color='pink')
            
            
            yyy_copies = np.repeat(muestra_gap, 8, axis=-1)   #df_for_training.shape[1]
            yyy=np.reshape(yyy_copies, (len(muestra_gap), -1))            
            muestra_gap = scaler.inverse_transform(yyy)[:,myLSTMnet.Y_supervised]   ### DESESCALAR
            #yyy.shape = (len(muestra_gap))             
            plt.plot(muestra_gap, color='red',label='Origen Prediccion')
            
            
            plt.plot(pred_gap, color='lightgreen', label='Predicción')
            plt.title(myLSTMnet.instrumento+' predicción a '+str(myLSTMnet.n_future)+' dias' ) 
            #plt.text(0, np.amax(myLSTMnet.trainY_test) , 'Ultimo dia ' + str( myLSTMnet.dfx.index[-1]))
            #plt.text(0, (np.amin(trainY_test) - ((np.amax(trainY_test)-np.amin(trainY_test))/3.64))-0.1, 'datos'  + str(cols))
            plt.legend()
            
            if (pdf_flag == True):
                plt.savefig("../docs/temp/"+myLSTMnet.instrumento+str(myLSTMnet.n_future)+".pdf")
            plt.show()

   
        return

    
#################################################### Clase FIN






#/******************************** FUNCION PRINCIPAL main() *********/
#     def main():   
if __name__ == '__main__':    
        
    """Esta parte del codigo se ejecuta cuando llamo tipo EXE
    Abajo tenemos el else: librería que es cuando se ejecuta como libreria.
        
    Parámetros:
    a -- 
    b -- 
    c -- 
    
    Devuelve:
    Valores 

    Excepciones:
    ValueError -- Si (a == 0)
    
    """   

    print(sys.argv[1])   #se configura en 'run' 'configuration per file'

    print ('version(J): ',versionVersion) 

    if (False and sys.argv[1]== 'train'):
        print('Train & Save')
        # Determino las fechas
        fechaInicio_ = dt.datetime(2008,1,10)
        fechaFin_ = dt.datetime.today()  - dt.timedelta(days=1)    
        
        
        ####################################### Entreno la RED y la guardo
        myLSTMnet_ = LSTMClass(previson_a_x_days=4, Y_supervised_ = 'hull')          #Creamos la clase
        myLSTMnet_.train_and_save(tickers_ibex[6], fechaInicio_, fechaFin_)
        print (fechaFin_)
        print ('Dias de prevision  ', myLSTMnet_.n_future)
        print ('Variable objetivo  ', myLSTMnet_.Y_supervised )
        print ('Instrumento        ', myLSTMnet_.instrumento )
        
        sys.exit()
    
    if (True or sys.argv[1]== 'prod' ):
        print('Produccion')

        instrumento_ = sys.argv[2]        
        #Recuepro el modelos entrenado       
        #instrumento_ = tickers_ibex[6]
        n_future = 4
        new_model = tf.keras.models.load_model("../models/mod_"+str(n_future)+"d_"+instrumento_)

                
        
        # Determino las fechas
        fechaInicio_ = dt.datetime(2008,1,10)
        fechaFin_ = dt.datetime.today()  - dt.timedelta(days=1)    
        
        
        ####################################### Entreno la RED y la guardo
        myLSTMnet_ = LSTMClass(previson_a_x_days=4, Y_supervised_ = 'hull')          #Creamos la clase
        myLSTMnet_.train_and_save(tickers_ibex[6], fechaInicio_, fechaFin_)
        print (fechaFin_)
        print ('Dias de prevision  ', myLSTMnet_.n_future)
        print ('Variable objetivo  ', myLSTMnet_.Y_supervised )
        print ('Instrumento        ', myLSTMnet_.instrumento )
        
        sys.exit()

    
    #################### PROBAMOS LA ESTRATEGIA
    for jjj in range(0,len(tickers_ibex)):    ##tickers_sp500
        myLSTMnet_6D =LSTMClass(6, Y_supervised_ = 'hull')          #Creamos la clase
        df_signal= myLSTMnet_6D.estrategia_LSTM_01( tickers_ibex[jjj], fechaInicio_, fechaFin_)
    
    print('This is it................ ')
    
    
    sys.exit()
    
    

    for jjj in range(0,len(tickers_eurostoxx)):    ##tickers_sp500

         
        ## Primera RED
        myLSTMnet_2 =LSTMClass(previson_a_x_days=2, Y_supervised_ = 'hull')          #Creamos la clase
        #Preparo los datos
        myLSTMnet_2.dataPreparation_1(tickers_eurostoxx[jjj],fechaInicio_, fechaFin_)
        #creo y entreno la NET
        myLSTMnet_2.LSTM_net_2()
        
        
        ## Segunda RED
        myLSTMnet_5 =LSTMClass(previson_a_x_days=5,Y_supervised_ = 'hull')          #Creamos la clase
        #Preparo los datos
        myLSTMnet_5.dataPreparation_1(tickers_eurostoxx[jjj],fechaInicio_, fechaFin_)
        #creo y entreno la NET
        myLSTMnet_5.LSTM_net_2()
     
        
        ## Tercera RED
        myLSTMnet_12 =LSTMClass(previson_a_x_days=12, Y_supervised_ = 'hull')          #Creamos la clase
        #Preparo los datos
        myLSTMnet_12.dataPreparation_1(tickers_eurostoxx[jjj],fechaInicio_, fechaFin_)
        #creo y entreno la NET
        myLSTMnet_12.LSTM_net_2()
        
        
        # Pinto la grafica
        LSTMClass.plottingSecuence_prevision(myLSTMnet_2)
        LSTMClass.plottingSecuence_prevision(myLSTMnet_5)
        LSTMClass.plottingSecuence_prevision(myLSTMnet_12)
        
        #df_predi= myLSTMnet_5.predicionLSTM(tickers_ibex[jjj], myLSTMnet_5)
        print (fechaFin_)
        print ('Prevision a  ', myLSTMnet_5.n_future)
        print ('Prevision a  ', myLSTMnet_5.Y_supervised )
        print ('Instrumento  ', myLSTMnet_5.instrumento )

        # nte break  #solo hago una iteracion :-)
    
    print('This is it................ 6')
    
    
    """
    Entrada por la librería.
    """
else:
    """
    Esta parte del codigo se ejecuta si uso como libreria/paquete""    
    """    
    print (' libreria')
    print ('version(l): ',versionVersion)    
    

    
    
    # Entreno la red
    
    # Lanzo la prediccion
    
    