#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://youtu.be/tepxdcepTbY
"""
@author: J3viton
Code tested on Tensorflow: 2.2.0
    Keras: 2.4.3
dataset: https://finance.yahoo.com/quote/GE/history/
Also try S&P: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
"""


# In[2]:

import logging    #https://docs.python.org/3/library/logging.html
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='registro.log', level=logging.INFO ,force=True,
                    format='%(asctime)s:%(levelname)s:%(message)s')

logging.warning('esto es una kkk')


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import seaborn as sns



# In[3]:


import generarCSV as fileCSV
import datetime as dt

mycsv = fileCSV.dataGenClass()    #Creamos la clase


df32 = pd.DataFrame(columns=['instrumento', 'divergencia', 'fecha'])

# In[4]:


import sys
import os
sys.path.insert(0,"C:\\Users\\INNOVACION\\Documents\\J3\\100.- cursos\\Quant_udemy\\programas\\Projects\\libreria")
import quant_j3_lib as quant_j

from sp500 import tickers_sp500
from nasdaq import tickers_nasdaq
from ibex import tickers_ibex
from comodity import tickers_comodities


pdf_flag =True

# ### Instrumentos

# In[5]:


tickers5 = ['mrl.mc']

#VALORES DEL SP500
###(tickers_sp500)): 

# In[6]:


#VALORES DEL IBEX 
for jjj in range(0,len(tickers_ibex)):    ##tickers_sp500
    #### FECHAS
    #start =dt.datetime(2000,1,1)
    ##startD =dt.datetime.today() - dt.timedelta(days=5*250)    #un año tiene 250 sesiones.
            
    startD = dt.datetime(2002,1,10)
    endD= dt.datetime.today()  - dt.timedelta(days=1)        #Quito hoy para no ver el valor al ejecutar antes del cierre
    #end = '2021-9-19'
    
    instrumento = tickers_ibex[jjj]  ##'TEF.MC'
    print(instrumento)
    df=mycsv.creaCSV_01(instrumento, startD, endD)
    
    if df.empty:   #Error en la recogida de datos
        logging.info('No existe  {}'.format(instrumento))
        continue

    # #Read the csv file
    # #df = pd.read_csv('GE.csv')
    # df= pd.read_csv('telefonica_01.csv')    ### 'rovi_05.csv''zara_02.csv'telefonica_01.csv
    # print(df.tail()) #7 columns, including the Date. 
    # 
    
    # #Separate dates for future plotting
    # train_dates = pd.to_datetime(df['Date'])
    # print(train_dates.tail(5)) #Check last few dates. 
    
    # In[7]:
    
    
    print (df.shape)
    
    
    # In[8]:
    
    
    #Variables for training
    cols = list(df)[0:8]  #df.columns.tolist()
    #Date and volume columns are not used in training. 
    print(cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']
    
    
    # In[9]:
    
    
    cols[0]
    
    
    # In[10]:
    
    
    hull_col=df.columns.get_loc("hull")
    
    
    
    
    
    # In[11]:
    
    
    #New dataframe with only training data - 5 columns
    df_for_training = df[cols].astype(float)
    
    df_for_plot=df_for_training.tail(500)
    ##j df_for_plot.plot.line()
    
    
    # In[12]:
    
    
    ### ESTUDIA como funciona este tema del scaler... deberia ser entre 0 y 1, no?
    
    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    #Estándariza los datos eliminando la media y escalando los datos de forma que su varianza sea igual a 1.
    
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)
    df_for_training_scaled
    
    
    # In[13]:
    
    
    df_for_training_scaled.shape
    
    
    # df_data = pd.DataFrame(df_for_training_scaled ,columns = ['close','hull','50','100','30'])
    # df_data.to_excel("telefonica_scaled.xlsx", 
    #           index=True,
    #           sheet_name="data")
    
    # In[ ]:
    
    
    
    
    
    # In[14]:
    
    
    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
    #In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training). 
    
    #Empty lists to be populated using formatted training data
    trainXX = []
    trainYY = []
    trainX_test=[]
    
    n_future =1   # origina=1,   Number of days we want to look into the future based on the past days.
    n_past = 14  # Number of past days we want to use to predict the future.  FILAS
    
    
    # In[15]:
    
    
    #Reformat input data into a shape: (n_samples x timesteps x n_features)
    #In my example, my df_for_training_scaled has a shape (12823, 5)
    #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
    for i in range(n_past, len(df_for_training_scaled) - n_future +1):
        ## en este caso Append añade un elemento que es un array de dos dimensiones.
        trainXX.append( df_for_training_scaled[ i - n_past : i ,  0:df_for_training.shape[1]])  #n_past filas X 5 columnas (feautures)
        #slicing: fila desde (i-n_past) hasta i///// Columna desde 0: 5 =>(df_for_training.shape[1])
        
        trainYY.append(df_for_training_scaled[i + n_future - 1:i + n_future, hull_col])  ##[17:18,0] un posicoin de la fila para la columna 0
        ### el 4/hull_col es la caracteritica elegida Close//EMA//EMA100//
    
    
    # ## separar Training y TEST
    
    # In[16]:
    
    
    trainX, trainX_test, trainY, trainY_test  = train_test_split(trainXX, trainYY, test_size = 0.05,shuffle = False)
    
    
    # trainX_test[-1,-1, hull_col]
    
    # trainY_test[-1]
    
    # In[17]:
    
    
    trainX, trainY = np.array(trainX), np.array(trainY)
    trainX_test, trainY_test = np.array(trainX_test), np.array(trainY_test)
    
    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))
    print('trainX_test shape == {}.'.format(trainX_test.shape))
    
    """
    tenemos un array de 238 elementos en el que cada elemento es un array de 14x5
    """
    
    

    
    
    
    # In[21]:
    
    
    # define the Autoencoder model
    
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)) #64
    model.add(LSTM(128, activation='relu', return_sequences=False))   #32
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    
    # fit the model
    history = model.fit(trainX, trainY, epochs=18, batch_size=16, validation_split=0.15, verbose=1) #batch=16
    
    ##plt.plot(history.history['loss'], label='Training loss')
    ##plt.plot(history.history['val_loss'], label='Validation loss')
    ##plt.legend()
    
    #logging.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))
    logging.debug('Loss: {}'.format(n_future))
    logging.info('mensaje 32')
    
    # In[22]:
    
    
    #Predicting...
    #Libraries that will help us extract only business days in the US.
    #Otherwise our dates would be wrong when we look back (or forward).  
    #from pandas.tseries.holiday import USFederalHolidayCalendar
    #from pandas.tseries.offsets import CustomBusinessDay
    
    #us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    #Remember that we can only predict one day in future as our model needs 5 variables
    #as inputs for prediction. We only have all 5 variables until the last day in our dataset.
    
    
    ##n_days_for_prediction=10  #let us predict past 15 days
    
    ##predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_past, freq='D').tolist()
    ##print(predict_period_dates)
    
    #Make prediction
    prediction = model.predict(trainX[0:1])
    #prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction
                                                                # desde -n hasta el final. Cada elemento es un array bidimensional
    #Pido una predcicion para un array de fechas, me devuelve la predicion para cada una
    
    #Nos vamos n_daysforPredcition atras y calculamos la precidion a n_future (6) days despues.
    
    
    
    ## Pinto en un grafico los parametros del ejercico.
    if (jjj ==0):
        plt.title("Datos del ejercicio")
        plt.text(0.1,0.6, 'Predicción a '+str(n_future)+' dias' ) 
        plt.text(0.1, 0.2, 'Ultimo dia ' + str( df.index[-1]))
        plt.text(0.1, 0.4, 'datos'  + str(cols))
        plt.legend()
        
        if (pdf_flag == True):
            plt.savefig("0_descricion.pdf")
        plt.show()
    
    # In[23]:
    
    
    #print (trainX[-2:])  # dos ultimos elementos. Cada elemento es un array de 15x5
    #print (prediction)
    
    
    # In[24]:
    
    
    #trainY[-n_past+n_future:]
    
    
    # In[25]:
    
    
    #xx=(range(n_past))
    
    #pred= prediction
    #pred.shape = (n_past)
    #real=trainY[-n_past:]
    #real.shape = (n_past)
    
    #sns.lineplot(x=xx, y=pred, color='lightgreen')
    #sns.lineplot(x=xx, y=real)
    
    
    # real
    
    # In[26]:
    
    
    #pred
    
    
    # trainX.shape
    
    ### VISUALIZAION DATOS TEST  (esto no son lo datos de test, TrainX_test deberia ser)
    
    pred_gap = np.zeros(n_future)
    muestra_gap =[]
    xx=[]
    
    muestreo=35  # minimo 20
    gapmuestras=np.zeros(muestreo-n_past)
    gapprevion=np.zeros(n_future)
    gappostprevison= np.zeros(muestreo-n_future-n_past)
    
    for ii in range (len(gapmuestras)):
        gapmuestras[ii] = np.nan
    for ii in range (len(gapprevion)):
        gapprevion[ii] = np.nan
    for ii in range (len(gappostprevison)):
        gappostprevison[ii] = np.nan 
    for ii in range (len(pred_gap)):
        pred_gap[ii] = np.nan      
    
    
    inicio= np.zeros(n_future+n_past)    
    
    origen=n_past+n_future
    
    
    for i in range(n_past, len(trainX_test)-20, muestreo):  # vamos avanzando a saltos de longuitud muestreo
        
        prediction = model.predict(trainX_test[i-n_past:i])
        prediction.shape = (n_past)   #me devuleve 15//n_past previsiones a 6//n_future dias vista
        pred_gap=np.concatenate((pred_gap, prediction), axis=0)  #predcicion son n_past... en un futuro de n_futre muestras
        pred_gap=np.concatenate((pred_gap, gapmuestras), axis=0)
    
        ##xx=(trainX[i-n_past:i,0,hull_col])
        xx=(trainX_test[i-n_past:i,-1,hull_col])    ##yyy=trainX_test[-(n_past+n_future):,-1,hull_col]   
        
        ##xx= trainXX[-1:, -n_past:, hull_col]  # coje el ultimo tramo de datos 
        xx.shape = ( n_past)
        
        muestra_gap=np.concatenate((muestra_gap, xx  ), axis=0)
        muestra_gap=np.concatenate((muestra_gap, gapmuestras), axis=0)
    
        #print(trainX[i:i+n_days_for_prediction])
        ##print(xx)
        #fake =input()


    # Ahora ultimos datos, me reserve 20 en el for de arriba
    try: 
        prediction = model.predict(trainX_test[-n_past:])
        prediction.shape = (n_past)   #me devuleve 15//n_past previsiones a 6//n_future dias vista
    except:
        logging.debug('Error en la cantidad de los datos de test')
        
        continue  #pondría un break para salir del todo del bucle for, continue ten lleva la sigueitne iteracion
    finally:
        logging.info('se ejecuta siempre... try')
    
 
    ##des Scaler
    """
    prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=0)
    predd= prediction_copies.reshape(14,8)
    y_pred_future = scaler.inverse_transform(predd)[:,0]
    del prediction
    prediction=y_pred_future
    """
    
    ## Preparo las graficas 
    pred_gap=np.concatenate((pred_gap, gapmuestras[0:5]), axis=0)
    pred_gap=np.concatenate((pred_gap, prediction), axis=0)  #predcicion son n_past... en un futuro de n_futre muestras
    pred_gap=np.concatenate((pred_gap, gapmuestras), axis=0)
   
    yyy=trainX_test[0:(len(trainX_test)-6),-1,hull_col]        ## yyy=trainXX[-1:,-n_past:,hull_col]   de todos los grupos de 14 observaciones, cojo la primero y la column 'hull' 
    plt.plot(yyy, label='datos REALES',color='pink')
    
    plt.plot(muestra_gap, color='red',label='Origen Prediccion')
    plt.plot(pred_gap, color='lightgreen', label='Predicción')
    plt.title(instrumento+' predicción a '+str(n_future)+' dias' ) 
    plt.text(0, np.amin(trainY_test) - ((np.amax(trainY_test)-np.amin(trainY_test))/3.64), 'Ultimo dia ' + str( df.index[-1]))
    #plt.text(0, (np.amin(trainY_test) - ((np.amax(trainY_test)-np.amin(trainY_test))/3.64))-0.1, 'datos'  + str(cols))
    plt.legend()
    
    if (pdf_flag == True):
        plt.savefig(instrumento+"3.pdf")
    plt.show()
    
    
    #break

    
    # In[30]:
    
  
    ##########################################################################################
    # ## Preparo los datos para la predicción final
    """ JJ
    
    # In[33]:
    trainXFinal = []
    trainYFinal = []
        
    
    #for i in range(-n_past, len(df_for_training_scaled) ):   #cambio n_past por -n_past
    ""
    for i in range (n_past):
        ## en este caso Append añade un elemento que es un array de dos dimensiones.
        trainXFinal.append( df_for_training_scaled[ i - n_past : i ,  0:df_for_training.shape[1]]) 
    trainXFinal = np.array(trainXFinal)
    ""    
    
    
    pred_gappo = np.zeros(n_future)
    muestra_gappo =[]
    gappreviono=[]
    xxx=[]
    
    muestreo=30  # minimo 20

    gappreviono=np.zeros(0)
    

    for ii in range (len(gappreviono)):
        gappreviono[ii] = np.nan

    for ii in range (len(pred_gappo)):
        pred_gappo[ii] = np.nan  
    
            ##j#prediction = model.predict(trainXFinal[-n_past:])
    
    predictiono = model.predict(trainX_test[-n_past:])
    predictiono.shape = (n_past)   #me devuleve 14//n_past previsiones a 6//n_future dias vista desde la ultima referencia
    pred_gappo=np.concatenate((pred_gappo, predictiono), axis=0)  #predcicion son n_past... en un futuro de n_futre muestras
    ##pred_gapp=np.concatenate((pred_gapp, gapmuestras), axis=0)  #sobra?
    
    
    ## MAL xx=(trainXX[-n_past:,0,hull_col]) ## Mal porque coje el primer elemento de cada uno de los tramos... parecido pero mal
    xxx= trainX_test[-1:, -n_past:, hull_col]  # coje el ultimo tramo de datos   ##FALTA EL ULTIMO DATO!!!!!!!!!!!!!!!!!
    xxx.shape = ( n_past)
    muestra_gappo=np.concatenate((muestra_gappo, xxx  ), axis=0)
    #muestra_gappo=np.concatenate((muestra_gappo, gapmuestras), axis=0)
    
    
    
    # %matplotlib widget
    
    # In[34]:
    
    
    ##Plot
    yy2y=trainX_test[-n_past:,-1,0]
                #yy2y.shape = (len(df_for_training_scaled[0]) )
    ##plt.plot(yy2y, label='curva Close',color='lightblue')
    
    
    yyy=trainX_test[-(n_past+n_future):,-1,hull_col]                    ###
    yyy.shape = ( n_past+n_future)
    plt.plot(yyy, label='curva MEDIA (referencia)',color='pink')
    
    #plt.plot(muestra_gappo, color='red',label='datos origen Prediccion .. Hulk')
    
    plt.plot(pred_gappo, color='lightgreen', label='Predicción')
    plt.title(instrumento) 
    plt.legend()
    
    if (pdf_flag == True):
        plt.savefig(instrumento+"2.pdf")

    plt.show()

    plt.title(instrumento)    
    

    
    ##############################################
    # ## Busco divergencia Predicción versus Actual
    
    # In[42]:
    
    
    fd= pd.DataFrame(pred_gappo, columns =['1'])
    
    
    # In[43]:
    
    
    #fd.tail()
    
    
    # In[44]:
    
    
    #################################################### RegresionLineal()
    fd_pred= pd.DataFrame(pred_gappo[15:(15+n_past)], columns =['1'])   # Que es el 16???
    
    # 1.- Calculamos media de las ultimas sesiones y la regresion lineal
    coef_p, intercept_ =quant_j.linearRegresion_J3(fd_pred['1'],instrumento='_')
    
    
    # pred_gapp[16:(16+n_past)]
    
    # In[45]:
    
    
    #################################################### RegresionLineal()
    fd2_sample= pd.DataFrame(muestra_gappo[0:n_past], columns =['1'])   # Que es el 16???
    
    # 1.- Calculamos media de las ultimas sesiones y la regresion lineal
    coef2_s, intercept2_ =quant_j.linearRegresion_J3(fd2_sample['1'],instrumento='_')
    
    

    
    #### Información final
    
    if((coef2_s * coef_p)<0):
        if(coef_p >0):
            print ('==================================================//////////////////////  Invierte en ', instrumento)
            df32 = df32.append({'instrumento': instrumento, 'divergencia':'yes', 'fecha':endD}, ignore_index=True)
    else:
        print(' Ese intrumento NO presenta divergencia', instrumento)
        df32 = df32.append({'instrumento': instrumento, 'divergencia':'no', 'fecha': endD}, ignore_index=True)
    print(df32)
    
    JJ"""


print('This is it................ ')
