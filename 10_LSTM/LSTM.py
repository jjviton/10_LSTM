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
versionVersion = 1
globalVar  = True

###################### DATOS
from sp500 import tickers_sp500
from nasdaq import tickers_nasdaq
from ibex import tickers_ibex
from comodity import tickers_comodities


pdf_flag =True

#################################################### Clase Estrategia 

class LSTMClass:

    """CLASE ESTRATEGIA

       
    """  
    
    #Variable de CLASE
    backtesting = False  #variable de la clase, se accede con el nombre
    n_past = 14  # Number of past days we want to use to predict the future.  FILAS
   
    def __init__(self, previson_a_x_days=1, para1=False, para2=1):
        
        #Variable de INSTANCIA
        self.para_02 = para2   #variable de la isntancia
        
        self.trainXX = []
        self.trainYY = []
        self.trainX_test=[]
        self.trainX = []
        self.trainY = []
        
        self.dfx = pd.DataFrame()
        self.cols =0
        self.hull_col =0

        self.n_future=previson_a_x_days
        
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
    
    def predicionLSTM(self, instrumento, startDate, endDate, DF):
        """
        Descripcion: Metodo para calcular una prediccion con la red entrenada
        
        Parameters
        ----------
        beneficio : TYPE

        Returns
        -------


        """
        pass
   
        return
    
    
    def dataPreparation_1(self, instrumento='san', fechaInicio=5, fechaFin=6):
        
        startD = dt.datetime(2018,1,10)
        endD= dt.datetime.today()  - dt.timedelta(days=1)        #Quito hoy para no ver el valor al ejecutar antes del cierre
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
        self.cols = list(self.dfx)[0:8]  #df.columns.tolist()
        #Date and volume columns are not used in training. 
        print(self.cols) #['Open', 'High', 'Low', 'Close', 'Adj Close']


        self.hull_col=self.dfx.columns.get_loc("hull")

        
        #New dataframe with only training data - 5 columns
        df_for_training = self.dfx[self.cols].astype(float)
        
        df_for_plot=df_for_training.tail(500)
        ##j df_for_plot.plot.line()
        
        ### ESTUDIA como funciona este tema del scaler... deberia ser entre 0 y 1, no?
        
        #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
        # normalize the dataset
        #Estándariza los datos eliminando la media y escalando los datos de forma que su varianza sea igual a 1.
        
        scaler = StandardScaler()
        scaler = scaler.fit(df_for_training)
        df_for_training_scaled = scaler.transform(df_for_training)
        df_for_training_scaled
        
       
        df_for_training_scaled.shape
        
        
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
            
            self.trainYY.append(df_for_training_scaled[i + self.n_future - 1:i + self.n_future, self.hull_col])  ##[17:18,0] un posicoin de la fila para la columna 0
            ### el 4/hull_col es la caracteritica elegida Close//EMA//EMA100//
    
        # ## separar Training y TEST
        self.trainX, self.trainX_test, self.trainY, self.trainY_test  = train_test_split(self.trainXX, self.trainYY, test_size = 0.05,shuffle = False)
        
        
        # trainX_test[-1,-1, hull_col]
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
        Descripcion: this method evaluates the convenience of the inversion measuring Profit and loss
        Currrent method asumes profit line three times bigger than stopLoss line
        
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
        history = self.model.fit(self.trainX, self.trainY, epochs=4, batch_size=16, validation_split=0.15, verbose=1) #batch=16
        
        plt.title("LSTM training")
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()
        
        
        #logging.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))
        logging.info('Loss: {}'.format(self.n_future))
        logging.info('mensaje 32')
        
        #Predicting...
        prediction = self.model.predict(self.trainX[0:1])
        #prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction
                                                                    # desde -n hasta el final. Cada elemento es un array bidimensional
        #Pido una predcicion para un array de fechas, me devuelve la predicion para cada una
        
        #Nos vamos n_daysforPredcition atras y calculamos la precidion a n_future (6) days despues.
        
        
        """
        try: 
            dff = yf.download(instrumento, startDate,endDate)
        except:
            logging.info('Ticker no existe'+instrumento)
            return dff    #Irá empty
        if dff.empty:
            return dff
        
        dff=self.featured_Data(dff)
        """
        
        return



    
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

    ## Primera RED
    myLSTMnet_1 =LSTMClass(previson_a_x_days=1)          #Creamos la clase
    #Preparo los datos
    myLSTMnet_1.dataPreparation_1('BBVA')
    #creo y entreno la NET
    myLSTMnet_1.LSTM_net_2()
 
    ## Segunda RED
    myLSTMnet_12 =LSTMClass(previson_a_x_days=12)          #Creamos la clase
    #Preparo los datos
    myLSTMnet_12.dataPreparation_1('BBVA')
    #creo y entreno la NET
    myLSTMnet_12.LSTM_net_2()


       
    #VALORES DEL IBEX 

    #for jjj in range(0,len(tickers_nasdaq)):    ##tickers_sp500
        #### FECHAS
        #start =dt.datetime(2000,1,1)
        ##startD =dt.datetime.today() - dt.timedelta(days=5*250)    #un año tiene 250 sesiones.
                
        

        ## Pinto en un grafico los parametros del ejercico.
    if True:
        plt.title("Datos del ejercicio")
        plt.text(0.1,0.6, 'Predicción a '+str(myLSTMnet_1.n_future)+' dias' ) 
        plt.text(0.1, 0.2, 'Ultimo dia ' + str( myLSTMnet_1.dfx.index[-1]))
        plt.text(0.1, 0.4, 'datos'  + str(myLSTMnet_1.cols))
        plt.legend()
        
        if (pdf_flag == True):
            plt.savefig("../docs/tmp/0_descricion.pdf")
        plt.show()
        
        
        ### VISUALIZAION DATOS TEST  (esto no son lo datos de test, TrainX_test deberia ser)
        
        pred_gap = np.zeros(myLSTMnet_1.n_future)
        muestra_gap =[]
        xx=[]
        
        muestreo=35  # minimo 20
        gapmuestras=np.zeros(muestreo-LSTMClass.n_past)
        gapprevion=np.zeros(myLSTMnet_1.n_future)
        gappostprevison= np.zeros(muestreo-myLSTMnet_1.n_future-LSTMClass.n_past)
        
        for ii in range (len(gapmuestras)):
            gapmuestras[ii] = np.nan
        for ii in range (len(gapprevion)):
            gapprevion[ii] = np.nan
        for ii in range (len(gappostprevison)):
            gappostprevison[ii] = np.nan 
        for ii in range (len(pred_gap)):
            pred_gap[ii] = np.nan      
        
        
        inicio= np.zeros(myLSTMnet_1.n_future+LSTMClass.n_past)    
        
        origen=LSTMClass.n_past+myLSTMnet_1.n_future
        
        
        for i in range(LSTMClass.n_past, len(myLSTMnet_1.trainX_test)-20, muestreo):  # vamos avanzando a saltos de longuitud muestreo
            
            prediction = myLSTMnet_1.model.predict(myLSTMnet_1.trainX_test[i-LSTMClass.n_past:i])
            prediction.shape = (LSTMClass.n_past)   #me devuleve 15//n_past previsiones a 6//n_future dias vista
            pred_gap=np.concatenate((pred_gap, prediction), axis=0)  #predcicion son n_past... en un futuro de n_futre muestras
            pred_gap=np.concatenate((pred_gap, gapmuestras), axis=0)
        
            ##xx=(trainX[i-n_past:i,0,hull_col])
            xx=(myLSTMnet_1.trainX_test[i-LSTMClass.n_past:i,-1,myLSTMnet_1.hull_col])    ##yyy=trainX_test[-(n_past+n_future):,-1,hull_col]   
            
            ##xx= trainXX[-1:, -n_past:, hull_col]  # coje el ultimo tramo de datos 
            xx.shape = ( LSTMClass.n_past)
            
            muestra_gap=np.concatenate((muestra_gap, xx  ), axis=0)
            muestra_gap=np.concatenate((muestra_gap, gapmuestras), axis=0)
        
            #print(trainX[i:i+n_days_for_prediction])
            ##print(xx)
            #fake =input()
    
    
        # Ahora ultimos datos, me reserve 20 en el for de arriba
        try: 
            prediction = myLSTMnet_1.model.predict(myLSTMnet_1.trainX_test[-LSTMClass.n_past:])
            prediction.shape = (LSTMClass.n_past)   #me devuleve 15//n_past previsiones a 6//n_future dias vista
        except:
            logging.debug('Error en la cantidad de los datos de test')
            
            #continue  #pondría un break para salir del todo del bucle for, continue ten lleva la sigueitne iteracion
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
       
        yyy=myLSTMnet_1.trainX_test[0:(len(myLSTMnet_1.trainX_test)-6),-1,myLSTMnet_1.hull_col]        ## yyy=trainXX[-1:,-n_past:,hull_col]   de todos los grupos de 14 observaciones, cojo la primero y la column 'hull' 
        plt.plot(yyy, label='datos REALES',color='pink')
        
        plt.plot(muestra_gap, color='red',label='Origen Prediccion')
        plt.plot(pred_gap, color='lightgreen', label='Predicción')
        plt.title(myLSTMnet_1.instrumento+' predicción a '+str(myLSTMnet_1.n_future)+' dias' ) 
        plt.text(0, np.amax(myLSTMnet_1.trainY_test) , 'Ultimo dia ' + str( myLSTMnet_1.dfx.index[-1]))
        #plt.text(0, (np.amin(trainY_test) - ((np.amax(trainY_test)-np.amin(trainY_test))/3.64))-0.1, 'datos'  + str(cols))
        plt.legend()
        
        if (pdf_flag == True):
            plt.savefig("../docs/tmp/"+myLSTMnet_1.instrumento+".pdf")
        plt.show()
            
        #break
    
    print('This is it................ ')
    
    
    
else:
    """Esta parte del codigo se ejecuta si uso como libreria"""    
    print ('formato libreria')
    print ('version: ',versionVersion)    