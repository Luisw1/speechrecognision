"""
Created on Tue May 25 15:06:16 2021

@author: Luis

Important stats:

    epoch: Numbers of time the neuronal network is training
    path: path to dataset
    
    Functions:
        __init__(): constructor: by starting the programm starting data_arrangment() and plot_graphs()
        data_arrangment(): arranging data in test and training data.
        create_model(): compiling a deep neural network
        plot_graphs(): plotting graph1(Training acurazy) and confusion graph for test data 
        test(): testing one data file. 
        plot_mel_MFCC(audio_file): evaluating a MFCC on a wavfile. But only wav files above 600kb
    """
    
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import random as rn
import warnings
with warnings.catch_warnings(): # for warnings spam by tensorflow
    warnings.filterwarnings("ignore",category=FutureWarning) 
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
import seaborn as sns
import speech_recognition as sr
import shutil
import time
import augly.audio as audaugs
import zaf

class Spracherkennung():
    def __init__(self): 
        self.model = None
        
    path = 'Audio/recordings'
    path_testaudio = 'Audio/testaudio'
    path_audioaugmentation = 'Audio/augmentation'
    
    epoch = 50
    earlystopping_epochs = 10
    global data_augmentation
    
    def getAllFilenamesFromFolder(self, path):
        print(f"Getting all files from {path}")
        liste = []
        for flist in os.listdir(path):
            liste.append(flist)
        rn.shuffle(liste)
        print(f"Number of samples in databank: {len(liste)}")  
        return liste
        
    def new_data_augmentations(self, rg =3.0):
        start = time.perf_counter()
        # aug_audio is a NumPy array with your augmentations applied!
        def data_augmentation_pitch(data, sr, rg):
            for j in np.arange(-rg, rg, 0.5):
                aug_audio, sr = audaugs.pitch_shift(audio= data, sample_rate= sr, n_steps= float(j),
                output_path= self.path_audioaugmentation + f"/{self.digit}_{self.person}_pitch_({j})_{self.sample}")

        def data_augmentation_time(data, sr):
            for j in np.arange( 0.5 , 1.5, 0.125):
                aug_audio, sr = audaugs.time_stretch(audio= data, sample_rate= sr, rate= float(j),
                output_path= self.path_audioaugmentation + f"/{self.digit}_{self.person}_time_({j})_{self.sample}")
                    
        def data_augmentation_noise(data, sr, rg):
            for j in np.arange(-rg, rg, 0.5):
                aug_audio, sr = audaugs.add_background_noise(audio= data, sample_rate= sr, snr_level_db= float(j+6),
                output_path= self.path_audioaugmentation + f"/{self.digit}_{self.person}_noise_({j})_{self.sample}")
                
        filenames = self.getAllFilenamesFromFolder(self.path)
        
        for f in os.listdir(self.path_audioaugmentation):
            os.remove(os.path.join(self.path_audioaugmentation, f))
            
        for i in range(len(filenames)): #for train data
            struct = filenames[i].split('_')
            self.digit = struct[0]
            self.sample = struct[2] 
            self.person = struct[1]
            wav, sr = librosa.load(os.path.join(self.path , filenames[i]))
            data_augmentation_pitch(wav, sr, rg)
            data_augmentation_time(wav, sr)
            data_augmentation_noise(wav, sr, rg)    
        
        finish = time.perf_counter()  
        print(f'Data processing finished in {round(finish-start, 2)} second(s); {round(finish-start, 2)}/60 minutes')
                
    def process_mfcc_train(self, audio_wav, digit):
        mfcc = librosa.feature.mfcc(audio_wav)
        padded_mfcc = self.pad2d(mfcc,40)
        self.train_mfccs.append(padded_mfcc)
        self.train_y.append(digit)
    
    def process_mfcc_test(self, audio_wav, digit):
        mfcc = librosa.feature.mfcc(audio_wav)
        padded_mfcc = self.pad2d(mfcc,40)
        self.test_mfccs.append(padded_mfcc)
        self.test_y.append(digit)
    
    def data_processing(self):  
        
        start = time.perf_counter()
        self.pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a,
                            np.zeros((a.shape[0],i - a.shape[1]))))
        
        def split_to_percentage(data, percentage):
            return  data[0: int(len(data)*percentage)] , data[int(len(data)*percentage):]     
        
        print("Arranging data...")
        files = self.getAllFilenamesFromFolder(self.path_audioaugmentation)
        
        train_data, test_data = split_to_percentage(files, 0.8)
        
        self.testwithdata = train_data[0:10]
        self.train_mfccs = []
        self.train_y = []
        self.test_mfccs = []
        self.test_y = []
        
        for i in range(len(train_data)): #for train data
            struct = train_data[i].split('_')
            self.digit = struct[0]
            self.sample = struct[4]
            wav, sr = librosa.load(os.path.join(self.path_audioaugmentation, train_data[i]))
            self.process_mfcc_train(wav, self.digit)

        print("moving on to test data")
        for i in range(len(test_data)): #for test data
            struct = test_data[i].split('_')
            self.digit = struct[0]
            self.sample = struct[4]
            wav, sr = librosa.load(os.path.join(self.path_audioaugmentation, test_data[i]))
            self.process_mfcc_test(wav, self.digit)

        self.train_mfccs = np.array(self.train_mfccs)
        self.train_y = to_categorical(np.array(self.train_y))
        test_mfccs = np.array(self.test_mfccs)
        self.test_y = to_categorical(np.array(self.test_y))
        self.train_X_ex = np.expand_dims(self.train_mfccs, -1)
        self.test_X_ex = np.expand_dims(test_mfccs, -1)     

        finish = time.perf_counter()   
        print(f'Data processing finished in {round(finish-start, 2)} second(s); {round(finish-start, 2)}/60 minutes')
        
    def create_model(self):
        
        #Create a deep neural network
        print(f"creating a deep neutral network with {self.epoch} epochs")
        ip = tf.keras.Input(shape=self.train_X_ex[0].shape)
        
        m = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(ip)
        m = tf.keras.layers.BatchNormalization()(m)
        m = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(m)
        m = tf.keras.layers.BatchNormalization()(m)
        m = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(m)
        m = tf.keras.layers.Dropout(0.2)(m)
        
        m = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(m)
        m = tf.keras.layers.BatchNormalization()(m)
        m = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(m)
        m = tf.keras.layers.BatchNormalization()(m)
        m = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(m)
        m = tf.keras.layers.Dropout(0.2)(m)
        
        # m = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(m)
        # m = tf.keras.layers.BatchNormalization()(m)
        # m = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(m)
        # m = tf.keras.layers.BatchNormalization()(m)
        # m = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(m)
        # m = tf.keras.layers.Dropout(0.2)(m)
            
        m = tf.keras.layers.BatchNormalization()(m)
        m = tf.keras.layers.Flatten()(m)
        
        m = tf.keras.layers.Dense(64, activation='relu')(m)
        m = tf.keras.layers.Dense(32, activation='relu')(m)
        m = tf.keras.layers.BatchNormalization()(m)
        m = tf.keras.layers.Dropout(0.2)(m)
        op = tf.keras.layers.Dense(10, activation='softmax')(m)
        self.model = tf.keras.Model(inputs=ip, outputs=op)
        
        self.model.summary()
            
        #Compile and fit the neural network
        self.model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                 metrics=['accuracy'])
        return self.model
    
    def classification(self):
        start = time.perf_counter()
        #creating model
        self.create_model()

        # Create a callback that saves the model's weights
        checkpoint_path = "cp.ckpt"
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
            
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        save_best_only=True, mode='max', monitor='val_accuracy', verbose=1)
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        
        callbacks_list = [cp_callback, early_stop]
       
        self.history = self.model.fit(self.train_X_ex,
              self.train_y,
              epochs = self.epoch,
              batch_size=32,
              validation_data=(self.test_X_ex, self.test_y),
              callbacks=callbacks_list)
        
        print("Saving the deep neural network:")
          
        # Loads the weights
        self.model.load_weights(checkpoint_path).expect_partial()   
        
        # Re-evaluate the model
        loss, acc = self.model.evaluate(self.test_X_ex, self.test_y, verbose=2, )
        print("Restored ceckpoint model, accuracy: {:5.2f}%".format(100 * acc))
        print("Restored ceckpoint model, loss: {:5.2f}%".format(100 * loss))
        
        finish = time.perf_counter()
        print(f'Finished in {round(finish-start, 2)} second(s)')
        print("finish with initialing. Ready for tests :)")
        
    def test(self, path, test_file):#mit eigenen Dateien testen
    
        print("Test:")
        print(f"Testfile: {test_file}")
        
        sample_mfccs = []
        sample_wav, sr = librosa.load(os.path.join(path, test_file))
        sample_mfcc = librosa.feature.mfcc(sample_wav)
        sample_padded_mfcc = self.pad2d(sample_mfcc,40)
        sample_mfccs.append(sample_padded_mfcc) 
        sample_X_ex = np.expand_dims(sample_mfccs, -1)
        
        predictions = self.model.predict(sample_X_ex)
        predictions = predictions.astype(float)
        
        for i in predictions:
            if i.any() != 0 :
                print(f"Predicted number: {np.argmax(predictions)}")
            else:
                print("Predicted number: System couldn't predict a number")
        
    def plot_graphs(self):
        
        print("Plotting graphs...")
        plt.plot(self.history.history['loss'], label='Train loss')
        plt.plot(self.history.history['val_loss'], label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        
        #Display the Confusion Matrix for test data
        y_pred= self.model.predict(self.test_X_ex)
        y_p= np.argmax(y_pred, axis=1)
        y_pred=y_pred.astype(int)
        y_t=np.argmax(self.test_y, axis=1)
        confusion_mtx = tf.math.confusion_matrix(y_t, y_p) 
        plt.figure(figsize=(5, 5))
        sns.heatmap(confusion_mtx, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()

    speech_engine = sr.Recognizer()
    def from_microphone(self):
        with sr.Microphone() as micro:
            print("Recording...")
            self.audio = self.speech_engine.record(micro, duration=2)
           
            # write audio to a WAV file
            with open("Audio/testaudio/output.wav", "wb") as f:
                self.audio = f.write(self.audio.get_wav_data())
           
                print("Recognition...")
                SE.test("Audio/testaudio", "output.wav")
  
        os.remove("Audio/testaudio/output.wav")
    
    def testownsamples(self):
        for flist in os.listdir(self.path_testaudio):
            SE.test(self.path_testaudio, flist)      
    
    def testrandomtestdata(self):
        for f in range(len(self.testwithdata)):
            SE.test(self.path, self.testwithdata[f])
            
    def durchgang(self):
        # SE.new_data_augmentations()
        # SE.data_processing()
        # SE.classification()
        # SE.plot_graphs()
        # SE.testrandomtestdata()
        # SE.testownsamples()
        SE.from_microphone()
        
if __name__ == "__main__":
    SE = Spracherkennung()
    SE.durchgang()
    
            
        # print(f"Test_mfccs:  {test_mfccs.shape}")
        # print(f"Test_X_ex: {self.test_X_ex.shape}")
        # print(f"Length of test_y: {len(self.test_y)}")
        
        # print(f"Train_mfccs: {self.train_mfccs.shape}")
        # print(f"Train_X_ex: {self.train_X_ex.shape}")
        # print(f"Length of self.train_y: {len(self.train_y)}")
        