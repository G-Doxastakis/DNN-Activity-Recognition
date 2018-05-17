import motionDataset
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

data, class_names = motionDataset.load('MotionDatasets/AcRec/PhoneGyroProcessed.csv')
sequence_len = 200
idx_train = range(sequence_len, round(len(data)*0.6))
idx_val = range(round(len(data)*0.6), round(len(data)*0.8))
idx_test = range(round(len(data)*0.8), len(data))

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=9, padding='causal', dilation_rate=1, activation='relu', input_shape=(sequence_len, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=5, padding='causal', dilation_rate=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=5, padding='causal', dilation_rate=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(128, activation='tanh'))

model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

stop = EarlyStopping(monitor='val_acc', patience=5)
log = TensorBoard()
model.fit_generator(generator=motionDataset.randgenerator(data, idx_train, sequence_len),
                    validation_data=motionDataset.randgenerator(data, idx_val, sequence_len),
                    steps_per_epoch=20000, validation_steps=6000, epochs=100, callbacks=[stop, log])

scores = model.evaluate_generator(generator=motionDataset.randgenerator(data, idx_test, sequence_len), steps=25000)
print("Accuracy: %.2f%%" % (scores[1]*100))
motionDataset.confusionMatrix(model, motionDataset.randgenerator(data, idx_test, sequence_len), 25000, class_names)
#model.save('CNN_model_phone.h5')
motionDataset.exportmodel('CNNLSTM_Phone_gyro','conv1d_1_input','dense_2/Softmax')