import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

df = pd.read_csv('lung.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())

df['Level'] = df['Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Gender'] = df['Gender'].map({1: 0, 2: 1})  

print("\nLung.csv için Min ve Max Değerler:")
for column in df.columns:
    min_value = df[column].min()
    max_value = df[column].max()
    print(f"{column}: Min = {min_value}, Max = {max_value}")


X = df.iloc[:, 2:-1].values  
y = df['Level'].values

print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i : class_weights[i] for i in range(len(class_weights))}


model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  
model.add(Dropout(0.4))  
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping], class_weight=class_weights)  

model.save('lung_cancer_model.keras')

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(history.history['accuracy'], label='Training Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Training Loss')
ax[1].plot(history.history['val_loss'], label='Validation Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.tight_layout()
plt.savefig('lung_training_results.png')
plt.show()
