# Importare le librerie necessarie
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Caricare i dati
df = pd.read_csv('./data/input/netflix_titles.csv')

# Visualizzare le prime righe per verificare il caricamento corretto
print(df.head())

# 2. Pre-processing
# Selezionare solo le colonne rilevanti per l'analisi
df = df[['type', 'release_year', 'rating', 'duration', 'listed_in', 'description']]

# Rimuovere righe con valori nulli nelle colonne critiche
df = df.dropna(subset=['description', 'listed_in', 'rating', 'duration'])

# Codificare la variabile target `listed_in` (il genere) per trasformarla in numeri
# Nota: qui `listed_in` può contenere più generi, quindi estrarremo solo il primo genere per semplicità
df['genre'] = df['listed_in'].apply(lambda x: x.split(',')[0].strip())

# Convertire il testo in numeri usando TfidfVectorizer per la colonna 'description'
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
description_matrix = vectorizer.fit_transform(df['description'])

# Codificare le variabili categoriche come 'type' e 'rating'
df['type_encoded'] = LabelEncoder().fit_transform(df['type'])
df['rating_encoded'] = LabelEncoder().fit_transform(df['rating'])

# Processare la durata come numerica (estraendo il numero di minuti o stagioni)
df['duration_numeric'] = df['duration'].str.extract('(\d+)').astype(float)

# Combinare tutte le caratteristiche
import numpy as np
X = np.hstack((description_matrix.toarray(), df[['type_encoded', 'rating_encoded', 'release_year', 'duration_numeric']].values))

# Variabile target
y = df['genre']

# 3. Suddivisione in train e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Addestramento del modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Valutazione del modello
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
