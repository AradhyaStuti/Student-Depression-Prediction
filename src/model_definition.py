import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Define the structure of the neural network
class DepressionModel(nn.Module):
    def __init__(self, input_size):
        super(DepressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Primera capa
        self.fc2 = nn.Linear(64, 32)          # Segunda capa
        self.fc3 = nn.Linear(32, 2)           # Capa de salida (clasificación binaria)
        
    def forward(self, x):
        x = self.fc1(x)   # Aplicar ReLU a la capa 1
        x = F.relu(self.fc2(x))   # Aplicar ReLU a la capa 2
        x = self.fc3(x)   # Aplicar la capa de salida
        return F.softmax(x, dim=1)  # Retorna las probabilidades para cada clase
    
# Create a PyTorch Dataset
class DepressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the training function
def train_and_export_model():
    # Load data from the CSV file and filter the columns we want
    df = pd.read_csv('data/Student Depression Dataset.csv')
    columns_to_keep = ['Gender', 'Age', 'Work/Study Hours', 'Academic Pressure',
                      'Financial Stress', 'Study Satisfaction', 'Sleep Duration',
                      'Dietary Habits', 'Have you ever had suicidal thoughts ?',
                      'Family History of Mental Illness', 'Depression']
    df = df[columns_to_keep]
    df = df.dropna()

# Split the data into features (X) and target (y)
    X = df.drop(columns=['Depression'])
    y = df['Depression']

   # Separate the categorical columns from the numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

   # Apply One-hot encoding to the categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X[categorical_cols])
    joblib.dump(encoder, 'model_files/encoder.joblib')
    encoded_cat_data = encoder.fit_transform(X[categorical_cols])

 # Standardize the numerical data
    scaler = StandardScaler()
    scaler.fit(X[numerical_cols])
    joblib.dump(scaler, 'model_files/scaler.joblib')
    scaled_num_data = scaler.fit_transform(X[numerical_cols])

   # Combine the processed data
    X_processed = np.hstack((encoded_cat_data, scaled_num_data))

    # Split the data into training and test sets with 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=17)

    # Convert the data to PyTorch tensors (tensorization)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create the data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

   # Create the data loader
    input_size = X_train.shape[1]  # Número de características en el dataset procesado
    print(input_size)
    model = DepressionModel(input_size)

   # Train the model
    criterion = nn.CrossEntropyLoss()  # Para clasificación binaria
    optimizer = torch.optim.SGD(params=model.parameters(), 
                                lr=0.1)
    # Training phase
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Ponemos a cero los gradientes
            outputs = model(inputs)  # Pasamos los datos por el modelo
            loss = criterion(outputs, labels)  # Calculamos la pérdida
            loss.backward()  # Propagación hacia atrás
            optimizer.step()  # Actualizamos los pesos
            
            # Accuracy tracking
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

 # Evaluate the model
    model.eval()  # Ponemos el modelo en modo evaluación
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Precisión en prueba: {100 * correct / total:.2f}%")

    # Export the model
    torch.save(model.state_dict(), 'model_files/depression_model.pth')

def predict(respuestas: dict) -> float:

   # Convert the responses into a DataFrame
    df = pd.DataFrame.from_dict(respuestas)

    # Load our model, encoder, and scaler that we saved during training
    encoder = joblib.load('model_files/encoder.joblib')
    scaler = joblib.load('model_files/scaler.joblib')
    model = DepressionModel(input_size=20)
    model.load_state_dict(torch.load('model_files/depression_model.pth'))
    model.eval()

    print(df.dtypes)

    # Separate the categorical columns from the numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['float64']).columns

    # We separate the categorical columns from the numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['float64']).columns

    
    # We apply One-hot encoding to the categorical columns using the loaded encoder
    encoded_cat_data = encoder.transform(df[categorical_cols])

    # We standardize the numerical data using the loaded scaler
    scaled_num_data = scaler.transform(df[numerical_cols])

    #We combine the processed data
    df_processed = np.hstack((encoded_cat_data, scaled_num_data)) 

    torched_data = torch.tensor(df_processed, dtype=torch.float32)
    with torch.no_grad():
        output = model(torched_data)

    depression_probability = output[:, 1].item()

    return depression_probability









