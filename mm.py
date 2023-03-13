with open("kc_house_data.csv", "r") as f:
    data = f.readlines()
house_data = [line.strip().split(",") for line in data]


# %%
import numpy as np
np_array = np.array(house_data[1:])

# %%


# %% [markdown]
# ##Task 1.2 
# ####Ajustar un modelo polinomial (regresión lineal) en base al juego de datos cargado de forma matricial que relaciona las variables de precio con los pies cuadrados del espacio habitable interior de los apartamentos (price - sqft living).

# %%
# Obteniendo el tamaño y el límite del dataset
DATASET_SIZE, DATASET_X_LIM = np_array.shape

# %%
print(np.max(np_array[:, 5]))