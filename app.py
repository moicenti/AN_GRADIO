import gradio as gr
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

with open("modelo.json", "r") as f:
    modelo_seguro = json.load(f)

def predict_seguro(age, bmi, smoker, children):
    smoker_code = 1 if smoker == "Sí" else 0
    cost = (
        modelo_seguro["intercepto"] +
        modelo_seguro["pesoEdad"] * age +
        modelo_seguro["pesoBMI"] * bmi +
        modelo_seguro["pesoFumador"] * smoker_code +
        modelo_seguro["pesoHijos"] * children
    )
    return f"${cost:.2f}"


df = pd.read_csv("./seeds.csv", sep=r"\s+", header=None)
df.columns = [
    "Area", "Perimeter", "Compactness", "Length of Kernel",
    "Width of Kernel", "Asymmetry Coefficient", "Length of Kernel Groove", "Variety"
]
X = df.drop("Variety", axis=1).values
y = df["Variety"].values
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # normalización min-max

np.random.seed(0)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_logistica(y, h):
    m = y.shape[0]
    return -1/m * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))

def entrenar_modelo_logistico(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    X_bias = np.c_[np.ones((m, 1)), X]
    theta = np.zeros(n + 1)
    historial = []

    for _ in range(epochs):
        z = np.dot(X_bias, theta)
        h = sigmoid(z)
        grad = np.dot(X_bias.T, (h - y)) / m
        theta -= lr * grad
        historial.append(loss_logistica(y, h))

    return theta, historial

def one_vs_rest(X, y, clases, lr=0.1, epochs=1000):
    modelos = {}
    historiales = {}
    for c in clases:
        y_binaria = (y == c).astype(int)
        theta, hist = entrenar_modelo_logistico(X, y_binaria, lr, epochs)
        modelos[c] = theta
        historiales[c] = hist
    return modelos, historiales

clases = np.unique(y_train)
modelos_ovr, historiales_log = one_vs_rest(X_train, y_train, clases)

def predecir_clase(muestra, modelos):
    X_muestra = np.insert(muestra, 0, 1)  # bias
    probs = {clase: sigmoid(np.dot(X_muestra, theta)) for clase, theta in modelos.items()}
    prediccion = max(probs, key=probs.get)
    return f"Clase predicha: {prediccion}"

with gr.Blocks(title="App Multi-Modelo") as app:
    gr.Markdown("# Proyecto 1 con Dataset 3 y 6")

    with gr.Tab("Seguro Médico"):
        gr.Markdown("## Estimador de Costo de Seguro")
        edad = gr.Slider(18, 100, step=1, label="Edad (años)")
        bmi = gr.Number(label="BMI (kg/m²)")
        fumador = gr.Radio(["Sí", "No"], label="¿Fumador?")
        hijos = gr.Number(label="Número de hijos", precision=0, minimum=0, maximum=10, step=1, value=0)
        salida_seguro = gr.Textbox(label="Costo estimado")
        btn_seguro = gr.Button("Calcular")
        btn_seguro.click(fn=predict_seguro, inputs=[edad, bmi, fumador, hijos], outputs=salida_seguro)

    with gr.Tab("Clasificación de Semillas"):
        gr.Markdown("## Clasificador de Semillas con Regresión Logística OvR")
        area = gr.Number(label="Área (mm²)")
        perimeter = gr.Number(label="Perímetro (mm)")
        compactness = gr.Number(label="Compacidad")
        length_kernel = gr.Number(label="Longitud del grano (mm)")
        width_kernel = gr.Number(label="Ancho del grano (mm)")
        asymmetry = gr.Number(label="Coef. de asimetría")
        groove_length = gr.Number(label="Longitud del surco (mm)")
        salida_semilla = gr.Textbox(label="Clase Predicha")
        btn_semilla = gr.Button("Clasificar")

        def wrapper_pred(area, perimeter, compactness, length_kernel, width_kernel, asymmetry, groove_length):
            muestra = np.array([area, perimeter, compactness, length_kernel,
                                width_kernel, asymmetry, groove_length])
            muestra = (muestra - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # normalización
            return predecir_clase(muestra, modelos_ovr)

        btn_semilla.click(
            fn=wrapper_pred,
            inputs=[area, perimeter, compactness, length_kernel, width_kernel, asymmetry, groove_length],
            outputs=salida_semilla
        )

app.launch()
