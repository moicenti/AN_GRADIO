import gradio as gr
import json
import pandas as pd
import numpy as np
import os

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
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

np.random.seed(0)
indices = np.random.permutation(len(X_norm))
train_size = int(0.7 * len(X_norm))
train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, y_train = X_norm[train_idx], y[train_idx]
X_test, y_test = X_norm[test_idx], y[test_idx]

modelo_path = "./modelo_centroides.npz"
centroides = {}

if os.path.exists(modelo_path):
    modelo = np.load(modelo_path)
    centroides = {int(k): modelo[k] for k in modelo}
else:
    for clase in np.unique(y_train):
        centroides[clase] = X_train[y_train == clase].mean(axis=0)
    np.savez(modelo_path, **{str(k): v for k, v in centroides.items()})

def clasificar_semilla(area, perimeter, compactness, length_kernel, width_kernel, asymmetry, groove_length):
    muestra = np.array([
        area, perimeter, compactness, length_kernel,
        width_kernel, asymmetry, groove_length
    ])
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    muestra_norm = (muestra - min_vals) / (max_vals - min_vals)
    distancias = {clase: np.linalg.norm(muestra_norm - centroide) for clase, centroide in centroides.items()}
    prediccion = min(distancias, key=distancias.get)
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
        gr.Markdown("## Clasificador de Semillas por Centroide")
        area = gr.Number(label="Área (mm²)")
        perimeter = gr.Number(label="Perímetro (mm)")
        compactness = gr.Number(label="Compacidad")
        length_kernel = gr.Number(label="Longitud del grano (mm)")
        width_kernel = gr.Number(label="Ancho del grano (mm)")
        asymmetry = gr.Number(label="Coef. de asimetría")
        groove_length = gr.Number(label="Longitud del surco (mm)")
        salida_semilla = gr.Textbox(label="Clase Predicha")
        btn_semilla = gr.Button("Clasificar")
        btn_semilla.click(
            fn=clasificar_semilla,
            inputs=[area, perimeter, compactness, length_kernel, width_kernel, asymmetry, groove_length],
            outputs=salida_semilla
        )

app.launch(share=True)
