from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
#import requests

app = Flask(__name__)


def get_numeric_and_categorical_columns(df, numeric_as_category = None):
    if numeric_as_category is None:
        numeric_as_category = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(numeric_as_category).tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.union(numeric_as_category).tolist()
    return numeric_cols, categorical_cols

def desglose_fechas(df, columna_fecha, tipo, sufijo):
    if columna_fecha not in df.columns:
        print(f"La columna '{columna_fecha}' no existe en el DataFrame.")
        return

    if tipo == 'dia':
        df[f'dia_{sufijo}'] = df[columna_fecha].dt.day_name()
    elif tipo == 'mes':
        df[f'mes_{sufijo}'] = df[columna_fecha].dt.month_name()
    else:
        print("El parámetro 'tipo' debe ser 'dia' o 'mes'.")

#feature_names = scaler.feature_names_in_.tolist()

# Datos de login
USERNAME = "admin"
PASSWORD = "dechava"

# Cargar modelo y scaler

modelo = joblib.load('svm_model.pkl')

# AZURE_ENDPOINT = "https://svm-model-azure-endpoint.azurewebsites.net/score"
# AZURE_API_KEY = "your_api_key_here"  # Reemplaza con tu clave de API


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == USERNAME and password == PASSWORD:
            return redirect(url_for("formulario"))
        else:
            return render_template("login.html", error="Credenciales incorrectas")
    return render_template("login.html")

@app.route("/formulario", methods=["GET", "POST"])
def formulario():
    if request.method == "POST":
        # Obtener datos del formulario
        values = [
            (request.form["h_res_fec"]),
            (request.form["h_fec_lld"]),
            int(request.form["h_num_adu"]),
            int(request.form["h_num_men"]),
            int(request.form["h_num_noc"]),
            int(request.form["ID_Tipo_Habitacion"]),
            int(request.form["ID_Paquete"]),
            int(request.form["h_can_res"])
        ]

        df = pd.DataFrame([values], columns=['h_res_fec', 'h_fec_lld', 'h_num_adu', 'h_num_men', 'h_num_noc', 'ID_Tipo_Habitacion', 'ID_Paquete', 'h_can_res'])
        #Formato de modelo



        df['h_res_fec'] = pd.to_datetime(df['h_res_fec'])
        df['h_fec_lld'] = pd.to_datetime(df['h_fec_lld'])
        df['ID_Tipo_Habitacion'] = df['ID_Tipo_Habitacion'].replace('Otro', '0')
        df['h_can_res'] = df['h_can_res'].replace('DI', '05')
        
        desglose_fechas(df, 'h_res_fec', 'dia', 'reservacion')
        desglose_fechas(df, 'h_fec_lld', 'dia', 'entrada')
        desglose_fechas(df, 'h_res_fec', 'mes', 'reservacion')
        desglose_fechas(df, 'h_fec_lld', 'mes', 'entrada')
        df.drop(columns=['h_res_fec', 'h_fec_lld'], inplace=True)

        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6}

        df['dia_reservacion'] = df['dia_reservacion'].map(day_map).astype(int)
        df['dia_entrada'] = df['dia_entrada'].map(day_map).astype(int)


        mes_map = {
            'January': 0, 'February': 1, 'March': 2, 'April': 3,
            'May': 4, 'June': 5, 'July': 6, 'August': 7,
            'September': 8, 'October': 9, 'November': 10, 'December': 11
        }

        df['mes_reservacion'] = df['mes_reservacion'].map(mes_map)
        df['mes_entrada'] = df['mes_entrada'].map(mes_map)
        
        numerical_cols, categorical_cols = get_numeric_and_categorical_columns(df, numeric_as_category=['ID_Paquete', 'mes_entrada', 'mes_reservacion', 'dia_entrada', 'dia_reservacion'])
        df = df[numerical_cols + categorical_cols]

        # Obtener los 2 valores más frecuentes para 'ID_Tipo_Habitacion'
        top2_habitacion = df['ID_Tipo_Habitacion'].value_counts().nlargest(2).index.tolist()

        # Reemplazar los valores que no están en top2 con 'Otro_Habitacion'
        df['ID_Tipo_Habitacion'] = df['ID_Tipo_Habitacion'].apply(lambda x: x if x in top2_habitacion else 'Otro_Habitacion')

        # Convertir 'h_can_res' a numérico para poder encontrar los 2 valores más frecuentes numéricamente
        df['h_can_res_numeric'] = pd.to_numeric(df['h_can_res'], errors='coerce')

        # Obtener los 2 valores más frecuentes para 'h_can_res' (usando la columna numérica)
        top2_res = df['h_can_res_numeric'].dropna().value_counts().nlargest(2).index.tolist()

        # Convertir los valores numéricos de top2_res de vuelta a string para compararlos con la columna original 'h_can_res'
        top2_res_str = [str(int(x)) for x in top2_res]

        # Reemplazar los valores en 'h_can_res' que no están en top2 con 'Otro_Res'
        df['h_can_res'] = df['h_can_res'].apply(lambda x: x if str(x) in top2_res_str else 'Otro_Res')

        # Eliminar la columna temporal numérica
        df.drop(columns=['h_can_res_numeric'], inplace=True)

        # Actualizar las listas de columnas numéricas y categóricas después de la modificación
        numerical_cols, categorical_cols = get_numeric_and_categorical_columns(df, numeric_as_category=['ID_Paquete', 'mes_entrada', 'mes_reservacion', 'dia_entrada', 'dia_reservacion', 'ID_Tipo_Habitacion', 'h_can_res'])

        df[categorical_cols] = df[categorical_cols].astype(str)
        df = df[numerical_cols + categorical_cols]
        ##################
 
        prediction = modelo.predict(df)
        # headers = {
        #     'Content-Type': 'application/json', 
        #     'Authorization': f'Bearer {AZURE_API_KEY}' 
        # }

        # data_to_send = {df.to_dict(orient='records')}
        # response = requests.post(AZURE_ENDPOINT, headers=headers, json=data_to_send) 

        # if response.status_code == 200: 
        #     prediction = response.json()
        #     return render_template("formulario.html", prediction=int(prediction[0]))
        # else:
        #     return render_template("formulario.html", error="Error al obtener la predicción del modelo en Azure")

        return render_template("formulario.html", prediction=int(prediction[0]))
    
    return render_template("formulario.html")

if __name__ == "__main__":
    app.run(debug=True)