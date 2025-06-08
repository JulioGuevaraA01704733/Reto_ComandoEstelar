##################################

# IMPORTANCION DE LIBRERÍAS

##################################

import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
# Ignorar alertas
import warnings
warnings.simplefilter("ignore")
# Manejar los meses y fechas en español
import locale
locale.setlocale(locale.LC_TIME, 'Spanish_Spain.1252')

##################################

# PRE-PROCESAMIENTO DE LOS DATOS

##################################

# Leer dataset
df = pd.read_csv("C:/workspace/Reto/Datos/iar_Reservaciones.csv")

# Eliminar columnas con valores confidenciales
df.drop(['h_correo_e', 'h_nom'], axis = 'columns', inplace = True)

# Sustituir valores 'vacios' (espacios) con nulos
df.replace(r'^\s*$', np.nan, regex=True, inplace = True)

# Formatear correctamente las fechas que se usaran
columnas_fechas = ['h_res_fec', 'h_fec_lld']
for fecha in columnas_fechas:
    df[fecha] = pd.to_datetime(df[fecha], format='%Y%m%d', errors='coerce')

# Eliminar columnas por tener muchos valores nulos
df.drop(columns=['h_cod_reserva', 'h_codigop'], axis='columns', inplace = True)

# Eliminar registros que tienen algun campo con valor nulo
df.dropna(inplace = True)

# Quedarnos solo con las columnas que utilizaremos
cols_relevantes = ['ID_Reserva', 'ID_estatus_reservaciones', 'ID_Agencia', 'ID_Paquete', 'ID_Tipo_Habitacion', 
                   'h_tot_hab', 'h_num_men', 'h_num_adu', 'h_num_per', 
                   'h_res_fec', 'h_fec_lld', 'h_num_noc',
                   'h_tfa_total']
df = df[cols_relevantes]

# Extraer el mes de las fechas que se usaran
for fecha in columnas_fechas:
    df[f'{fecha}_mes'] = df[fecha].dt.strftime("%B").astype("category")

# Establecer indica
df.set_index(df['ID_Reserva'], inplace = True)
df.drop(['ID_Reserva'], axis = 'columns', inplace = True)

# Sustituir todos los valores de no. de personas por la suma de no. de menores y no. de adultos
df.loc[df['h_num_per'], 'h_num_per'] = df['h_num_men'] + df['h_num_adu']

# Crear nuevas columnas indicando si es fin de semana o entre semana
fechas = ['reservacion', 'llegada']

for i in range(len(fechas)):
    df[f'entre_fin_{fechas[i]}'] = df[columnas_fechas[i]].dt.weekday.apply(lambda x: 'Fin de semana' if x >= 5 else 'Entre semana')


# Para reducir la cardinalidad de las columnas, quedándonos con el porcentaje del umbral 
# y agrupando lo restante en una categoría nueva llamada "Otro"
umbral = 0.90
cols_reduccion_cardinalidad = ['ID_Agencia', 'ID_Tipo_Habitacion']

for col in cols_reduccion_cardinalidad:
    freqs = df[col].value_counts(normalize=True)
    top_cats = freqs.cumsum()[freqs.cumsum() <= umbral].index
    df[col + '_reducida'] = df[col].where(df[col].isin(top_cats), other='Otro')

df.drop(columns = cols_reduccion_cardinalidad, inplace = True)
df.columns = [col.replace('_reducida', '') for col in df.columns]

# Conservar solamente los registros en los que la tarifa total y el no. de noches es diferente de 0
df = df[(df['h_tfa_total'] != 0) & (df['h_num_noc'] != 0)]

# Generar una columna nueva para tener la tarifa por noche
df['tarifa_x_noche'] = df['h_tfa_total'] / df['h_num_noc']

##################################

# GENERAR EL DATASET PARA K-MODES

##################################

# Esta primera columna se usa para crear el nuevo dataframe de valores categoricos
df_categ = pd.DataFrame(df['ID_estatus_reservaciones'].astype(str))

# Conservamos el indice del dataframe original para el analisis de clústers
df_categ = df_categ.set_index(df.index)  

# Columnas que se agregarán al nuevo dataframe de valores categóricos y no requieren ningún tipo de transformación
Columns_to_categ = ['h_tot_hab', 'ID_Paquete', 'h_num_men', 'h_num_adu', 'h_num_per', 'entre_fin_llegada',
                    'entre_fin_reservacion', 'ID_Agencia', 'ID_Tipo_Habitacion', 'h_res_fec_mes', 'h_fec_lld_mes']

for column in Columns_to_categ:
    df_categ[column] = df[column].astype(str)

# Dividir los registros categoricamente por cuartiles (cantidades iguales de valores por rango),
# basados en la tarifa por noche
df_categ['tarifa_x_noche'] = pd.qcut(df['tarifa_x_noche'], q=4, labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])

##################################

# APLICAR ANALISIS K-MODES

##################################

# Aplicar K-Modes con K=4
km = KModes(n_clusters=4, init="Huang", n_init=5, verbose=0, random_state=42)
clusters = km.fit_predict(df_categ)

# Agregar la columna de clusters al DataFrame
df_categ['cluster'] = clusters

# añadimos los resultados al dataframe completo
df = df.join(df_categ[['cluster']], how='inner')

##################################

# EXTRACCION DE INDICADORES PARA
# EL DASHBOARD

##################################

# Columnas a las que se le aplicara el analisis numerico
numericas = ['h_tot_hab', 'tarifa_x_noche', 'h_num_per', 'h_num_adu', 'h_num_men', 'h_num_noc']

# Guardamos un df con todos los valores del resumen estadistico, lo usaremos mas tarde
resumen_total = df.groupby('cluster')[numericas].agg(['mean', 'std'])

# Diccionario en el que guardarán los resultados por columna por cluster 
# (el arreglo 0 será media y 1 será desv. est.)
resultados = {}
for valor in numericas:
    medias = df.groupby('cluster')[valor].agg(['mean']).to_numpy().flatten()
    medias = np.round(medias, 2)
    desvests = df.groupby('cluster')[valor].agg(['std']).to_numpy().flatten()
    desvests = np.round(desvests, 2)
    resultados[valor] = []
    resultados[valor].append(medias)
    resultados[valor].append(desvests)

# Funcion para extraer el valor valor maximo, cluster al que corresponde, 
# y diferencia porcentual con el segundo maximo valor (aplica a las medias)
def resumen_maximo(arr):
    max_val = np.max(arr)
    max_idx = np.argmax(arr)
    arr_sin_max = np.delete(arr, max_idx)
    segundo_max = np.max(arr_sin_max)
    diferencia_pct = int(100 * (max_val - segundo_max) / max_val)
    return max_val, max_idx, diferencia_pct

# Lo mismo pero con el minimo (aplica a las desvests)
# Se consideran casos en que la desv. est. sea 0
def resumen_minimo(arr):
    min_val = np.nanmin(arr)
    min_idx = np.nanargmin(arr)
    arr_sin_min = np.delete(arr, min_idx)
    arr_sin_min = arr_sin_min[~np.isnan(arr_sin_min)]
    if arr_sin_min.size == 0:
        segundo_min = 0
        diferencia_pct = 0
    else:
        segundo_min = np.nanmin(arr_sin_min)
        diferencia_pct = int(100 * min_val / segundo_min) if segundo_min != 0 else None
    return min_val, min_idx, diferencia_pct

# Funcion que regresa un dataframe con el porcentaje de presencia de cada categoria agrupando 
# por otra (para aplicarse a mes de reservacion y de estadia agrupando por cluster)
def porcentaje_por_categoria(df, columna_grupo, columna_valor):
    conteos = df.groupby(columna_grupo)[columna_valor].value_counts(normalize=True)
    porcentaje_df = conteos.rename("porcentaje").reset_index()
    porcentaje_df["porcentaje"] *= 100
    porcentaje_df["porcentaje"] = np.round(porcentaje_df["porcentaje"], 2)
    return porcentaje_df

# Agrupamos el resumen en diccionarios, cada llave es el nombre de la columna y cada valor es el resumen guardado en una tupla:
# el primer valor de esta es mejor valor, el segundo el cluster al que corresponde y el tercerso su comparativa porcentual 
# con el segundo mejor (en el caso de la media, es que tanto % es mayor el primero que el segundo, y en el caso de la desviacion 
# estandar, es el % del segundo que constituye el primero; ej.: si el mejor es 1 y el segundo mejor 2, la comparativa es 50%)
resumen_medias = {}
resumen_desvests = {}
for i in range(len(numericas)):
    resumen_medias[numericas[i]] = resumen_maximo(resultados[numericas[i]][0])
    resumen_desvests[numericas[i]] = resumen_minimo(resultados[numericas[i]][1])

# Guardamos los dataframes que contienen ordenados los porcentajes de cada mes
df_top_meses_res = porcentaje_por_categoria(df, 'cluster', 'h_res_fec_mes')
df_top_meses_est = porcentaje_por_categoria(df, 'cluster', 'h_fec_lld_mes')

# Creamos diccionarios para guardar los dataframes con los top 3 meses mas prevalentes en fecha de reservacion y de 
# estadia para cada cluster y sus respectivos porcentajes
top_meses_res = {}
top_meses_est = {}
for i in range(0,4):
    top_meses_res[i] = df_top_meses_res[df_top_meses_res['cluster'] == i].iloc[:3][['h_res_fec_mes','porcentaje']]
    top_meses_est[i] = df_top_meses_est[df_top_meses_est['cluster'] == i].iloc[:3][['h_fec_lld_mes','porcentaje']]

##################################

# LOGICA PARA RECOMENDACIONES 
# PERSONALIZADAS

##################################

# Diccionario donde se guardaran los aspectos destacados por cluster
destacados = {0: [], 1: [], 2: [], 3: []}
recomendaciones = {0: [], 1: [], 2: [], 3: []}

for i in destacados.keys():
    for columna in resumen_medias.keys():
        # Cambiamos nombres para un output mas legible
        if columna == 'h_tot_hab': 
            nombre = 'no. de habitaciones'
        elif columna == 'tarifa_x_noche':
            nombre = 'tarifa por noche'
        elif columna == 'h_num_per':
            nombre = 'no. de personas'
        elif columna == 'h_num_adu':
            nombre = 'no. de adultos'
        elif columna == 'h_num_men':
            nombre = 'no. de menores'
        elif columna == 'h_num_noc':
            nombre = 'no. de noches'
        
        # Se agregaran comentarios a los clusters que tienen mayor promedio en cada una de las columnas categoricas. 
        # El criterio es: el promedio del cluster es al menos 15% mayor que cualquier otro
        if i == resumen_medias[columna][1] and resumen_medias[columna][2] > 15:
            destacados[i].append(f'Cluster con mayor promedio de {nombre}: {resumen_medias[columna][0]}, un {resumen_medias[columna][2]}% mayor a cualquer otro')
            if nombre == 'tarifa por noche':
                recomendaciones[i].append('Se recomienda presentar una tarifa alta')
            elif nombre == 'no. de menores':
                recomendaciones[i].append('Se recomienda llamar la atención y ofrecer servicios y actividades familiares o para niños')
            elif nombre == 'no. de habitaciones':
                recomendaciones[i].append('Se recomienda ofrecer mayor cantidad de habitaciones')
            
        # Se agregaran comentarios a los clusters que tienen la menor 
        # desviacion estandar en cada una de las columnas categoricas. 
        # El criterio es: la menor desv. est. es 33% o menos de la segunda menor
        # Pero primero se revisa si la desv. est. es 0, 
        # ya que en ese caso todos los valores son iguales
        if i == resumen_desvests[columna][1] and resumen_desvests[columna][0] == 0:
            destacados[i].append(f'Todos los valores de {nombre} son {resumen_desvests[columna][1]}')
            if nombre == 'tarifa por noche':
                recomendaciones[i].append(f'Proponer una tarifa de {resumen_desvests[columna][1]} sería extremadamente acertado')
            elif nombre == 'no. de habitaciones':
                recomendaciones[i].append(f'Proponer {resumen_desvests[columna][1]} sería extremadamente acertado')
        elif i == resumen_desvests[columna][1] and resumen_desvests[columna][1] < 33:
            media = np.round(resumen_total[columna]['mean'][i], 2)
            desvest = np.round(resumen_total[columna]['std'][i], 2)
            lim_inf = media - desvest
            lim_sup = media + desvest
            destacados[i].append(f'Valores muy cercanamente agrupados en {nombre}. Varían en +/- {resumen_desvests[columna][0]} unidades alrededor de su media {media}')
            if nombre == 'tarifa por noche':
                recomendaciones[i].append(f'Proponer una tarifa de entre {lim_inf:.2f} y {lim_sup:.2f} sería muy acertado')
            elif nombre == 'no. de noches':
                recomendaciones[i].append(f'Proponer entre {lim_inf:.2f} y {lim_sup:.2f} noches sería muy acertado')

    # Se agregaran comentarios a los clusters acerca de la prevalencia de los meses en reservaciones
    # Existen varios niveles de prevalencia:
    # Prevalencia definitiva: el mes mas prevalente tiene el doble de presencia que el segundo
    # Fuerte prevalencia: el mes mas prevalente tiene 1.5 la presencia del segundo
    # Ligera prevalencia: el mes mas prevalente tiene 1.24 la presencia del segundo 
    mes_res = top_meses_res[i].iloc[0][['h_res_fec_mes']].item()
    porcentaje_res = top_meses_res[i].iloc[0][['porcentaje']].item()
    relacion_primero_segundo_res = top_meses_res[i].iloc[0][['porcentaje']].item() / top_meses_res[i].iloc[1][['porcentaje']].item()
    if relacion_primero_segundo_res > 2:  
        destacados[i].append(f'El mes en que más se reservó fue {mes_res}, con una prevalencia definitiva ({porcentaje_res}%)')
    elif relacion_primero_segundo_res > 1.5:
        destacados[i].append(f'El mes en que más se reservó fue {mes_res}, con una fuerte prevalencia ({porcentaje_res}%)')
    elif relacion_primero_segundo_res > 1.25:
        destacados[i].append(f'El mes en que más se reservó fue {mes_res}, con una ligera prevalencia ({porcentaje_res}%)')

    # Se realiza el mismo proceso para la prevalencia de los meses de estadia
    mes_est = top_meses_est[i].iloc[0][['h_fec_lld_mes']].item()
    porcentaje_est = top_meses_est[i].iloc[0][['porcentaje']].item()
    relacion_primero_segundo_est = top_meses_est[i].iloc[0][['porcentaje']].item() / top_meses_est[i].iloc[1][['porcentaje']].item()
    if relacion_primero_segundo_est > 2:  
        destacados[i].append(f'El mes en que más estadías hubo fue {mes_est}, con una prevalencia definitiva ({porcentaje_est}%)')
    elif relacion_primero_segundo_est > 1.5:
        destacados[i].append(f'El mes en que más estadías hubo fue {mes_est}, con una fuerte prevalencia ({porcentaje_est}%)')
    elif relacion_primero_segundo_est > 1.25:
        destacados[i].append(f'El mes en que más estadías hubo fue {mes_est}, con una ligera prevalencia ({porcentaje_est}%)')
