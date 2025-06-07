import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os

# ============================
# Configuración de la página
# ============================

st.set_page_config(page_title="Cluster Insights Dashboard", layout="wide")

st.markdown(
    """
    <style>
        .main {background-color: #e4dadf;}
        h1, h2, h3 {color: #0c1013;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================
#  Login setup
# ======================

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Login
if not st.session_state.logged_in:
    placeholder = st.empty()

    with placeholder.form("login"):
        # Configuración del cover logo
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(current_dir, "images/tca_cover.jpg")
        st.image(image_path, use_container_width=True,  width=200)
        st.markdown("<h1 style='text-align: center;'>¡Bienvenido!</h1>", unsafe_allow_html=True)
        st.markdown("### Iniciar sesión")
        username = st.text_input("Usuario", placeholder="Ingresa tu usuario")
        password = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
        submit = st.form_submit_button("Iniciar sesión")

    if submit:
        if (username == "admin" and password == "admin") or (username == "user" and password == "user"):
            st.session_state.logged_in = True
            st.session_state.username = username
            placeholder.empty()
            st.success(f"Inicio de sesión exitoso para {username.upper()}")
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos")

if (st.session_state.logged_in == True):
    
    # ======================
    # Logo y título
    # ======================
    
    current_dir = os.path.dirname(__file__)
    image_path = os.path.join(current_dir, "images/tca_cover.jpg")
    st.image(image_path, width=1500) 

    st.markdown(
        """
        <h1 style='text-align: center; color: #bb0a01; font-family: 'Montserrat', sans-serif; font-weight: bold;'>
        Segmentación inteligente por reservaciones hoteleras
        </h1>
        """,
    unsafe_allow_html=True) 

    # ============================
    # Cargar datasets y modelos(?)
    # ============================

    #model = joblib.load("model.pkl")

    # Carga de datos
    @st.cache_data
    def load_data():
        #current_dir = os.path.dirname(__file__)
        #return pd.read_csv(os.path.join(current_dir, "df_gmm.csv"))
        try: 
            query = 'SELECT * "RAW"."GIT"."DF_GMM"'
            df = session.sql(query).topandas()
            return df
        except Exception as e: 
            st.error(f"Error: {e}")

    df = load_data()

    #Calcular el mes de la fecha de reserva
    #locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
    #df['h_res_fec_mes'] = pd.to_datetime(df['h_res_fec']).dt.strftime('%B').str.capitalize()

    #Calcular el mes de la fecha de estadía
    #locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
    #df['h_fec_lld_mes'] = pd.to_datetime(df['h_fec_lld']).dt.strftime('%B').str.capitalize()

    # Renombrando columnas pertinentes:
    df.rename(columns={'cluster_gmm': 'cluster'}, inplace=True)
    df.rename(columns={'mes_reservacion': 'h_res_fec_mes'}, inplace=True)
    df.rename(columns={'mes_entrada': 'h_fec_lld_mes'}, inplace=True)
    df.rename(columns={'tfa_xnoche': 'tarifa_x_noche'}, inplace=True)

    # Diccionario donde se guardaran los aspectos destacados por cluster
    destacados = {}
    for i in df['cluster'].unique():
        destacados[i] = []

    trad_meses = {
    'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
    'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
    'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
    'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'}

    cols_meses_a_traducir = ['h_res_fec_mes', 'h_fec_lld_mes', 'mes_salida']
    for col_mes in cols_meses_a_traducir:
        df[col_mes] = df[col_mes].map(lambda x: trad_meses.get(str(x).strip(), x))
    
    trad_dias = {
    'Monday': 'Lunes',
    'Tuesday': 'Martes',
    'Wednesday': 'Miércoles',
    'Thursday': 'Jueves',
    'Friday': 'Viernes',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
    }

    cols_dias_a_traducir = ['dia_reservacion', 'dia_entrada', 'dia_salida']
    for col_dia in cols_dias_a_traducir:
            df[col_dia] = df[col_dia].map(trad_dias)

    # ==========================
    # Separación por clústers
    # ==========================

    cluster0 = df[df['cluster'] == 0]
    cluster1 = df[df['cluster'] == 1]
    cluster2 = df[df['cluster'] == 2]
    cluster3 = df[df['cluster'] == 3]
    
    # Imágenes de clustyyrs 
    image_path0 = os.path.join(current_dir, "images/c0.jpg")
    image_path1 = os.path.join(current_dir, "images/c1.jpg")
    image_path2 = os.path.join(current_dir, "images/c2.jpg")
    image_path3 = os.path.join(current_dir, "images/c3.jpg")

    # Info de clusters
    cluster_info = [
        {"id": 0, "nombre": "Mini Escapada", "descripcion": "Viajeros que se escapan unos días —ya sea en solitario, con su pareja o con un par de amigos. Grupos chiquitos y, usualmente, de estancias cortas.", "imagen": image_path0, "color_bg": "#B8CABA", "color_text": "#0C1013"},
        {"id": 1, "nombre": "Exploradores de Estancias Largas", "descripcion": "Viajeros que se toman su tiempo para disfrutar. A veces viajan solos, a veces en grupo, pero suelen quedarse más noches.", "imagen": image_path1, "color_bg": "#91A294", "color_text": "#E1E1E1"},
        {"id": 2, "nombre": "La Dupla Perfecta", "descripcion": "Siempre de dos en dos. Parejas que viajan juntas, quizás recurrentes, con hábitos muy definidos.", "imagen": image_path2, "color_bg": "#FF616E", "color_text": "#E1E1E1"},
        {"id": 3, "nombre": "Plan Familiar y Amigos", "descripcion": "Grupos grandes, a veces muy grandes. Incluyen niños y buscan experiencias compartidas. Estancias vacacionales.", "imagen": image_path3, "color_bg": "#F34C3F", "color_text": "#0C1013"}
    ]

    ##################################
    # EXTRACCION DE INDICADORES 
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
    # Guardamos los dataframes que contienen ordenados los porcentajes de cada dia de la semana
    
    df_top_dias_res = porcentaje_por_categoria(df,'cluster', 'dia_reservacion')
    df_top_dias_ent = porcentaje_por_categoria(df,'cluster', 'dia_entrada')
    df_top_dias_sal = porcentaje_por_categoria(df,'cluster', 'dia_salida')

    # Creamos diccionarios para guardar los dataframes con los top 3 meses mas prevalentes en fecha de reservacion y de 
    # estadia para cada cluster y sus respectivos porcentajes
    top_meses_res = {}
    top_meses_est = {}

    top_dias_res = {}
    top_dias_ent = {}
    top_dias_sal = {}

    for i in range(0,4):
        top_meses_res[i] = df_top_meses_res[df_top_meses_res['cluster'] == i].iloc[:3][['h_res_fec_mes','porcentaje']]
        top_meses_est[i] = df_top_meses_est[df_top_meses_est['cluster'] == i].iloc[:3][['h_fec_lld_mes','porcentaje']]
        
        top_dias_res[i] = df_top_dias_res[df_top_dias_res['cluster'] == i].iloc[:3][['dia_reservacion','porcentaje']]
        top_dias_ent[i] = df_top_dias_ent[df_top_dias_ent['cluster'] == i].iloc[:3][['dia_entrada','porcentaje']]
        top_dias_sal[i] = df_top_dias_sal[df_top_dias_sal['cluster'] == i].iloc[:3][['dia_salida','porcentaje']]


    # ================================
    # Multipágina: Selección de página
    # ================================
    st.sidebar.title("Navegación")
    page = st.sidebar.selectbox("Selecciona una página", ['Global', 'Por cluster'])
    
    # ================================
    # Comparativas entre clústers
    # ================================

    if page == "Global":
        
        st.markdown("""<h2 style='font-size:16px'; color: #BC544B; font-family: "Roboto", sans-serif;'> Comparativas entre clústers </h2>""", unsafe_allow_html=True)
        
        # Media de la tarifa por noche para cada clúster
        st.markdown("""<h2 style='font-size:16px'; color: #BC544B; font-family: "Roboto", sans-serif;'> Media de tarifa por noche </h2""", unsafe_allow_html=True)

        cols = st.columns(4)
        for idx, cl in enumerate(cluster_info):
            with cols[idx]:
                st.metric(
                f"Cluster {cl['nombre']}",
                f"${resultados['tarifa_x_noche'][0][idx]:.2f}"
            )
                st.markdown(
                f"<span style='color:lightskyblue;'>±{resultados['tarifa_x_noche'][1][idx]:.2f}</span>",
                unsafe_allow_html=True
            )

        # Grafico de barras de la media de la tarifa por noche para cada clúster
        # DataFrame para el gráfico
        df_tarifa = pd.DataFrame({
            "Cluster": [0, 1, 2, 3],
            "Media_Tarifa_x_Noche": resultados['tarifa_x_noche'][0],
            "Desviacion_Tarifa_x_Noche": resultados['tarifa_x_noche'][1]
        })

        mean_tarifa_bars = px.bar(
            df_tarifa,
            x="Cluster",
            y="Media_Tarifa_x_Noche",
            error_y="Desviacion_Tarifa_x_Noche",
            labels={"Media_Tarifa_x_Noche": "Media de Tarifa por Noche", "Cluster": "Cluster"},
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        mean_tarifa_bars.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        mean_tarifa_bars.update_layout(yaxis=dict(rangemode='tozero'))
        mean_tarifa_bars.update_layout(title="Comparación por cluster", yaxis_title="Tarifa promedio por noche")
        st.plotly_chart(mean_tarifa_bars, use_container_width=True)


        st.markdown("""<h2 style='font-size:16px'; color: #BC544B; font-family: "Roboto", sans-serif;'> Media de número de noches </h2>""", unsafe_allow_html=True)


        cols = st.columns(4)
        for idx, cl in enumerate(cluster_info):
            with cols[idx]:
                st.metric(
                f"Cluster {cl['nombre']}",
                f"{resultados['h_num_noc'][0][idx]:.0f}"
            )
                st.markdown(
                f"<span style='color:lightskyblue;'>±{resultados['h_num_noc'][1][idx]:.2f}</span>",
                unsafe_allow_html=True
            )

        # Crear un DataFrame para la gráfica de barras de h_num_noc (media de noches por cluster)
        df_noches = pd.DataFrame({
            "Cluster": [0, 1, 2, 3],
            "Media_Noches": resultados['h_num_noc'][0],
            "Desviacion_Noches": resultados['h_num_noc'][1]
        })

        bars_mean_noches = px.bar(
            df_noches,
            x="Cluster",
            y="Media_Noches",
            error_y="Desviacion_Noches",
            color="Cluster",
            text="Media_Noches",
            color_discrete_sequence=px.colors.sequential.Reds
        )

        bars_mean_noches.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        bars_mean_noches.update_layout(title="Comparación por cluster", yaxis_title="Noches promedio")

        st.plotly_chart(bars_mean_noches, use_container_width=True)
    
    # ================================
    # Presentación de clústers
    # ================================
    
    if page == "Por cluster":
    
            # Inicializar el estado
        if "cluster_seleccionado" not in st.session_state:
            st.session_state.cluster_seleccionado = None

        st.markdown(
            """
            <h4 style='text-align: left; font-family: 'Montserrat', sans-serif; font-weight: bold;'>
            Nuestros segmentos
            </h4>
            """,
        unsafe_allow_html=True)

        # Mostrar cada clúster en una columna para visualización paralela
        cols = st.columns(len(cluster_info))
        for idx, cl in enumerate(cluster_info):
            with cols[idx]:
                st.image(cl['imagen'], use_container_width=False, width=300, output_format="JPEG", clamp=True)
                st.markdown(
                    f"""
                    <div style="
                        background-color: {cl['color_bg']};
                        border-radius: 10px;
                        padding: 20px;
                        margin-bottom: 10px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    ">
                        <div style="text-align: center;">
                            <h4 style=" color={cl['color_text']}; margin-bottom: 5px;">{cl['nombre']}</h4>
                            <p style="color={cl['color_text']}; margin-top: 0;">{cl['descripcion']}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(f"Ver más del Clúster {cl['nombre']}", key=f"btn_{cl['id']}"):
                    st.session_state.cluster_seleccionado = cl["id"]


   
            
        ##################################
        # LOGICA PARA RECOMENDACIONES 
        # PERSONALIZADAS
        ##################################


        
        # Cambiamos nombres para un output mas legible
        nom_cols = {
        'h_tot_hab': 'no. de habitaciones',
        'tarifa_x_noche': 'tarifa por noche',
        'h_num_per': 'no. de personas',
        'h_num_adu': 'no. de adultos',
        'h_num_men': 'no. de menores',
        'h_num_noc': 'no. de noches'
        }

        for i in destacados.keys():
            for columna in resumen_medias.keys():

                
                # Se agregaran comentarios a los clusters que tienen mayor promedio en cada una de las columnas categoricas. 
                # El criterio es: el promedio del cluster es al menos 15% mayor que cualquier otro
                if i == resumen_medias[columna][1] and resumen_medias[columna][2] > 15:
                    if nom_cols[columna] == 'tarifa por noche':
                        ad = 'Se recomienda presentar una tarifa alta'
                    elif nom_cols[columna] == 'no. de menores':
                        ad = 'Se recomienda llamar la atención y ofrecer servicios y actividades familiares o para niños'
                    elif nom_cols[columna] == 'no. de habitaciones':
                        ad = 'Se recomienda ofrecer mayor cantidad de habitaciones'
                    else:
                        ad = ''
                    destacados[i].append(f'- Cluster con mayor promedio de {nom_cols[columna]}: <span style="color:#d62728; font-weight:bold; font-size:21p;"> {resumen_medias[columna][0]} ({resumen_medias[columna][2]}% mayor a cualquier otro) </span> \n{ad}')

                    
                # Se agregaran comentarios a los clusters que tienen la menor 
                # desviacion estandar en cada una de las columnas categoricas. 
                # El criterio es: la menor desv. est. es 33% o menos de la segunda menor
                # Pero primero se revisa si la desv. est. es 0, 
                # ya que en ese caso todos los valores son iguales
                if i == resumen_desvests[columna][1] and resumen_desvests[columna][0] == 0:
                    mean = resumen_total[columna]['mean'][i]
                    destacados[i].append(f'- Todos los valores de {nom_cols[columna]} son <span style="color:#d62728; font-weight:bold; font-size:21p;"> {mean:.0f}')
                    if nom_cols[columna] == 'tarifa por noche':
                        destacados[i].append(f'- Proponer una tarifa de <span style="color:#d62728; font-weight:bold; font-size:21p;"> {resumen_medias[columna][0]} </span> sería extremadamente acertado')
                    elif nom_cols[columna] == 'no. de habitaciones':
                        mean = resumen_total[columna]['mean'][i]
                        destacados[i].append(f'- Proponer <span style="color:#d62728; font-weight:bold; font-size:21p;"> {mean} </span> habitaciones sería extremadamente acertado')
                elif i == resumen_desvests[columna][1] and resumen_desvests[columna][1] < 33:
                    media = np.round(resumen_total[columna]['mean'][i], 2)
                    desvest = np.round(resumen_total[columna]['std'][i], 2)
                    lim_inf = media - desvest
                    lim_sup = media + desvest
                    destacados[i].append(f'- Valores de {nom_cols[columna]} muy cercanamente agrupados entre sí. Varían en <span style="color:#d62728; font-weight:bold; font-size:21p;"> +/- {resumen_desvests[columna][0]} unidades alrededor de su media {media}')
                    if nom_cols[columna] == 'tarifa por noche':
                        destacados[i].append(f'- Proponer una tarifa de entre <span style="color:#d62728; font-weight:bold; font-size:21p;"> {lim_inf:.2f} y {lim_sup:.2f} </span> sería muy acertado')
                    elif nom_cols[columna] == 'no. de noches':
                        destacados[i].append(f'- Proponer entre {lim_inf:.0f} y {lim_sup:.0f} noches sería muy acertado')

            # Se agregaran comentarios a los clusters acerca de la prevalencia de los meses en reservaciones
            # Existen varios niveles de prevalencia:
            # Prevalencia definitiva: el mes mas prevalente tiene el doble de presencia que el segundo
            # Fuerte prevalencia: el mes mas prevalente tiene 1.5 la presencia del segundo
            # Ligera prevalencia: el mes mas prevalente tiene 1.24 la presencia del segundo 
            mes_res = top_meses_res[i].iloc[0][['h_res_fec_mes']].item()
            porcentaje_res = top_meses_res[i].iloc[0][['porcentaje']].item()
            relacion_primero_segundo_res = top_meses_res[i].iloc[0][['porcentaje']].item() / top_meses_res[i].iloc[1][['porcentaje']].item()
            if relacion_primero_segundo_res > 1.1:
                destacados[i].append('#### Meses de Reservación')
                if relacion_primero_segundo_res > 1.5:  
                    destacados[i].append(f'- Mes predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;"> {mes_res}</span>. Nivel de mayoría: Alto <span style="color:#d62728; font-weight:bold; font-size:21p;"> ({porcentaje_res:.2f}%)')
                elif relacion_primero_segundo_res > 1.25:
                    destacados[i].append(f'- Mes predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;"> {mes_res}</span>. Nivel de mayoría: Medio <span style="color:#d62728; font-weight:bold; font-size:21p;"> ({porcentaje_res:.2f}%)')
                elif relacion_primero_segundo_res > 1.1:
                    destacados[i].append(f'- Mes predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;"> {mes_res}</span>. Nivel de mayoría: Bajo <span style="color:#d62728; font-weight:bold; font-size:21p;"> ({porcentaje_res:.2f}%)')

            # Se realiza el mismo proceso para la prevalencia de los meses de estadia
            mes_est = top_meses_est[i].iloc[0][['h_fec_lld_mes']].item()
            porcentaje_est = top_meses_est[i].iloc[0][['porcentaje']].item()
            relacion_primero_segundo_est = top_meses_est[i].iloc[0][['porcentaje']].item() / top_meses_est[i].iloc[1][['porcentaje']].item()
            if relacion_primero_segundo_est > 1.1:
                destacados[i].append('#### Meses de Estadía')
                if relacion_primero_segundo_est > 1.5:  
                    destacados[i].append(f'- Mes predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;"> {mes_est}</span>. Nivel de mayoría: Alto <span style="color:#d62728; font-weight:bold; font-size:21p;"> ({porcentaje_est:.2f}%)')
                elif relacion_primero_segundo_est > 1.25:
                    destacados[i].append(f'- Mes predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;"> {mes_est}</span>. Nivel de mayoría: Medio <span style="color:#d62728; font-weight:bold; font-size:21p;"> ({porcentaje_est:.2f}%)')
                elif relacion_primero_segundo_est > 1.1:
                    destacados[i].append(f'- Mes predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;"> {mes_est}</span>. Nivel de mayoría: Bajo <span style="color:#d62728; font-weight:bold; font-size:21p;"> ({porcentaje_est:.2f}%)')
                
            # Se realiza el mismo proceso para la prevalencia de los dias de la semana en reservaciones
            dia_res = top_dias_res[i].iloc[0][['dia_reservacion']].item()
            porcentaje_est = top_dias_res[i].iloc[0][['porcentaje']].item()
            relacion_primero_segundo_est = top_dias_res[i].iloc[0][['porcentaje']].item() / top_dias_res[i].iloc[1][['porcentaje']].item()
            if relacion_primero_segundo_est > 1.1:
                destacados[i].append('#### Días de Reservación')
                if relacion_primero_segundo_est > 1.25:  
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;">{dia_res}</span>. Nivel de mayoría: Alto <span style="color:#d62728; font-weight:bold; font-size:21p;">({porcentaje_est:.2f}%)')
                elif relacion_primero_segundo_est > 1.15:
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;">{dia_res}</span>. Nivel de mayoría: Medio <span style="color:#d62728; font-weight:bold; font-size:21p;">({porcentaje_est:.2f}%)')
                elif relacion_primero_segundo_est > 1.1:
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;">{dia_res}</span>. Nivel de mayoría: Bajo <span style="color:#d62728; font-weight:bold; font-size:21p;">({porcentaje_est:.2f}%)')
                
            # Se realiza el mismo proceso para la prevalencia de los dias de la semana en llegadas
            dias_ent = top_dias_ent[i].iloc[0][['dia_entrada']].item()
            porcentaje_est = top_dias_ent[i].iloc[0][['porcentaje']].item()
            relacion_primero_segundo_est = top_dias_ent[i].iloc[0][['porcentaje']].item() / top_dias_ent[i].iloc[1][['porcentaje']].item()
            if relacion_primero_segundo_est > 1.1:
                destacados[i].append('#### Días de Llegada')
                if relacion_primero_segundo_est > 1.25:  
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;">{dias_ent}</span>. Nivel de mayoría: Alto <span style="color:#d62728; font-weight:bold; font-size:21p;">({porcentaje_est:.2f}%)')
                elif relacion_primero_segundo_est > 1.15:
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;">{dias_ent}</span>. Nivel de mayoría: Medio <span style="color:#d62728; font-weight:bold; font-size:21p;">({porcentaje_est:.2f}%)')
                elif relacion_primero_segundo_est > 1.1:
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;">{dias_ent}</span>. Nivel de mayoría: Bajo <span style="color:#d62728; font-weight:bold; font-size:21p;">({porcentaje_est:.2f}%)')
                
            # Se realiza el mismo proceso para la prevalencia de los dias de la semana en salidas
            dias_sal = top_dias_sal[i].iloc[0][['dia_salida']].item()
            porcentaje_est = top_dias_sal[i].iloc[0][['porcentaje']].item()
            relacion_primero_segundo_est = top_dias_sal[i].iloc[0][['porcentaje']].item() / top_dias_sal[i].iloc[1][['porcentaje']].item()
            if relacion_primero_segundo_est > 1.1:
                destacados[i].append('#### Días de Salida')
                if relacion_primero_segundo_est > 1.25:  
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;">{dias_sal}</span>. Nivel de mayoría: Alto  <span style="color:#d62728; font-weight:bold; font-size:21px;">({porcentaje_est:.2f}%)')
                elif relacion_primero_segundo_est > 1.15:
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21p;">{dias_sal}</span>. Nivel de mayoría: Medio  <span style="color:#d62728; font-weight:bold; font-size:21px;">({porcentaje_est:.2f}%)')
                elif relacion_primero_segundo_est > 1.1:
                    destacados[i].append(f'- Día predominante: <span style="color:#d62728; font-weight:bold; font-size:21px;">{dias_sal}</span>. Nivel de mayoría: Bajo <span style="color:#d62728; font-weight:bold; font-size:21px;">({porcentaje_est:.2f}%)')
            
        # Detalles expandidos
        if st.session_state.cluster_seleccionado is not None:
            cl = next(c for c in cluster_info if c['id'] == st.session_state.cluster_seleccionado)
            st.markdown("---")
            st.subheader(f"🔍 Detalles del Clúster {cl['id']}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📌 Destacados")
                for i in destacados[cl['id']]:
                    st.write(i, unsafe_allow_html=True)

            with col2:
                # Define custom color palette
                custom_colors = ['#b8caba', '#91A294', '#f34c3f', "#FF616E", '#cccccc']

                # Gráfica de pastel para Top 3 Meses de Estadía
                if '#### Meses de Estadía' in destacados[cl['id']]:
                    total = top_meses_est[cl['id']]['porcentaje'].sum()
                    top_meses_est[cl['id']] = pd.concat([
                        top_meses_est[cl['id']],
                        pd.DataFrame({'h_fec_lld_mes': ['Otro'], 'porcentaje': [100 - total]})
                    ], ignore_index=True)
                    fig = px.pie(
                        top_meses_est[cl['id']],
                        values='porcentaje',
                        names='h_fec_lld_mes',
                        title='Top 3 Meses de Estadía',
                        hole=0.3,
                        color_discrete_sequence=custom_colors
                    )
                    fig.update_traces(textinfo='percent+label')
                    fig.update_layout(
                        showlegend=True,
                        title_font_size=14,
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=350,
                        width=350
                    )
                    st.plotly_chart(fig, use_container_width=False)

                # Gráfica de pastel para Top 3 Meses de Reservación
                if '#### Meses de Reservación' in destacados[cl['id']]:
                    total = top_meses_res[cl['id']]['porcentaje'].sum()
                    top_meses_res[cl['id']] = pd.concat([
                        top_meses_res[cl['id']],
                        pd.DataFrame({'h_res_fec_mes': ['Otro'], 'porcentaje': [100 - total]})
                    ], ignore_index=True)
                    fig = px.pie(
                        top_meses_res[cl['id']],
                        values='porcentaje',
                        names='h_res_fec_mes',
                        title='Top 3 Meses de Reservación',
                        hole=0.3,
                        color_discrete_sequence=custom_colors
                    )
                    fig.update_traces(textinfo='percent+label')
                    fig.update_layout(
                        showlegend=True,
                        title_font_size=14,
                        margin=dict(l=10, r=10, t=30, b=10),
                        height=350,
                        width=350
                    )
                    st.plotly_chart(fig, use_container_width=False)
