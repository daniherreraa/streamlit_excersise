import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns  # Aunque importado, no se usa en las funciones de graficación del código provisto
import matplotlib.pyplot as plt  # Aunque importado, no se usa en las funciones de graficación del código provisto
from datetime import datetime, timedelta
import random
from io import StringIO  # Aunque importado, no se usa activamente en el código provisto

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Oncológico China",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS PERSONALIZADO ELIMINADO PARA USAR EL TEMA POR DEFECTO DE STREAMLIT

@st.cache_data
def generate_medical_dataset(n_samples=1000):
    """Genera un dataset médico realista basado en el ejemplo proporcionado"""
    np.random.seed(42)
    random.seed(42)

    # Listas de valores posibles
    provinces = ['Hunan', 'Sichuan', 'Guangdong', 'Anhui', 'Beijing', 'Shanghai',
                 'Jiangsu', 'Zhejiang', 'Shandong', 'Henan', 'Hubei', 'Guangxi']
    ethnicities = ['Han', 'Uyghur', 'Mongolian', 'Tibetan', 'Zhuang']
    tumor_types = ['Lung', 'Breast', 'Stomach', 'Liver', 'Colorectal', 'Cervical']
    cancer_stages = ['I', 'II', 'III', 'IV']
    treatment_types = ['Surgery', 'Chemotherapy', 'Radiation', 'Combined']
    smoking_status = ['Never', 'Former', 'Current']
    alcohol_use = ['None', 'Occasional', 'Regular', 'Heavy']
    genetic_mutations = ['None', 'EGFR', 'KRAS', 'TP53', 'BRCA1', 'BRCA2']
    comorbidities_list = ['None', 'Diabetes', 'Hypertension', 'Hepatitis B',
                          'Heart Disease', 'Diabetes, Hypertension']

    data = []
    for i in range(n_samples):
        # Generar fechas realistas
        diagnosis_date = datetime(2010, 1, 1) + timedelta(
            days=random.randint(0, (datetime(2023, 12, 31) - datetime(2010, 1, 1)).days)
        )
        surgery_date = diagnosis_date + timedelta(days=random.randint(7, 120)) if random.random() > 0.3 else None

        # Lógica para supervivencia basada en etapa
        stage = random.choice(cancer_stages)
        survival_prob = {'I': 0.9, 'II': 0.75, 'III': 0.5, 'IV': 0.2}[stage]
        survival_status = 'Alive' if random.random() < survival_prob else 'Deceased'

        # Seguimiento más largo para pacientes vivos
        if survival_status == 'Alive':
            follow_up = random.randint(12, 60)
        else:
            follow_up = random.randint(1, 48)

        # Correlaciones realistas
        age = random.randint(25, 80)
        tumor_size = round(random.uniform(2.0, 15.0), 1)
        metastasis = 'Yes' if stage in ['III', 'IV'] and random.random() > 0.3 else 'No'

        row = {
            'PatientID': f'CHN-{i + 1:05d}',
            'Gender': random.choice(['Male', 'Female', 'Other']),
            'Age': age,
            'Province': random.choice(provinces),
            'Ethnicity': random.choice(ethnicities),
            'TumorType': random.choice(tumor_types),
            'CancerStage': stage,
            'DiagnosisDate': diagnosis_date.strftime('%Y-%m-%d'),
            'TumorSize': tumor_size,
            'Metastasis': metastasis,
            'TreatmentType': random.choice(treatment_types),
            'SurgeryDate': surgery_date.strftime('%Y-%m-%d') if surgery_date else 'N/A',
            'ChemotherapySessions': random.randint(0, 20) if random.random() > 0.4 else 0,
            'RadiationSessions': random.randint(0, 30) if random.random() > 0.5 else 0,
            'SurvivalStatus': survival_status,
            'FollowUpMonths': follow_up,
            'SmokingStatus': random.choice(smoking_status),
            'AlcoholUse': random.choice(alcohol_use),
            'GeneticMutation': random.choice(genetic_mutations) if random.random() > 0.7 else 'None',
            'Comorbidities': random.choice(comorbidities_list) if random.random() > 0.6 else 'None'
        }
        data.append(row)

    return pd.DataFrame(data)


def create_stacked_bar_chart(df):
    """Gráfico de barras apiladas por tipo y etapa del cáncer"""
    pivot_data = df.groupby(['TumorType', 'CancerStage']).size().reset_index(name='Count')

    fig = px.bar(
        pivot_data,
        x='TumorType',
        y='Count',
        color='CancerStage',  # Plotly usará colores por defecto
        title='📊 Distribución de Pacientes por Tipo y Etapa del Cáncer'
        # color_discrete_sequence y template eliminados
    )

    fig.update_layout(
        height=500,
        xaxis_title='Tipo de Cáncer',
        yaxis_title='Número de Pacientes',
        font=dict(size=12),  # Se mantiene la personalización de fuente si estaba
        title_font_size=16,
        legend_title='Etapa del Cáncer'
    )

    return fig


def create_survival_pie_chart(df):
    """Pie chart de distribución de supervivencia"""
    survival_counts = df['SurvivalStatus'].value_counts()

    # colors eliminado

    fig = go.Figure(data=[go.Pie(
        labels=survival_counts.index,
        values=survival_counts.values,
        hole=0.4
        # marker_colors eliminado
    )])

    fig.update_layout(
        title='🎯 Estado de Supervivencia de Pacientes',
        height=400,
        font=dict(size=12),
        title_font_size=16
    )

    return fig


def create_age_histogram(df):
    """Histograma de distribución de edad"""
    fig = px.histogram(
        df,
        x='Age',
        nbins=20,
        title='📈 Distribución de Edad de Pacientes'
        # color_discrete_sequence y template eliminados
    )

    fig.update_layout(
        height=400,
        xaxis_title='Edad',
        yaxis_title='Frecuencia',
        font=dict(size=12),
        title_font_size=16
    )

    return fig


def create_age_survival_boxplot(df):
    """Boxplot de edad vs estado de supervivencia"""
    fig = px.box(
        df,
        x='SurvivalStatus',
        y='Age',
        title='📦 Distribución de Edad por Estado de Supervivencia',
        color='SurvivalStatus'  # Plotly usará colores por defecto
        # color_discrete_map y template eliminados
    )

    fig.update_layout(
        height=400,
        xaxis_title='Estado de Supervivencia',
        yaxis_title='Edad',
        font=dict(size=12),
        title_font_size=16,
        showlegend=False
    )

    return fig


def create_correlation_heatmap(df):
    """Heatmap de correlación entre variables cuantitativas"""
    numeric_cols = ['Age', 'TumorSize', 'FollowUpMonths', 'ChemotherapySessions', 'RadiationSessions']
    # Asegurar que solo columnas numéricas válidas y presentes en el df se usen
    valid_numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not valid_numeric_cols:
        # Devolver un gráfico vacío o con un mensaje si no hay datos
        fig = go.Figure()
        fig.update_layout(title='🔥 Mapa de Correlación (Datos insuficientes)', height=500)
        return fig

    corr_matrix = df[valid_numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title='🔥 Mapa de Correlación - Variables Cuantitativas'
        # color_continuous_scale y template eliminados
    )

    fig.update_layout(
        height=500,
        font=dict(size=12),
        title_font_size=16
    )

    return fig


def create_timeline_chart(df):
    """Línea de tiempo de seguimiento por paciente (muestra)"""
    sample_df = df.head(50).copy()  # Tomar una muestra para mejor visualización
    sample_df['DiagnosisDate'] = pd.to_datetime(sample_df['DiagnosisDate'], errors='coerce')

    # Calcular EndDate para la línea de tiempo si FollowUpMonths es numérico
    if pd.api.types.is_numeric_dtype(sample_df['FollowUpMonths']):
        # Opción 1: Usar pd.DateOffset (más preciso para meses calendario)
        # sample_df['EndDate'] = sample_df.apply(
        #     lambda row: row['DiagnosisDate'] + pd.DateOffset(months=int(row['FollowUpMonths']))
        #     if pd.notna(row['DiagnosisDate']) and pd.notna(row['FollowUpMonths']) else pd.NaT,
        #     axis=1
        # )

        # Opción 2: Aproximar un mes a 30.44 días (365.25 / 12) para un timedelta
        # O simplemente usar 30 días si la precisión no es crítica.
        days_per_month_approx = 30.4375
        sample_df['EndDate'] = sample_df.apply(
            lambda row: row['DiagnosisDate'] + pd.to_timedelta(row['FollowUpMonths'] * days_per_month_approx, unit='D')
            if pd.notna(row['DiagnosisDate']) and pd.notna(row['FollowUpMonths']) else pd.NaT,
            axis=1
        )
    else:
        fig = go.Figure()
        fig.update_layout(title='⏰ Línea de Tiempo (Datos de seguimiento no válidos)', height=500)
        return fig

    # Filtrar filas donde EndDate no pudo ser calculado o DiagnosisDate es NaT
    sample_df = sample_df.dropna(subset=['DiagnosisDate', 'EndDate'])

    if sample_df.empty:
        fig = go.Figure()
        fig.update_layout(title='⏰ Línea de Tiempo (No hay datos válidos para mostrar)', height=500)
        return fig

    fig = px.scatter(
        sample_df,
        x='DiagnosisDate',
        y='FollowUpMonths',
        color='SurvivalStatus',
        size='TumorSize',
        hover_data=['PatientID', 'TumorType', 'CancerStage'],
        title='⏰ Línea de Tiempo de Seguimiento (Muestra de 50 pacientes)'
    )

    fig.update_layout(
        height=500,
        xaxis_title='Fecha de Diagnóstico',
        yaxis_title='Meses de Seguimiento',
        font=dict(size=12),
        title_font_size=16
    )

    return fig


def create_treatment_bar_chart(df):
    """Gráfico de barras por tipo de tratamiento"""
    treatment_counts = df['TreatmentType'].value_counts()

    fig = px.bar(
        x=treatment_counts.index,
        y=treatment_counts.values,
        title='💊 Frecuencia de Tipos de Tratamiento',
        color=treatment_counts.values  # Mantenemos color por valor, Plotly usará escala secuencial por defecto
        # color_continuous_scale y template eliminados
    )

    fig.update_layout(
        height=400,
        xaxis_title='Tipo de Tratamiento',
        yaxis_title='Número de Pacientes',
        font=dict(size=12),
        title_font_size=16,
        showlegend=False  # Esto es útil si el color es por valor y no categórico
    )

    return fig


def apply_filters(df, filters):
    """Aplica filtros al dataframe"""
    filtered_df = df.copy()

    if filters['tumor_types']:
        filtered_df = filtered_df[filtered_df['TumorType'].isin(filters['tumor_types'])]

    if filters['age_range']:
        filtered_df = filtered_df[
            (filtered_df['Age'] >= filters['age_range'][0]) &
            (filtered_df['Age'] <= filters['age_range'][1])
            ]

    if filters['survival_status']:
        filtered_df = filtered_df[filtered_df['SurvivalStatus'].isin(filters['survival_status'])]

    if filters['follow_up_range']:
        filtered_df = filtered_df[
            (filtered_df['FollowUpMonths'] >= filters['follow_up_range'][0]) &
            (filtered_df['FollowUpMonths'] <= filters['follow_up_range'][1])
            ]

    if filters['gender']:
        filtered_df = filtered_df[filtered_df['Gender'].isin(filters['gender'])]

    if filters['cancer_stage']:
        filtered_df = filtered_df[filtered_df['CancerStage'].isin(filters['cancer_stage'])]

    if filters['genetic_mutation']:
        filtered_df = filtered_df[filtered_df['GeneticMutation'].isin(filters['genetic_mutation'])]

    return filtered_df


def main():
    # Título principal
    st.markdown("# 🏥 Dashboard Oncológico - Análisis de Pacientes en China")
    st.markdown("---")

    # Cargar datos
    with st.spinner('Cargando datos médicos...'):
        df = generate_medical_dataset(1500)

    # Sidebar con filtros
    st.sidebar.markdown("## 🔍 Filtros de Datos")
    st.sidebar.markdown("---")

    # Filtros interactivos
    # Usar df['ColumnName'].unique().tolist() para asegurar que las opciones son listas
    tumor_types_options = df['TumorType'].unique().tolist()
    tumor_types = st.sidebar.multiselect(
        "Tipo de Cáncer",
        options=tumor_types_options,
        default=tumor_types_options
    )

    age_range = st.sidebar.slider(
        "Rango de Edad",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

    survival_status_options = df['SurvivalStatus'].unique().tolist()
    survival_status = st.sidebar.multiselect(
        "Estado de Supervivencia",
        options=survival_status_options,
        default=survival_status_options
    )

    follow_up_range = st.sidebar.slider(
        "Meses de Seguimiento",
        min_value=int(df['FollowUpMonths'].min()),
        max_value=int(df['FollowUpMonths'].max()),
        value=(int(df['FollowUpMonths'].min()), int(df['FollowUpMonths'].max()))
    )

    gender_options = df['Gender'].unique().tolist()
    gender = st.sidebar.multiselect(
        "Género",
        options=gender_options,
        default=gender_options
    )

    cancer_stage_options = df['CancerStage'].unique().tolist()
    # Opcional: ordenar etapas de cáncer de forma lógica si es necesario
    # cancer_stage_options = sorted(cancer_stage_options, key=lambda x: ("I", "II", "III", "IV").index(x) if x in ["I", "II", "III", "IV"] else -1)
    cancer_stage = st.sidebar.multiselect(
        "Etapa del Cáncer",
        options=cancer_stage_options,
        default=cancer_stage_options
    )

    genetic_mutation_options = df['GeneticMutation'].unique().tolist()
    genetic_mutation = st.sidebar.multiselect(
        "Mutación Genética",
        options=genetic_mutation_options,
        default=genetic_mutation_options
    )

    # Aplicar filtros
    filters = {
        'tumor_types': tumor_types,
        'age_range': age_range,
        'survival_status': survival_status,
        'follow_up_range': follow_up_range,
        'gender': gender,
        'cancer_stage': cancer_stage,
        'genetic_mutation': genetic_mutation
    }

    filtered_df = apply_filters(df, filters)

    # Métricas principales
    st.markdown("## 📊 Métricas Principales")

    col1, col2, col3, col4 = st.columns(4)

    # Manejo de división por cero o DataFrames vacíos para métricas
    if not filtered_df.empty:
        with col1:
            st.metric(
                label="👥 Total Pacientes",
                value=len(filtered_df),
                delta=f"{len(filtered_df) - len(df)} del total" if len(df) > 0 else None
            )

        with col2:
            survival_rate = (filtered_df['SurvivalStatus'] == 'Alive').mean() * 100
            # Ejemplo de delta para survival_rate (requiere un valor base para comparar)
            # Aquí, 70 es un valor arbitrario para el ejemplo.
            base_survival_rate_general = 70
            st.metric(
                label="💚 Tasa Supervivencia",
                value=f"{survival_rate:.1f}%",
                delta=f"{survival_rate - base_survival_rate_general:.1f}% vs promedio" if base_survival_rate_general else None
            )

        with col3:
            avg_age = filtered_df['Age'].mean()
            avg_age_general = df['Age'].mean() if len(df) > 0 else avg_age
            st.metric(
                label="📅 Edad Promedio",
                value=f"{avg_age:.1f} años",
                delta=f"{avg_age - avg_age_general:.1f} vs general" if len(df) > 0 else None
            )

        with col4:
            avg_follow_up = filtered_df['FollowUpMonths'].mean()
            avg_follow_up_general = df['FollowUpMonths'].mean() if len(df) > 0 else avg_follow_up
            st.metric(
                label="⏱️ Seguimiento Promedio",
                value=f"{avg_follow_up:.1f} meses",
                delta=f"{avg_follow_up - avg_follow_up_general:.1f} vs general" if len(df) > 0 else None
            )
    else:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")
        # Mostrar métricas con N/A o 0 si el df filtrado está vacío
        with col1:
            st.metric(label="👥 Total Pacientes", value=0)
        with col2:
            st.metric(label="💚 Tasa Supervivencia", value="N/A")
        with col3:
            st.metric(label="📅 Edad Promedio", value="N/A")
        with col4:
            st.metric(label="⏱️ Seguimiento Promedio", value="N/A")

    st.markdown("---")

    # Visualizaciones principales (solo si hay datos)
    if not filtered_df.empty:
        st.markdown("## 📈 Visualizaciones Principales")

        col_viz1, col_viz2 = st.columns(2)  # Renombrado para evitar confusión con col1, col2 de métricas

        with col_viz1:
            fig1 = create_stacked_bar_chart(filtered_df)
            st.plotly_chart(fig1, use_container_width=True)

        with col_viz2:
            fig2 = create_survival_pie_chart(filtered_df)
            st.plotly_chart(fig2, use_container_width=True)

        col_viz3, col_viz4 = st.columns(2)  # Renombrado

        with col_viz3:
            fig3 = create_age_histogram(filtered_df)
            st.plotly_chart(fig3, use_container_width=True)

        with col_viz4:
            fig4 = create_age_survival_boxplot(filtered_df)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("## 🔥 Análisis Avanzado")

        fig5 = create_correlation_heatmap(filtered_df)
        st.plotly_chart(fig5, use_container_width=True)

        col_viz5, col_viz6 = st.columns(2)  # Renombrado

        with col_viz5:
            fig6 = create_timeline_chart(filtered_df)
            st.plotly_chart(fig6, use_container_width=True)

        with col_viz6:
            fig7 = create_treatment_bar_chart(filtered_df)
            st.plotly_chart(fig7, use_container_width=True)

        st.markdown("## 📋 Tabla Interactiva de Datos")

        columns_to_show_options = filtered_df.columns.tolist()
        default_columns = ['PatientID', 'Gender', 'Age', 'TumorType', 'CancerStage',
                           'SurvivalStatus', 'FollowUpMonths', 'TreatmentType']
        # Asegurar que las columnas por defecto existen en el df
        valid_default_columns = [col for col in default_columns if col in columns_to_show_options]

        columns_to_show = st.multiselect(
            "Selecciona las columnas a mostrar:",
            options=columns_to_show_options,
            default=valid_default_columns
        )

        if columns_to_show:
            st.dataframe(
                filtered_df[columns_to_show],
                use_container_width=True,
                height=400
            )

        st.markdown("## 💾 Exportar Datos")

        col_export1, col_export2 = st.columns(2)

        with col_export1:
            # El botón de descarga en Streamlit funciona mejor si la generación de datos
            # y el widget st.download_button están juntos o el dato se genera antes.
            # Para evitar problemas con el re-renderizado, se genera el csv aquí.
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Datos Filtrados (CSV)",  # Cambiado texto del label
                data=csv_data,
                file_name=f"datos_oncologicos_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"  # El tipo "primary" para el botón
            )

        with col_export2:
            st.info(f"📊 Datos filtrados: {len(filtered_df):,} registros de {len(df):,} totales")

    # Información adicional
    st.markdown("---")
    st.markdown("## ℹ️ Información del Dataset")

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.markdown("""
        **📈 Variables Cuantitativas:**
        - Edad del paciente
        - Tamaño del tumor
        - Sesiones de quimioterapia
        - Sesiones de radiación
        - Meses de seguimiento
        """)

    with col_info2:
        st.markdown("""
        **🏷️ Variables Categóricas:**
        - Tipo de cáncer
        - Etapa del cáncer
        - Estado de supervivencia
        - Tipo de tratamiento
        - Mutaciones genéticas
        """)

    with col_info3:
        st.markdown("""
        **🌍 Variables Geográficas:**
        - Provincia de origen
        - Etnia del paciente
        - Distribución regional
        """)


if __name__ == "__main__":
    main()