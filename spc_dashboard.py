import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import time
from typing import List, Dict, Any

# Configuración de la página
st.set_page_config(
    page_title="Dynamic Manufacturing SPC Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para el tema oscuro
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #1e40af 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1f2937;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #374151;
        text-align: center;
    }
    .operator-id {
        font-size: 2rem;
        color: #60a5fa;
        font-weight: bold;
        border: 2px solid #60a5fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        background: #1f2937;
    }
    .section-header {
        background: #374151;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        font-weight: bold;
        color: white;
    }
    .stButton > button {
        background: #1f2937;
        color: white;
        border: 2px solid #60a5fa;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .parameter-row {
        background: #1f2937;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        border-left: 3px solid #60a5fa;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_columns = self._detect_numeric_columns()
        self.batch_column = self._detect_batch_column()
        
    def _detect_numeric_columns(self) -> List[str]:
        """Detecta automáticamente todas las columnas numéricas"""
        numeric_cols = []
        exclude_keywords = ['batch', ' id ', 'index', 'number', 'seq']
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Convertir a formato para comparación
                col_lower = f" {col.lower()} "
                # Excluir solo si la columna es exactamente una columna ID
                if not any(keyword in col_lower for keyword in exclude_keywords):
                    numeric_cols.append(col)
        
        return numeric_cols
    def _detect_batch_column(self) -> str:
        """Detecta la columna que representa el batch/secuencia"""
        possible_batch_cols = []
        batch_keywords = ['batch', 'number', 'seq', 'index', ' id ', 'time', 'order']
        exclude_keywords = ['acidity', 'fixed']
        
        for col in self.df.columns:
            col_lower = col.lower().replace("_", " ")
            
            # Verificar si la columna contiene palabras clave de batch
            if any(keyword in f" {col_lower} " for keyword in batch_keywords):
                # Excluir si contiene palabras que no deben ser consideradas batch
                if not any(exclude in col_lower for exclude in exclude_keywords):
                    if pd.api.types.is_numeric_dtype(self.df[col]) or pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        possible_batch_cols.append(col)
        
        if possible_batch_cols:
            return possible_batch_cols[0]
        else:
            # Si no se encuentra ninguna columna de batch, crear un índice automático
            return 'auto_index'
    
    def prepare_data(self, selected_columns: List[str]) -> pd.DataFrame:
        """Prepara los datos para análisis SPC"""
        if self.batch_column == 'auto_index':
            df_prepared = self.df.copy()
            df_prepared['auto_index'] = range(1, len(df_prepared) + 1)
        else:
            df_prepared = self.df.copy()
        
        # Convertir a formato largo para análisis
        id_vars = [self.batch_column]
        value_vars = selected_columns
        
        df_melted = pd.melt(
            df_prepared, 
            id_vars=id_vars, 
            value_vars=value_vars,
            var_name='Parameter', 
            value_name='Value'
        )
        
        return df_melted
    
    def calculate_control_limits(self, values: np.ndarray) -> Dict[str, float]:
        """Calcula límites de control automáticamente"""
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        # Límites de control (3 sigma)
        ucl = mean + 3 * std
        lcl = mean - 3 * std
        
        # Límites de especificación estimados (6 sigma del proceso)
        usl = mean + 3 * std
        lsl = mean - 3 * std
        
        return {
            'mean': mean,
            'std': std,
            'ucl': ucl,
            'lcl': lcl,
            'usl': usl,
            'lsl': lsl,
            'target': mean
        }

def calculate_spc_metrics(df: pd.DataFrame, parameter: str, batch_col: str) -> Dict[str, Any]:
    """Calcula métricas SPC para un parámetro específico"""
    param_data = df[df['Parameter'] == parameter].copy()
    param_data = param_data.sort_values(batch_col)
    
    values = param_data['Value'].values
    
    if len(values) == 0:
        return None
    
    # Calcular límites automáticamente
    analyzer = DataAnalyzer(pd.DataFrame())
    limits = analyzer.calculate_control_limits(values)
    
    # OOC (Out of Control) - puntos fuera de límites de control
    ooc_control = np.sum((values > limits['ucl']) | (values < limits['lcl']))
    ooc_spec = np.sum((values > limits['usl']) | (values < limits['lsl']))
    total_ooc = max(ooc_control, ooc_spec)
    
    ooc_percentage = (total_ooc / len(values)) * 100 if len(values) > 0 else 0
    
    # Pass/Fail basado en % OOC
    pass_fail = "PASS" if ooc_percentage < 5 else "FAIL"
    
    # Capability indices
    cp = (limits['usl'] - limits['lsl']) / (6 * limits['std']) if limits['std'] > 0 else 0
    cpk = min(
        (limits['usl'] - limits['mean']) / (3 * limits['std']),
        (limits['mean'] - limits['lsl']) / (3 * limits['std'])
    ) if limits['std'] > 0 else 0
    
    return {
        'count': len(values),
        'mean': limits['mean'],
        'std': limits['std'],
        'ucl': limits['ucl'],
        'lcl': limits['lcl'],
        'usl': limits['usl'],
        'lsl': limits['lsl'],
        'target': limits['target'],
        'ooc_percentage': ooc_percentage,
        'pass_fail': pass_fail,
        'values': values,
        'cp': cp,
        'cpk': cpk,
        'batch_values': param_data[batch_col].values
    }

def create_sparkline(values: np.ndarray) -> go.Figure:
    """Crea un mini gráfico sparkline"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values,
        mode='lines',
        line=dict(color='#fbbf24', width=2),
        showlegend=False
    ))
    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Crea un gráfico de medidor circular"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': 'white'},
            'bar': {'color': "#fbbf24"},
            'steps': [
                {'range': [0, 50], 'color': "#1f2937"},
                {'range': [50, 100], 'color': "#374151"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        },
        number={'font': {'color': 'white', 'size': 40}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300
    )
    return fig

def create_spc_chart(df: pd.DataFrame, parameter: str, batch_col: str) -> go.Figure:
    """Crea gráfico SPC principal"""
    metrics = calculate_spc_metrics(df, parameter, batch_col)
    
    if metrics is None:
        return go.Figure()
    
    param_data = df[df['Parameter'] == parameter].sort_values(batch_col)
    
    fig = go.Figure()
    
    # Línea principal de datos
    fig.add_trace(go.Scatter(
        x=metrics['batch_values'],
        y=metrics['values'],
        mode='lines+markers',
        name=parameter,
        line=dict(color='#fbbf24', width=2),
        marker=dict(size=6, color='#fbbf24')
    ))
    
    # Límites de especificación
    fig.add_hline(y=metrics['usl'], line_dash="dot", line_color="red", 
                  annotation_text=f"USL: {metrics['usl']:.3f}")
    fig.add_hline(y=metrics['lsl'], line_dash="dot", line_color="red", 
                  annotation_text=f"LSL: {metrics['lsl']:.3f}")
    
    # Límites de control
    fig.add_hline(y=metrics['ucl'], line_dash="dash", line_color="orange", 
                  annotation_text=f"UCL: {metrics['ucl']:.3f}")
    fig.add_hline(y=metrics['lcl'], line_dash="dash", line_color="orange", 
                  annotation_text=f"LCL: {metrics['lcl']:.3f}")
    
    # Target/Mean
    fig.add_hline(y=metrics['target'], line_dash="solid", line_color="blue", 
                  annotation_text=f"Mean: {metrics['target']:.3f}")
    
    # Puntos fuera de control
    ooc_mask = (
        (metrics['values'] > metrics['ucl']) | 
        (metrics['values'] < metrics['lcl']) |
        (metrics['values'] > metrics['usl']) | 
        (metrics['values'] < metrics['lsl'])
    )
    
    if np.any(ooc_mask):
        fig.add_trace(go.Scatter(
            x=metrics['batch_values'][ooc_mask],
            y=metrics['values'][ooc_mask],
            mode='markers',
            name='Out of Control',
            marker=dict(size=10, color='red', symbol='x')
        ))
    
    fig.update_layout(
        title=f"Live SPC Chart - {parameter}",
        xaxis_title="Batch/Sequence",
        yaxis_title="Value",
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font=dict(color='white'),
        height=400,
        showlegend=True
    )
    
    return fig

def create_pie_chart(df: pd.DataFrame, parameters: List[str], batch_col: str) -> go.Figure:
    """Crea gráfico de pie para % OOC por parámetro"""
    ooc_data = []
    
    for param in parameters:
        metrics = calculate_spc_metrics(df, param, batch_col)
        if metrics:
            ooc_data.append({
                'Parameter': param,
                'OOC_Percentage': max(metrics['ooc_percentage'], 0.1)  # Mínimo para visualización
            })
    
    if not ooc_data:
        return go.Figure()
    
    ooc_df = pd.DataFrame(ooc_data)
    
    fig = px.pie(
        ooc_df, 
        values='OOC_Percentage', 
        names='Parameter',
        title="% OOC per Parameter",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font=dict(color='white'),
        height=300
    )
    
    return fig

def create_histogram(df: pd.DataFrame, parameter: str) -> go.Figure:
    """Crea histograma de distribución"""
    param_data = df[df['Parameter'] == parameter]
    
    if param_data.empty:
        return go.Figure()
    
    fig = px.histogram(
        param_data, 
        x='Value', 
        nbins=20,
        title=f"Distribution - {parameter}",
        color_discrete_sequence=['#fbbf24']
    )
    
    # Añadir líneas de estadísticas
    mean_val = param_data['Value'].mean()
    std_val = param_data['Value'].std()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="blue", 
                  annotation_text=f"Mean: {mean_val:.3f}")
    fig.add_vline(x=mean_val + std_val, line_dash="dot", line_color="orange", 
                  annotation_text=f"+1σ")
    fig.add_vline(x=mean_val - std_val, line_dash="dot", line_color="orange", 
                  annotation_text=f"-1σ")
    
    fig.update_layout(
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font=dict(color='white'),
        height=300,
        xaxis_title="Value",
        yaxis_title="Count"
    )
    
    return fig

def generate_sample_data() -> pd.DataFrame:
    """Genera datos de ejemplo con múltiples parámetros"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Batch_Number': range(1, n_samples + 1),
        'Temperature': np.random.normal(250, 5, n_samples),
        'Pressure': np.random.normal(15.5, 0.3, n_samples),
        'Flow_Rate': np.random.normal(100, 2, n_samples),
        'Thickness': np.random.normal(0.42, 0.005, n_samples),
        'Density': np.random.normal(2.5, 0.1, n_samples),
        'Viscosity': np.random.normal(1000, 20, n_samples),
        'pH_Level': np.random.normal(7.2, 0.1, n_samples),
        'Conductivity': np.random.normal(1.8, 0.05, n_samples)
    }
    
    return pd.DataFrame(data)

# Título principal
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; text-align: center;">DYNAMIC MANUFACTURING SPC DASHBOARD</h1>
    <p style="color: #e5e7eb; margin: 0; text-align: center;">Automatic Process Control and Exception Reporting</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para cargar datos y filtros
with st.sidebar:
    st.header("📁 Data Configuration")
    
    # Upload de archivo
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload your manufacturing data CSV file"
    )
    
    if uploaded_file is None:
        if st.button("🔄 Use Sample Data", type="primary"):
            st.session_state['raw_data'] = generate_sample_data()
            st.success("✅ Sample data loaded!")
    else:
        # Reemplazar el bloque try/except existente cuando se carga el archivo
        try:
            # Leer CSV asegurando que se respeten los nombres de columnas exactamente como están
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
            st.session_state['raw_data'] = df_raw
            st.success("✅ Data loaded successfully!")
            
            # Mostrar info del dataset
            st.info(f"📊 Dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
            
            # Mostrar las primeras columnas para confirmar que se cargaron correctamente
            with st.expander("View sample data"):
                st.write(df_raw.head())
            
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

# Verificar si hay datos
if 'raw_data' not in st.session_state:
    st.info("👆 Please upload a CSV file or use sample data to get started")
    st.stop()

df_raw = st.session_state['raw_data']

# Inicializar analizador
analyzer = DataAnalyzer(df_raw)

# Sidebar - Filtros de columnas
with st.sidebar:
    st.markdown("---")
    st.header("🎯 Parameter Selection")
    
    if analyzer.numeric_columns:
        # Opción de seleccionar todas
        select_all = st.checkbox("📋 Select All Parameters", value=True)
        
        if select_all:
            selected_columns = analyzer.numeric_columns
        else:
            selected_columns = st.multiselect(
                "Choose parameters to analyze:",
                options=analyzer.numeric_columns,
                default=analyzer.numeric_columns[:3] if len(analyzer.numeric_columns) >= 3 else analyzer.numeric_columns,
                help="Select one or more parameters for SPC analysis"
            )
    else:
        st.error("❌ No numeric columns found in the dataset")
        st.stop()
    
    # Mostrar información de columnas detectadas
    st.markdown("---")
    st.subheader("📋 Detected Columns")
    st.write(f"**Batch/Sequence Column:** {analyzer.batch_column}")
    st.write(f"**Numeric Columns:** {len(analyzer.numeric_columns)}")
    
    with st.expander("View all numeric columns"):
        for col in analyzer.numeric_columns:
            st.write(f"• {col}")

# Verificar selección
if not selected_columns:
    st.warning("⚠️ Please select at least one parameter to analyze")
    st.stop()

# Preparar datos para análisis
df_analysis = analyzer.prepare_data(selected_columns)

# Layout principal
col1, col2, col3 = st.columns([1, 2, 1])

# Panel izquierdo
with col1:
    st.markdown('<div class="section-header">OPERATOR CONTROLS</div>', unsafe_allow_html=True)
    
    # Operator ID (dinámico basado en datos)
    operator_id = f"{len(selected_columns)}{len(df_raw):02d}"
    st.markdown(f'<div class="operator-id">{operator_id}</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9ca3af;'>Operator ID</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Completion gauge (basado en % de datos procesados)
    completion_percentage = min(100, (len(df_raw) / 1000) * 100)
    gauge_fig = create_gauge_chart(completion_percentage, "Data Completeness")
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Control buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 START", type="primary"):
            st.success("Analysis started!")
    with col_btn2:
        if st.button("⏹️ STOP"):
            st.info("Analysis stopped!")

# Panel central
with col2:
    st.markdown('<div class="section-header">SPECIFICATION SETTINGS</div>', unsafe_allow_html=True)
    
    # Tabla de métricas de proceso
    st.subheader("Process Control Metrics Summary")
    
    # Crear tabla de métricas
    metrics_data = []
    
    for param in selected_columns:
        metrics = calculate_spc_metrics(df_analysis, param, analyzer.batch_column)
        if metrics:
            metrics_data.append({
                'Parameter': param,
                'Count': metrics['count'],
                'Mean': f"{metrics['mean']:.3f}",
                'Std': f"{metrics['std']:.3f}",
                'OOC%': f"{metrics['ooc_percentage']:.1f}%",
                'Cp': f"{metrics['cp']:.2f}",
                'Cpk': f"{metrics['cpk']:.2f}",
                'Pass/Fail': metrics['pass_fail'],
                'ooc_numeric': metrics['ooc_percentage']
            })
    
    # Mostrar tabla con sparklines
    for i, row in enumerate(metrics_data):
        param = row['Parameter']
        metrics = calculate_spc_metrics(df_analysis, param, analyzer.batch_column)
        
        # Crear fila con estilo
        st.markdown(f'<div class="parameter-row">', unsafe_allow_html=True)
        
        col_param, col_count, col_spark, col_stats, col_ooc, col_pass = st.columns([2, 1, 2, 2, 1, 1])
        
        with col_param:
            st.write(f"**{param}**")
        with col_count:
            st.metric("Count", row['Count'])
        with col_spark:
            if metrics and len(metrics['values']) > 0:
                spark_fig = create_sparkline(metrics['values'][-30:])  # Últimos 30 puntos
                st.plotly_chart(spark_fig, use_container_width=True)
        with col_stats:
            st.write(f"μ: {row['Mean']}")
            st.write(f"σ: {row['Std']}")
            st.write(f"Cpk: {row['Cpk']}")
        with col_ooc:
            ooc_val = row['ooc_numeric']
            color = "🟢" if ooc_val < 2 else "🟡" if ooc_val < 5 else "🔴"
            st.metric("OOC%", row['OOC%'], delta=None)
            st.write(color)
        with col_pass:
            status_color = "🟢" if row['Pass/Fail'] == 'PASS' else "🔴"
            st.write(f"{status_color}")
            st.write(row['Pass/Fail'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    # Gráfico SPC principal
    st.subheader("Live SPC Chart")
    selected_param_chart = st.selectbox(
        "Select Parameter for SPC Chart:", 
        selected_columns, 
        index=0,
        key="spc_chart_param"
    )
    
    spc_fig = create_spc_chart(df_analysis, selected_param_chart, analyzer.batch_column)
    st.plotly_chart(spc_fig, use_container_width=True)

# Panel derecho
with col3:
    st.markdown('<div class="section-header">CONTROL CHARTS DASHBOARD</div>', unsafe_allow_html=True)
    
    # Gráfico de pie
    pie_fig = create_pie_chart(df_analysis, selected_columns, analyzer.batch_column)
    st.plotly_chart(pie_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Histograma
    hist_param = st.selectbox(
        "Parameter for Distribution:", 
        selected_columns, 
        index=0, 
        key="hist_param"
    )
    hist_fig = create_histogram(df_analysis, hist_param)
    st.plotly_chart(hist_fig, use_container_width=True)

# Footer con estadísticas en tiempo real
st.markdown("---")
st.subheader("📊 Real-time Statistics")

col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)

with col_stat1:
    total_samples = len(df_raw)
    st.metric("Total Samples", total_samples)

with col_stat2:
    total_parameters = len(selected_columns)
    st.metric("Parameters Analyzed", total_parameters)

with col_stat3:
    if metrics_data:
        avg_ooc = np.mean([float(row['OOC%'].replace('%', '')) for row in metrics_data])
        st.metric("Avg OOC%", f"{avg_ooc:.1f}%")

with col_stat4:
    if metrics_data:
        pass_count = sum([1 for row in metrics_data if row['Pass/Fail'] == 'PASS'])
        st.metric("Parameters Passing", f"{pass_count}/{len(metrics_data)}")

with col_stat5:
    if metrics_data:
        avg_cpk = np.mean([float(row['Cpk']) for row in metrics_data])
        cpk_status = "🟢" if avg_cpk > 1.33 else "🟡" if avg_cpk > 1.0 else "🔴"
        st.metric("Avg Cpk", f"{avg_cpk:.2f} {cpk_status}")

# Controles de actualización
st.markdown("---")
col_refresh1, col_refresh2, col_refresh3 = st.columns([1, 1, 2])

with col_refresh1:
    if st.button("🔄 Refresh Analysis", type="primary"):
        st.rerun()

with col_refresh2:
    auto_refresh = st.checkbox("🔄 Auto-refresh (30s)")

with col_refresh3:
    if st.button("📥 Download Report"):
        # Crear reporte CSV
        report_df = pd.DataFrame(metrics_data)
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV Report",
            data=csv,
            file_name=f"spc_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Auto-refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Información adicional en sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ Analysis Info")
    st.info(f"""
    **Dataset Summary:**
    - Total Rows: {len(df_raw):,}
    - Selected Parameters: {len(selected_columns)}
    - Batch Column: {analyzer.batch_column}
    
    **SPC Calculations:**
    - Control Limits: ±3σ
    - Specification Limits: Auto-detected
    - Capability Index: Cp, Cpk
    """)