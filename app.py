"""
🔬 Orquestador de IAs - Modo Ciencia (Fase 1)
Roles: Revisor, Metodólogo, Asesor, Estadístico, Editor, Revisor Crítico
"""

import os
import sys
import json
from datetime import datetime, date
from openai import OpenAI
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import streamlit as st

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    st.error("❌ No se encontró OPENROUTER_API_KEY. Crea un archivo .env con tu API key.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/luislinares-tech/orquestador-ias",
        "X-Title": "Orquestador de IAs - Modo Ciencia"
    }
)

TASA_USD_BRL = 5.15  # Actualiza esta tasa según el día
LIMITE_DIARIO_USD = 2.0

# =============================================================================
# ROLES CIENTÍFICOS
# =============================================================================

ROLES = {
    "📚 Revisor de Literatura": {
        "descripcion": "Busca, resume y sintetiza literatura científica. Identifica gaps y tendencias.",
        "prompt_sistema": """Eres un revisor de literatura experto. Tu trabajo es:
1. Buscar y sintetizar información de papers científicos
2. Identificar gaps en la literatura
3. Proponer líneas de investigación futuras
4. Organizar hallazgos por temas cronológicos o temáticos
5. Evaluar la calidad de las evidencias encontradas""",
        "ia_preferida": ["Gemini", "Kimi", "ChatGPT"],
        "color": "#9B59B6"
    },
    "🔬 Metodólogo": {
        "descripcion": "Evalúa diseños experimentales, validez, confiabilidad y rigor metodológico.",
        "prompt_sistema": """Eres un metodólogo experto en investigación científica. Tu trabajo es:
1. Evaluar la adecuación del diseño experimental
2. Identificar amenazas a la validez interna y externa
3. Sugerir mejoras metodológicas
4. Evaluar el tamaño de muestra y poder estadístico
5. Revisar la consistencia entre objetivos, hipótesis y métodos""",
        "ia_preferida": ["Claude", "DeepSeek", "ChatGPT"],
        "color": "#E74C3C"
    },
    "👨‍🏫 Asesor Académico": {
        "descripcion": "Orienta en estructura de tesis, objetivos, marco teórico y cronograma.",
        "prompt_sistema": """Eres un asesor académico con 20 años de experiencia. Tu trabajo es:
1. Sugerir estructura de tesis/artículos según la disciplina
2. Formular objetivos e hipótesis claros y medibles
3. Proponer marcos teóricos relevantes
4. Crear cronogramas realistas de investigación
5. Identificar riesgos y contingencias del proyecto""",
        "ia_preferida": ["ChatGPT", "Claude", "Kimi"],
        "color": "#3498DB"
    },
    "📊 Experto en Estadística": {
        "descripcion": "Sugiere pruebas estadísticas, interpreta resultados, genera código R/SPSS.",
        "prompt_sistema": """Eres un estadístico experto. Tu trabajo es:
1. Sugerir pruebas estadísticas apropiadas según los datos y diseño
2. Interpretar resultados estadísticos (p-values, IC, tamaño del efecto)
3. Generar código en R, Python o SPSS para análisis
4. Evaluar supuestos de las pruebas (normalidad, homocedasticidad, etc.)
5. Sugerir tamaño de muestra y análisis de poder""",
        "ia_preferida": ["DeepSeek", "ChatGPT", "Claude"],
        "color": "#2ECC71"
    },
    "✍️ Editor Académico": {
        "descripcion": "Revisa gramática, estilo, coherencia y formato académico.",
        "prompt_sistema": """Eres un editor académico profesional. Tu trabajo es:
1. Corregir gramática, ortografía y puntuación
2. Mejorar claridad, concisión y flujo del texto
3. Verificar consistencia terminológica
4. Sugerir mejoras en la estructura de párrafos
5. Asegurar tono académico apropiado""",
        "ia_preferida": ["ChatGPT", "Kimi", "Claude"],
        "color": "#F39C12"
    },
    "🔍 Revisor Crítico": {
        "descripcion": "Peer review simulado: evalúa fortalezas, debilidades y validez del trabajo.",
        "prompt_sistema": """Eres un revisor crítico experto (simulando peer review para revista indexada). Tu trabajo es:
1. Evaluar la originalidad y contribución del trabajo
2. Identificar fortalezas y debilidades metodológicas
3. Detectar sesgos, conflictos de interés o limitaciones no declaradas
4. Evaluar la calidad de las referencias bibliográficas
5. Dar una recomendación: Aceptar, Revisar Menor, Revisar Mayor, Rechazar""",
        "ia_preferida": ["Claude", "DeepSeek", "ChatGPT"],
        "color": "#E67E22"
    }
}

# =============================================================================
# DICCIONARIO DE IAS
# =============================================================================

IAS = {
    "DeepSeek": {
        "modelo": "deepseek/deepseek-r1",
        "palabras_clave": [
            "código", "programa", "debug", "error", "math", "matemática",
            "algoritmo", "lógica", "función", "clase", "programación",
            "sintaxis", "compilar", "ejecutar", "optimizar", "complejidad",
            "python", "javascript", "java", "cpp", "sql", "api", "backend",
            "frontend", "script", "bug", "fix", "github", "vscode", 
            "visual studio", "microsoft", "terminal", "shell", "bash", "powershell",
            "estadística", "spss", "r-project", "regresión", "anova", "t-test"
        ],
        "precios": (0.14, 2.19),
        "descripcion": "Excelente para código, matemáticas, debugging y lógica compleja",
        "emoji": "💻",
        "color": "#FF6B6B"
    },
    "ChatGPT": {
        "modelo": "openai/gpt-4o",
        "palabras_clave": [
            "creativo", "creatividad", "imagen", "marketing", "brainstorming",
            "lluvia de ideas", "diseño", "campaña", "publicidad", "branding",
            "estrategia", "contenido", "redes sociales", "copywriting",
            "poema", "cuento", "historia", "guion", "novela",
            "editar", "redactar", "escribir", "tesis", "artículo", "publicación"
        ],
        "precios": (5.0, 15.0),
        "descripcion": "Ideal para creatividad, marketing, escritura académica y brainstorming",
        "emoji": "🎨",
        "color": "#4ECDC4"
    },
    "Gemini": {
        "modelo": "google/gemini-2.5-pro",
        "palabras_clave": [
            "buscar", "internet", "web", "google", "noticias", "actualidad",
            "tiempo real", "búsqueda", "investigar", "noticia", "hoy",
            "última hora", "clima", "precio", "cotización", "trend", "trending",
            "literatura", "papers", "doi", "pubmed", "arxiv", "scielo", "scholar"
        ],
        "precios": (1.5, 10.0),
        "descripcion": "Búsqueda en línea, información actualizada, noticias e investigación web",
        "emoji": "🔍",
        "color": "#45B7D1"
    },
    "Claude": {
        "modelo": "anthropic/claude-sonnet-4.6",
        "palabras_clave": [
            "ético", "moral", "dilema", "decisión difícil", "reflexivo",
            "consecuencias", "valores", "principios", "integridad",
            "justicia", "responsabilidad", "empatía", "perspectiva",
            "filosofía", "análisis profundo", "crítico", "sesgo",
            "metodología", "validez", "confiabilidad", "rigor", "peer review"
        ],
        "precios": (3.0, 15.0),
        "descripcion": "Análisis ético, decisiones difíciles, metodología y tono reflexivo",
        "emoji": "🤔",
        "color": "#96CEB4"
    },
    "Kimi": {
        "modelo": "moonshotai/kimi-k3",
        "palabras_clave": [
            "contrato", "legal", "documento largo", "resumen extenso",
            "analizar contrato", "cláusula", "acuerdo", "legislación",
            "normativa", "documento extenso", "pdf", "libro", "tesis",
            "memoria", "extenso", "largo", "páginas", "literatura", "revisión"
        ],
        "precios": (2.0, 8.0),
        "descripcion": "Especialista en documentos largos, contratos, resúmenes y revisión de literatura",
        "emoji": "📚",
        "color": "#FFEAA7"
    }
}

ORDEN_PRIORIDAD = ["DeepSeek", "ChatGPT", "Gemini", "Claude", "Kimi"]

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def convertir_a_reales(dolares: float) -> float:
    """Convierte USD a BRL"""
    return round(dolares * TASA_USD_BRL, 4)

def detectar_ia(prompt: str, rol_preferido: Optional[List[str]] = None) -> Tuple[List[str], Dict]:
    prompt_lower = prompt.lower()
    puntuaciones = {}
    
    for nombre, config in IAS.items():
        palabras = config["palabras_clave"]
        coincidencias = sum(1 for palabra in palabras if palabra.lower() in prompt_lower)
        puntuaciones[nombre] = coincidencias
    
    # Si hay rol preferido, dar prioridad a esas IAs
    if rol_preferido:
        def prioridad_rol(nombre):
            if nombre in rol_preferido:
                return (puntuaciones[nombre], 100 - rol_preferido.index(nombre))
            return (puntuaciones[nombre], 0)
        
        ias_ordenadas = sorted(
            puntuaciones.keys(),
            key=lambda n: prioridad_rol(n),
            reverse=True
        )
    else:
        ias_ordenadas = sorted(
            puntuaciones.keys(),
            key=lambda n: (puntuaciones[n], ORDEN_PRIORIDAD.index(n) if n in ORDEN_PRIORIDAD else 999),
            reverse=True
        )
    
    return ias_ordenadas, puntuaciones

def llamar_ia(prompt: str, modelo: str, sistema: str = "Eres un asistente útil.") -> Tuple[str, Dict]:
    try:
        response = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        respuesta_texto = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        return respuesta_texto, usage
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")

def calcular_costo(usage: Dict, nombre_ia: str) -> Tuple[float, float]:
    precio_entrada, precio_salida = IAS[nombre_ia]["precios"]
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    costo_usd = (prompt_tokens * precio_entrada + completion_tokens * precio_salida) / 1_000_000
    costo_brl = convertir_a_reales(costo_usd)
    
    return round(costo_usd, 6), round(costo_brl, 4)

def verificar_limite_diario(costo_nuevo: float) -> bool:
    """Verifica si el gasto diario excede el límite"""
    hoy = date.today().isoformat()
    
    if 'gastos_diarios' not in st.session_state:
        st.session_state.gastos_diarios = {}
    
    if hoy not in st.session_state.gastos_diarios:
        st.session_state.gastos_diarios[hoy] = 0.0
    
    gasto_actual = st.session_state.gastos_diarios[hoy]
    
    if gasto_actual + costo_nuevo > LIMITE_DIARIO_USD:
        return False
    
    st.session_state.gastos_diarios[hoy] = gasto_actual + costo_nuevo
    return True

# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="🔬 Orquestador de IAs - Modo Ciencia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .rol-card {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border: 2px solid transparent;
        transition: all 0.3s;
    }
    .rol-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .costo-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .limite-alerta {
        background: #e74c3c;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔬 Orquestador de IAs - Modo Ciencia</div>', unsafe_allow_html=True)
st.markdown("### *Asistente inteligente para investigación académica*")

# Inicializar session state
if 'historial' not in st.session_state:
    st.session_state.historial = []
    st.session_state.costo_total = 0.0
    st.session_state.gastos_diarios = {date.today().isoformat(): 0.0}

# Sidebar
with st.sidebar:
    st.header("🎭 Roles Científicos")
    
    for nombre_rol, config in ROLES.items():
        st.markdown(f"""
        <div class="rol-card" style="background-color: {config['color']}15; border-color: {config['color']}30;">
            <b>{nombre_rol}</b><br>
            <small>{config['descripcion']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🤖 IAs Disponibles")
    
    for nombre, config in IAS.items():
        st.markdown(f"""
        <div style="background-color: {config['color']}20; padding: 10px; border-radius: 8px; margin: 5px 0;">
            <b>{config['emoji']} {nombre}</b><br>
            <small>{config['descripcion']}</small><br>
            <small>💰 ${config['precios'][0]} / ${config['precios'][1]} por 1M tokens</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mostrar límite diario
    hoy = date.today().isoformat()
    gasto_hoy = st.session_state.gastos_diarios.get(hoy, 0.0)
    gasto_brl = convertir_a_reales(gasto_hoy)
    
    st.header("💰 Control de Gastos")
    
    if gasto_hoy >= LIMITE_DIARIO_USD * 0.9:
        st.markdown(f"""
        <div class="limite-alerta">
            ⚠️ ALERTA: Has gastado ${gasto_hoy:.4f} / ${LIMITE_DIARIO_USD:.2f} USD<br>
            (R$ {gasto_brl:.2f} BRL)<br>
            ¡Cerca del límite diario!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="costo-box">
            Gasto hoy:<br>
            <b>${gasto_hoy:.4f} USD</b><br>
            <b>R$ {gasto_brl:.2f} BRL</b><br>
            <small>Límite: ${LIMITE_DIARIO_USD:.2f} USD</small>
        </div>
        """, unsafe_allow_html=True)

# Área principal
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎭 Selecciona tu Rol")
    
    rol_seleccionado = st.radio(
        "¿Qué necesitas?",
        list(ROLES.keys()),
        label_visibility="collapsed"
    )
    
    rol_config = ROLES[rol_seleccionado]
    
    st.info(f"**{rol_seleccionado}**\n\n{rol_config['descripcion']}")
    
    st.markdown("---")
    
    # Opciones avanzadas
    with st.expander("⚙️ Opciones avanzadas"):
        forzar_ia = st.selectbox(
            "Forzar IA específica (opcional)",
            ["Auto (recomendado)"] + list(IAS.keys())
        )
        
        max_tokens = st.slider("Máximo de tokens", 500, 4000, 2000)
        
        formato_salida = st.selectbox(
            "Formato de salida",
            ["Texto libre", "Lista numerada", "Tabla", "JSON", "Markdown"]
        )

with col2:
    st.subheader("📝 Describe tu tarea")
    
    prompt_usuario = st.text_area(
        "¿Qué necesitas analizar?",
        placeholder=f"Ej: Como {rol_seleccionado.split()[1]}, necesito que...",
        height=200
    )
    
    col_enviar, col_limpiar = st.columns([3, 1])
    
    with col_enviar:
        enviar = st.button("🚀 Analizar", type="primary", use_container_width=True)
    
    with col_limpiar:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.historial = []
            st.session_state.costo_total = 0.0
            st.rerun()

# Procesar cuando se envía
if enviar and prompt_usuario:
    # Verificar límite estimado antes de ejecutar
    costo_estimado = 0.005  # Estimación conservadora
    
    if not verificar_limite_diario(costo_estimado):
        st.error(f"❌ ¡Límite diario alcanzado! Has gastado ${st.session_state.gastos_diarios[date.today().isoformat()]:.4f} USD")
        st.stop()
    
    with st.spinner(f"🔍 Analizando como {rol_seleccionado}..."):
        # Detectar IA según rol y prompt
        if forzar_ia != "Auto (recomendado)":
            ias_ordenadas = [forzar_ia] + [ia for ia in IAS.keys() if ia != forzar_ia]
            puntuaciones = {ia: 999 if ia == forzar_ia else 0 for ia in IAS.keys()}
        else:
            ias_ordenadas, puntuaciones = detectar_ia(prompt_usuario, rol_config["ia_preferida"])
        
        ia_elegida = ias_ordenadas[0]
        config_ia = IAS[ia_elegida]
    
    # Mostrar detección
    st.markdown("---")
    st.subheader("🔍 IA Seleccionada")
    
    cols = st.columns(len(IAS))
    for idx, (nombre, conf) in enumerate(IAS.items()):
        with cols[idx]:
            puntos = puntuaciones.get(nombre, 0)
            es_elegida = nombre == ia_elegida
            borde = f"3px solid {conf['color']}" if es_elegida else "1px solid #ddd"
            bg = f"{conf['color']}20" if es_elegida else "transparent"
            
            st.markdown(f"""
            <div style="border: {borde}; border-radius: 10px; padding: 10px; text-align: center; background: {bg};">
                <div style="font-size: 1.5rem;">{conf['emoji']}</div>
                <b>{nombre}</b><br>
                <small>{puntos} pts</small>
                {'<br>✅ <b>ELEGIDA</b>' if es_elegida else ''}
            </div>
            """, unsafe_allow_html=True)
    
    # Ejecutar
    with st.spinner(f"🚀 Consultando {config_ia['emoji']} {ia_elegida}..."):
        try:
            # Usar el prompt del rol seleccionado
            sistema_completo = f"""{rol_config['prompt_sistema']}

Formatea tu respuesta en {formato_salida}.
Sé conciso pero completo. Usa ejemplos cuando sea útil."""
            
            respuesta, usage = llamar_ia(
                prompt_usuario, 
                config_ia["modelo"],
                sistema=sistema_completo
            )
            
            costo_usd, costo_brl = calcular_costo(usage, ia_elegida)
            
            # Verificar límite real
            if not verificar_limite_diario(costo_usd - costo_estimado):
                st.error("❌ Límite diario excedido con el costo real")
                st.stop()
            
            # Guardar en historial
            st.session_state.historial.append({
                "timestamp": datetime.now().isoformat(),
                "rol": rol_seleccionado,
                "prompt": prompt_usuario,
                "ia": ia_elegida,
                "costo_usd": costo_usd,
                "costo_brl": costo_brl,
                "tokens": usage["total_tokens"],
                "respuesta": respuesta
            })
            st.session_state.costo_total += costo_usd
            
            # Mostrar respuesta
            st.markdown("---")
            st.subheader(f"✅ Resultado de {config_ia['emoji']} {ia_elegida}")
            
            # Métricas
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Tokens entrada", usage["prompt_tokens"])
            with col_m2:
                st.metric("Tokens salida", usage["completion_tokens"])
            with col_m3:
                st.metric("Costo USD", f"${costo_usd:.6f}")
            with col_m4:
                st.metric("Costo BRL", f"R$ {costo_brl:.4f}")
            
            # Respuesta
            st.markdown("### 📄 Respuesta:")
            
            if formato_salida == "JSON":
                try:
                    st.json(json.loads(respuesta))
                except:
                    st.markdown(respuesta)
            else:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid {config_ia['color']};">
                    {respuesta}
                </div>
                """, unsafe_allow_html=True)
            
            # Botones de acción
            st.markdown("---")
            col_acc1, col_acc2 = st.columns(2)
            
            with col_acc1:
                st.download_button(
                    "📥 Descargar respuesta (.txt)",
                    respuesta,
                    file_name=f"respuesta_{ia_elegida}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col_acc2:
                if ia_elegida == "DeepSeek" and "```" in respuesta:
                    # Extraer código si hay bloques de código
                    import re
                    codigos = re.findall(r'```(?:\w+)?\n(.*?)```', respuesta, re.DOTALL)
                    if codigos:
                        codigo_completo = "\n\n".join(codigos)
                        st.download_button(
                            "📥 Exportar código (.py/.R)",
                            codigo_completo,
                            file_name=f"codigo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                            mime="text/plain"
                        )
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Intentando con fallback...")
            
            for fallback_ia in ias_ordenadas[1:]:
                try:
                    fallback_config = IAS[fallback_ia]
                    respuesta, usage = llamar_ia(
                        prompt_usuario, 
                        fallback_config["modelo"],
                        sistema=rol_config['prompt_sistema']
                    )
                    costo_usd, costo_brl = calcular_costo(usage, fallback_ia)
                    
                    st.session_state.historial.append({
                        "timestamp": datetime.now().isoformat(),
                        "rol": rol_seleccionado,
                        "prompt": prompt_usuario,
                        "ia": fallback_ia,
                        "costo_usd": costo_usd,
                        "costo_brl": costo_brl,
                        "tokens": usage["total_tokens"],
                        "respuesta": respuesta
                    })
                    st.session_state.costo_total += costo_usd
                    
                    st.success(f"✅ Fallback exitoso: {fallback_ia}")
                    st.markdown(respuesta)
                    break
                    
                except:
                    continue

# Historial
if st.session_state.historial:
    st.markdown("---")
    st.subheader("📜 Historial de Análisis")
    
    for idx, item in enumerate(reversed(st.session_state.historial[-10:])):
        with st.expander(f"#{len(st.session_state.historial) - idx} | {item['rol']} | {item['ia']} | ${item['costo_usd']:.6f} | {item['timestamp'][:16]}"):
            col_h1, col_h2 = st.columns([1, 3])
            
            with col_h1:
                st.write(f"**Rol:** {item['rol']}")
                st.write(f"**IA:** {item['ia']}")
                st.write(f"**Tokens:** {item['tokens']}")
                st.write(f"**Costo USD:** ${item['costo_usd']:.6f}")
                st.write(f"**Costo BRL:** R$ {item['costo_brl']:.4f}")
            
            with col_h2:
                st.write("**Prompt:**")
                st.info(item['prompt'][:200] + "...")
                st.write("**Respuesta:**")
                st.markdown(item['respuesta'][:500] + "...")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <small>🔬 <b>Orquestador de IAs - Modo Ciencia</b> | 
    DeepSeek 💻 | ChatGPT 🎨 | Gemini 🔍 | Claude 🤔 | Kimi 📚 | 
    <br>💰 Límite diario: $2.00 USD | Desarrollado con Streamlit + OpenRouter</small>
</div>
""", unsafe_allow_html=True)