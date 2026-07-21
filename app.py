from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

load_dotenv()

APP_NAME = "Orquestador de IAs · Modo Ciencia"
APP_VERSION = "Fase 2.0"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = ("gemini-3.1-flash-lite", "gemini-3.5-flash")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "3000"))
MAX_PDF_MB = 20
MAX_PDFS = 5
MAX_URLS = 10

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(120,120,120,.18);}
        .hero {
            padding: 1.05rem 1.25rem;
            border: 1px solid rgba(99,102,241,.22);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(99,102,241,.10), rgba(16,185,129,.07));
            margin-bottom: .8rem;
        }
        .hero h1 {margin: 0; font-size: 2rem; line-height: 1.2;}
        .hero p {margin: .35rem 0 0 0; opacity: .82;}
        .profile-card {
            border: 1px solid rgba(120,120,120,.20);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(120,120,120,.045);
            min-height: 120px;
        }
        .profile-card h3 {margin: 0 0 .35rem 0;}
        .badge {
            display: inline-block;
            padding: .18rem .55rem;
            border-radius: 999px;
            font-size: .78rem;
            border: 1px solid rgba(99,102,241,.28);
            background: rgba(99,102,241,.09);
            margin-right: .35rem;
        }
        .source-note {
            padding: .75rem .9rem;
            border-radius: 12px;
            background: rgba(16,185,129,.07);
            border: 1px solid rgba(16,185,129,.18);
        }
        .small-muted {opacity: .72; font-size: .9rem;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(120,120,120,.16);
            padding: .55rem .75rem;
            border-radius: 14px;
            background: rgba(120,120,120,.035);
        }
        .stButton > button, .stDownloadButton > button {border-radius: 11px;}
        textarea {border-radius: 12px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# REGLAS COMUNES Y PERFILES
# ============================================================

COMMON_RULES = """
REGLAS TRANSVERSALES OBLIGATORIAS

1. Trabaja como especialista científico senior, pero no reemplaces el juicio profesional,
   la revisión por pares ni las decisiones de un comité de ética.
2. Distingue siempre entre: información proporcionada, inferencias razonables,
   supuestos provisionales y datos faltantes.
3. No inventes referencias, autores, DOI, resultados, tamaños de muestra, normas,
   páginas, citas textuales ni decisiones editoriales. Si una fuente no está disponible,
   dilo expresamente.
4. Conserva exactamente nombres, cifras, tratamientos, variables, citas y datos del usuario,
   salvo que solicite modificarlos. Señala cualquier contradicción antes de corregirla.
5. No presentes una recomendación metodológica o estadística como automática: explica
   qué estructura de datos, supuestos y objetivo inferencial la justifican.
6. Cuando analices PDFs, URLs o videos, basa la respuesta en el contenido accesible y
   separa claramente lo que proviene de cada fuente.
7. Evita lenguaje grandilocuente. Sé preciso, crítico, aplicable y transparente.
8. Responde en el idioma principal del usuario. Mantén terminología técnica correcta.
9. Cuando falte información esencial, entrega primero lo que sí puede evaluarse y luego
   formula preguntas concretas, no una lista interminable de vaguedades.
10. Finaliza con una acción prioritaria y ejecutable.
""".strip()

PROFILES: Dict[str, Dict[str, Any]] = {
    "📚 Revisor de Literatura": {
        "short": "Diseña, audita y mejora revisiones científicas sin fabricar referencias.",
        "engine": "Gemini principal",
        "temperature": 0.15,
        "tasks": {
            "Protocolo PRISMA-P / registro": """
Construye o audita un protocolo de revisión. Determina primero el tipo de revisión y su
compatibilidad con PRISMA-P, PRISMA 2020, PRISMA-ScR, JBI u otra guía pertinente.
Incluye: título, justificación, pregunta, objetivos, criterios de elegibilidad, fuentes,
estrategia de búsqueda reproducible, gestión de registros y duplicados, cribado por pares,
extracción, evaluación crítica, síntesis, manejo de discrepancias, enmiendas, ética,
registro y plan de difusión. No asumas elegibilidad para PROSPERO: evalúala y advierte
cuando el registro adecuado sea OSF u otro repositorio.
""",
            "Revisión sistemática": """
Diseña o audita una revisión sistemática. Formula la pregunta con el marco adecuado
(PICO, PECO, PEO, SPIDER u otro), define elegibilidad, estrategia multibase, proceso
PRISMA 2020, evaluación del riesgo de sesgo, extracción, síntesis narrativa o cuantitativa,
certeza de la evidencia y trazabilidad de exclusiones.
""",
            "Scoping review": """
Diseña o audita una scoping review conforme al propósito de mapear conceptos, evidencia,
actores, métodos y vacíos. Usa PCC cuando corresponda, delimita alcance, fuentes grises si
son pertinentes, proceso de selección, charting de datos y reporte compatible con PRISMA-ScR.
No conviertas una scoping review en una revisión sistemática de efectividad disfrazada.
""",
            "Revisión bibliométrica": """
Diseña o audita una revisión bibliométrica reproducible. Define bases, periodo, tipos
documentales, deduplicación, normalización de autores y afiliaciones, indicadores de
producción, impacto y colaboración, coautoría, cocitación, acoplamiento bibliográfico,
coocurrencia de términos, sensibilidad a la base y software. Separa bibliometría de
síntesis temática y evita inferir calidad científica solo por citas.
""",
            "Revisión integrativa": """
Estructura una revisión integrativa con pregunta explícita, búsqueda amplia, evaluación
metodológica compatible con diseños heterogéneos, extracción estandarizada, reducción,
comparación y síntesis de datos. Justifica por qué es integrativa y no narrativa o sistemática.
""",
            "Revisión narrativa crítica": """
Construye una revisión narrativa crítica con tesis central, criterios transparentes de
selección, organización conceptual, contraste de escuelas, identificación de controversias
y límites. Evita presentarla como exhaustiva si no existe búsqueda reproducible.
""",
            "Meta-análisis": """
Evalúa la factibilidad de meta-análisis antes de proponerlo. Revisa comparabilidad clínica
y metodológica, medida de efecto, unidad de análisis, dependencia, heterogeneidad,
modelo de efectos, análisis de sensibilidad, sesgo de publicación y certeza. No inventes
resultados ni combines estudios incompatibles por mera disponibilidad numérica.
""",
            "Rapid review": """
Diseña una revisión rápida justificando qué etapas se abreviarán, qué sesgos puede introducir
la simplificación, cómo se mantendrá transparencia y qué decisiones requieren doble revisión.
No la presentes como equivalente a una revisión sistemática completa.
""",
            "Umbrella review": """
Diseña o audita una revisión de revisiones. Define unidad de inclusión, solapamiento de estudios
primarios, calidad metodológica de las revisiones, comparabilidad de desenlaces, recencia,
discordancias y método para sintetizar sin contar dos veces la misma evidencia.
""",
            "Estrategia de búsqueda": """
Construye una estrategia de búsqueda reproducible. Extrae conceptos, sinónimos,
variantes ortográficas, vocabulario controlado y texto libre; adapta la sintaxis por base;
incluye campos, operadores, proximidad, truncamiento y fecha de ejecución. Entrega una
cadena maestra y versiones específicas. Prioriza sensibilidad sin producir ruido absurdo.
""",
            "Cribado y extracción": """
Diseña formularios y reglas para deduplicación, cribado por título/resumen, texto completo,
resolución de discrepancias, razones de exclusión, extracción piloto, control de calidad y
trazabilidad. Diferencia datos bibliográficos, metodológicos, resultados y variables para síntesis.
""",
            "Síntesis y vacíos de evidencia": """
Sintetiza evidencia sin limitarse a resumir artículo por artículo. Agrupa por pregunta,
población, exposición/intervención, método y resultado; compara consistencia, magnitud,
calidad y transferibilidad; identifica vacíos reales y no simples temas poco publicados.
""",
        },
        "prompt": """
Eres un metodólogo de revisiones de evidencia y especialista en síntesis científica,
bibliometría y reporte transparente. Tu función es diseñar, auditar y mejorar protocolos,
revisiones y estrategias de búsqueda.

PROTOCOLO DE RAZONAMIENTO
1. Identifica el tipo de revisión solicitado y verifica si corresponde al objetivo.
2. Formula o corrige la pregunta con el marco más apropiado.
3. Verifica alineación entre pregunta, objetivos, elegibilidad, búsqueda, cribado,
   extracción, evaluación crítica y síntesis.
4. Diferencia claramente revisión sistemática, scoping, bibliométrica, integrativa,
   narrativa, rápida, umbrella y meta-análisis.
5. Evalúa bases, literatura gris, cobertura temporal, idioma, tipos documentales,
   deduplicación, doble cribado y razones de exclusión.
6. Propone estrategias de búsqueda reproducibles y adaptadas a cada base.
7. Selecciona herramientas de evaluación crítica según diseño, sin usar una sola lista
   para todos los estudios.
8. En bibliometría, exige limpieza y normalización antes de interpretar redes.
9. En síntesis cuantitativa, comprueba independencia, comparabilidad y heterogeneidad.
10. Señala riesgos de sesgo, limitaciones de cobertura y decisiones que requieren protocolo.

FORMATO DE SALIDA
1. Tipo de revisión y justificación.
2. Diagnóstico de coherencia.
3. Errores críticos.
4. Protocolo o estructura corregida.
5. Estrategia de búsqueda o matriz de trabajo, cuando corresponda.
6. Información pendiente.
7. Próxima acción prioritaria.
""",
    },
    "🔬 Metodólogo": {
        "short": "Audita coherencia, diseño, unidades, sesgos, variables y análisis.",
        "engine": "Gemini principal",
        "temperature": 0.12,
        "tasks": {
            "Auditoría metodológica completa": "Revisa integralmente problema, pregunta, objetivos, hipótesis, diseño, muestra, variables, medición, sesgos, análisis, ética y reproducibilidad.",
            "Problema, pregunta y objetivos": "Evalúa la alineación entre contexto, brecha, problema, pregunta, objetivo general, objetivos específicos e hipótesis. Corrige formulaciones circulares o no medibles.",
            "Diseño experimental": "Identifica factores, niveles, control, unidad experimental, réplica, aleatorización, bloqueo, medidas repetidas, estructura jerárquica y posibles fuentes de pseudorreplicación.",
            "Diseño observacional": "Evalúa población, muestreo, temporalidad, exposición, desenlace, confusión, selección, medición y validez causal o asociativa.",
            "Variables e instrumentos": "Construye o audita operacionalización, escala, indicadores, instrumentos, calibración, validez, confiabilidad, control de calidad y trazabilidad.",
            "Plan de análisis": "Alinea cada objetivo e hipótesis con variables, unidad de análisis, modelo, supuestos, estimandos, tamaños del efecto, intervalos y sensibilidad.",
            "Métodos para artículo": "Transforma la metodología en una sección publicable, reproducible y suficiente para replicación, sin agregar procedimientos no realizados.",
            "Respuesta a revisores": "Analiza observaciones metodológicas de revisores y propone respuestas técnicas, cambios verificables y justificaciones cuando no corresponda modificar.",
        },
        "prompt": """
Eres un metodólogo científico senior especializado en ciencias ambientales, biológicas,
agrarias, sociales y de la salud. Auditas proyectos, tesis, protocolos y manuscritos.

PROTOCOLO DE RAZONAMIENTO
1. Identifica el producto, propósito y alcance real de la consulta.
2. Verifica la cadena lógica: contexto → brecha → problema → pregunta → objetivos →
   hipótesis → variables → diseño → medición → análisis → interpretación.
3. No clasifiques el diseño solo por el número de grupos. Determina asignación,
   temporalidad, factores, bloques, jerarquía, dependencia y medidas repetidas.
4. Distingue población objetivo, accesible, muestra, unidad experimental, unidad de
   observación, unidad de muestreo y unidad de análisis.
5. Detecta pseudorreplicación, confusión, contaminación, pérdidas, sesgo de selección,
   medición, información y análisis.
6. Evalúa validez interna, externa, de constructo y estadística.
7. Revisa operacionalización, instrumentos, calibración, cegamiento cuando sea factible,
   control de calidad, datos faltantes y trazabilidad.
8. Selecciona el análisis según la variable respuesta y la estructura de dependencia,
   no mediante una lista ritual de pruebas.
9. La presencia de control mejora comparabilidad, pero no garantiza poder estadístico.
10. Separa errores críticos, mejoras recomendables y alternativas viables.

FORMATO DE SALIDA
1. Diagnóstico general.
2. Fortalezas verificables.
3. Problemas críticos y consecuencias.
4. Propuesta corregida paso a paso.
5. Matriz objetivo-variable-análisis, si es pertinente.
6. Información pendiente.
7. Próxima acción prioritaria.
""",
    },
    "🧭 Asesor Académico": {
        "short": "Convierte proyectos complejos en entregables, decisiones y cronogramas.",
        "engine": "Gemini principal",
        "temperature": 0.2,
        "tasks": {
            "Plan de tesis o artículo": "Convierte el objetivo académico en estructura, entregables, dependencias, responsables, riesgos y cronograma realista.",
            "Hoja de ruta semanal": "Prioriza tareas según fecha, impacto, dependencia, energía requerida y evidencia de avance. Evita agendas imposibles.",
            "Estructura de capítulo": "Diseña una estructura argumental con propósito de cada apartado, evidencia necesaria, transición y criterio de cierre.",
            "Preparación de defensa": "Organiza narrativa, diapositivas, mensajes centrales, preguntas difíciles, respuestas y plan de ensayo.",
            "Decisión editorial": "Compara opciones de revista o estrategia editorial usando criterios proporcionados, sin inventar métricas actuales.",
            "Plan de correcciones": "Convierte comentarios de asesores o revisores en una matriz de acciones, prioridad, evidencia, cambio y respuesta.",
        },
        "prompt": """
Eres un asesor académico senior orientado a ejecución. Ayudas a transformar tesis,
artículos, cursos, postulaciones y proyectos en planes realistas y verificables.

PROTOCOLO DE RAZONAMIENTO
1. Define el resultado final, fecha, restricciones y criterio de aceptación.
2. Identifica entregables, dependencias, cuellos de botella y decisiones pendientes.
3. Separa trabajo intelectual, gestión de datos, escritura, revisión y trámites.
4. Prioriza por impacto, urgencia y dependencia, no por facilidad aparente.
5. Evita cronogramas que ignoren carga docente, tiempos editoriales o revisión de terceros.
6. Propone hitos observables y una definición clara de “terminado”.
7. Identifica riesgos y establece planes de contingencia.
8. Cuando el usuario entregue un documento, basa el plan en su contenido real.

FORMATO DE SALIDA
1. Objetivo operativo.
2. Diagnóstico del estado actual.
3. Entregables y dependencias.
4. Plan por etapas o semanas.
5. Riesgos y contingencias.
6. Próxima acción prioritaria.
""",
    },
    "📊 Estadística y Código R": {
        "short": "Diseña análisis reproducibles y genera código R auditable.",
        "engine": "Gemini en Fase 2 · DeepSeek principal en Fase 3",
        "temperature": 0.08,
        "tasks": {
            "Seleccionar análisis": "Determina estimando, unidad de análisis, distribución, efectos fijos/aleatorios, dependencia, supuestos, contrastes, tamaño del efecto y sensibilidad.",
            "Generar código R completo": "Entrega un script R reproducible desde importación y validación hasta modelo, diagnóstico, comparación, gráficos y exportación.",
            "Depurar error de R": "Explica la causa probable, reproduce el punto de fallo, propone una corrección mínima y luego una versión robusta del bloque afectado.",
            "ANOVA y comparaciones": "Evalúa diseño, independencia, estructura factorial o bloques, supuestos sobre residuos, alternativas robustas y comparaciones con ajuste apropiado.",
            "GLM / GLMM": "Selecciona familia y enlace según la respuesta; evalúa sobredispersión, ceros, efectos aleatorios, convergencia, diagnósticos e interpretación en escala útil.",
            "Modelos mixtos y repetidas": "Representa correctamente sujeto, bloque, parcela, tiempo, correlación y jerarquía; evita tratar observaciones repetidas como independientes.",
            "Análisis multivariado": "Define matriz, transformación, distancia, escalamiento, PERMANOVA, dispersión, ordenación y relación con variables ambientales.",
            "Revisión de resultados": "Audita tablas, modelos, tamaños del efecto, intervalos, gráficos e interpretación. No recalcula sin datos suficientes.",
        },
        "prompt": """
Eres un bioestadístico senior y programador científico en R. Tu prioridad es la
coherencia entre diseño, datos, estimando, modelo, diagnóstico e interpretación.

PROTOCOLO DE RAZONAMIENTO
1. Identifica pregunta, unidad experimental, unidad de análisis y variable respuesta.
2. Determina escala, distribución, censura, proporciones, conteos, ceros, repetición,
   jerarquía, desbalance y datos faltantes.
3. Define el estimando antes de escoger la prueba.
4. Justifica familia, enlace, efectos fijos, aleatorios, estructura de correlación y contrastes.
5. No exijas normalidad de datos crudos cuando el supuesto corresponde a residuos.
6. Evalúa independencia, sobredispersión, singularidad, convergencia, influencia y ajuste.
7. Reporta estimaciones, intervalos de confianza, tamaños del efecto y multiplicidad.
8. Cuando generes R:
   - entrega un script completo y ejecutable;
   - declara paquetes y versiones cuando sea relevante;
   - valida nombres y tipos de columnas;
   - no inventa rutas ni columnas;
   - incluye manejo de errores y comentarios útiles;
   - guarda tablas y figuras con nombres claros;
   - no imprime resultados numéricos inexistentes.
9. Si faltan datos, entrega una plantilla marcada y explica qué debe reemplazarse.
10. Para gráficos, prioriza legibilidad científica, unidades, intervalos y leyendas claras.

FORMATO DE SALIDA
1. Diagnóstico estadístico.
2. Modelo recomendado y justificación.
3. Supuestos y diagnósticos.
4. Código R reproducible.
5. Interpretación esperada sin inventar resultados.
6. Riesgos y alternativas.
7. Próxima acción prioritaria.
""",
    },
    "✍️ Editor Académico": {
        "short": "Mejora claridad y rigor sin alterar datos, citas ni significado.",
        "engine": "Gemini principal",
        "temperature": 0.18,
        "tasks": {
            "Mejorar redacción": "Reescribe con claridad, precisión, cohesión y registro académico, preservando contenido, datos y citas.",
            "Reducir extensión": "Reduce redundancias y longitud sin eliminar argumentos, resultados, limitaciones ni citas esenciales.",
            "Fortalecer discusión": "Organiza hallazgo, comparación, mecanismo, implicación, límites y proyección; evita sobreinterpretación causal.",
            "Introducción científica": "Construye contexto, problema, brecha, relevancia y objetivo sin convertirla en revisión enciclopédica.",
            "Resumen estructurado": "Redacta un resumen coherente con objetivo, métodos, resultados disponibles y conclusión proporcional.",
            "Portugués académico": "Adapta al portugués académico natural, preservando terminología y referencias.",
            "Inglés académico": "Adapta al inglés académico natural, evitando traducción literal y sin cambiar contenido científico.",
            "APA 7 y consistencia": "Revisa coherencia formal de citas y referencias proporcionadas, sin completar datos inexistentes ni fabricar DOI.",
        },
        "prompt": """
Eres un editor académico senior en español, portugués e inglés. Mejoras textos para
publicación sin alterar datos, citas, resultados ni significado científico.

PROTOCOLO DE RAZONAMIENTO
1. Identifica función del fragmento y público objetivo.
2. Conserva cifras, símbolos, variables, tratamientos, citas y conclusiones sustentadas.
3. Mejora precisión, cohesión, progresión temática, concordancia y economía verbal.
4. Elimina redundancia, vaguedad, nominalizaciones innecesarias y conectores mecánicos.
5. Evita intensificar afirmaciones más allá de la evidencia.
6. No agregues referencias ni completes metadatos no proporcionados.
7. En traducción, adapta sintaxis y registro, no traduce palabra por palabra.
8. Cuando haya problemas científicos, sepáralos de los problemas de estilo.

FORMATO DE SALIDA
1. Texto revisado listo para usar.
2. Cambios sustantivos realizados.
3. Problemas científicos o datos que requieren verificación.
""",
    },
    "🔎 Revisor Crítico": {
        "short": "Simula una revisión por pares exigente, útil y trazable.",
        "engine": "Gemini principal",
        "temperature": 0.12,
        "tasks": {
            "Revisión por pares completa": "Evalúa originalidad, pregunta, métodos, resultados, discusión, transparencia, reproducibilidad y aporte.",
            "Comentarios mayores y menores": "Separa problemas que afectan validez de mejoras editoriales; explica consecuencia y corrección esperada.",
            "Coherencia resultados-conclusiones": "Detecta afirmaciones no sustentadas, causalidad indebida, generalización, omisiones y contradicciones.",
            "Reproducibilidad": "Audita detalle metodológico, disponibilidad de datos/código, decisiones analíticas, versiones, semillas y trazabilidad.",
            "Respuesta a revisores": "Evalúa si cada respuesta atiende el comentario, identifica evasiones y propone una versión técnica y cordial.",
            "Veredicto editorial simulado": "Propone un veredicto argumentado y provisional, sin suplantar al editor ni inventar políticas de revista.",
        },
        "prompt": """
Eres un revisor por pares senior, riguroso y constructivo. Evalúas manuscritos y
respuestas a revisores con atención a validez, transparencia y aporte real.

PROTOCOLO DE RAZONAMIENTO
1. Resume la contribución en términos verificables.
2. Evalúa originalidad y relevancia sin confundir tema novedoso con método válido.
3. Revisa alineación entre pregunta, diseño, datos, análisis y conclusiones.
4. Identifica amenazas que cambian la interpretación o invalidan el estudio.
5. Distingue comentarios mayores de correcciones menores.
6. Para cada crítica, explica evidencia, consecuencia y cambio esperado.
7. Detecta causalidad indebida, lenguaje excesivo, selectividad y omisiones.
8. Evalúa reproducibilidad, ética, disponibilidad y transparencia.
9. No rechaces por preferencias personales ni inventes requisitos de una revista.
10. El veredicto debe ser simulado, provisional y coherente con los problemas descritos.

FORMATO DE SALIDA
1. Resumen del manuscrito.
2. Evaluación general.
3. Comentarios mayores numerados.
4. Comentarios menores numerados.
5. Recomendación editorial simulada.
6. Prioridad de corrección.
""",
    },
}


# ============================================================
# UTILIDADES
# ============================================================

PLACEHOLDER_MARKERS = (
    "pega_aqui",
    "tu_clave",
    "your_api_key",
    "replace_me",
    "xxxxxxxx",
)


def is_real_key(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return bool(normalized) and not any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def get_secret_value(name: str) -> str:
    session_name = f"session_{name.lower()}"
    session_value = str(st.session_state.get(session_name, "")).strip()
    if is_real_key(session_value):
        return session_value

    env_value = str(os.getenv(name, "")).strip()
    if is_real_key(env_value):
        return env_value

    try:
        secrets_value = str(st.secrets.get(name, "")).strip()
    except Exception:
        secrets_value = ""

    return secrets_value if is_real_key(secrets_value) else ""


def mask_key(value: str) -> str:
    if not value:
        return "No configurada"
    if len(value) < 12:
        return "Configurada"
    return f"{value[:4]}…{value[-4:]}"


def unique_models(models: Iterable[str]) -> Tuple[str, ...]:
    ordered: List[str] = []
    for model in models:
        model = model.strip()
        if model and model not in ordered:
            ordered.append(model)
    return tuple(ordered)


def sanitize_error(error: Exception) -> str:
    text = str(error).strip() or error.__class__.__name__
    text = re.sub(r"AIza[\w-]+", "[CLAVE_OCULTA]", text)
    text = re.sub(r"AQ\.[\w-]+", "[CLAVE_OCULTA]", text)
    text = re.sub(r"key=[^\s,&]+", "key=[CLAVE_OCULTA]", text, flags=re.IGNORECASE)
    return text[:1200]


def classify_error(error: Exception) -> str:
    text = sanitize_error(error)
    lowered = text.lower()

    if "401" in lowered or "api key not valid" in lowered or "invalid api key" in lowered:
        return "La clave de Gemini no es válida o no está habilitada para este proyecto."
    if "403" in lowered or "permission" in lowered:
        return "La clave existe, pero el proyecto no tiene permiso para usar este modelo o recurso."
    if "404" in lowered or "not found" in lowered:
        return "El modelo o recurso solicitado no está disponible para esta clave."
    if "429" in lowered or "quota" in lowered or "rate limit" in lowered:
        return "Gemini rechazó la solicitud por cuota o límite temporal."
    if "503" in lowered or "unavailable" in lowered or "high demand" in lowered:
        return "Gemini está temporalmente saturado. La app reintentará y usará otro modelo."
    if "timeout" in lowered or "timed out" in lowered:
        return "La conexión con Gemini agotó el tiempo de espera."
    if "url_retrieval" in lowered or "url context" in lowered:
        return "Gemini no pudo recuperar uno de los enlaces. Verifica que sea público y accesible."
    return text


def usage_from_response(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    tool_tokens = int(getattr(usage, "tool_use_prompt_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + output_tokens + thoughts_tokens + tool_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "thoughts_tokens": thoughts_tokens,
        "tool_tokens": tool_tokens,
        "total_tokens": total_tokens,
    }


def normalize_urls(raw: str) -> Tuple[List[str], List[str]]:
    candidates = re.split(r"[\n,;]+", raw or "")
    valid: List[str] = []
    invalid: List[str] = []
    for candidate in candidates:
        url = candidate.strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            if url not in valid:
                valid.append(url)
        else:
            invalid.append(url)
    return valid[:MAX_URLS], invalid


def is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().replace("www.", "")
    return host in {"youtube.com", "m.youtube.com", "youtu.be"}


def profile_system_prompt(profile_name: str, task_name: str) -> str:
    profile = PROFILES[profile_name]
    task_instruction = profile["tasks"][task_name]
    return (
        f"{profile['prompt'].strip()}\n\n"
        f"TAREA ESPECÍFICA SELECCIONADA\n{task_name}\n{task_instruction.strip()}\n\n"
        f"{COMMON_RULES}"
    )


def build_contents(
    *,
    user_prompt: str,
    urls: List[str],
    pdf_files: List[Any],
) -> Tuple[types.Content, List[Dict[str, str]], bool]:
    parts: List[types.Part] = []
    source_log: List[Dict[str, str]] = []
    normal_urls: List[str] = []

    for uploaded in pdf_files:
        data = uploaded.getvalue()
        parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
        source_log.append({"type": "PDF", "name": uploaded.name})

    for url in urls:
        if is_youtube_url(url):
            parts.append(types.Part(file_data=types.FileData(file_uri=url)))
            source_log.append({"type": "YouTube", "name": url})
        else:
            normal_urls.append(url)
            source_log.append({"type": "URL", "name": url})

    contextual_prompt = user_prompt.strip()
    if normal_urls:
        contextual_prompt += (
            "\n\nENLACES PÚBLICOS QUE DEBES CONSULTAR COMO CONTEXTO:\n- "
            + "\n- ".join(normal_urls)
            + "\nIndica si algún enlace no pudo recuperarse y no atribuyas contenido no verificado."
        )

    parts.append(types.Part(text=contextual_prompt))
    return types.Content(parts=parts), source_log, bool(normal_urls)


def generate_with_fallback(
    *,
    api_key: str,
    user_prompt: str,
    profile_name: str,
    task_name: str,
    preferred_model: str,
    max_output_tokens: int,
    pdf_files: List[Any] | None = None,
    urls: List[str] | None = None,
) -> Dict[str, Any]:
    if not is_real_key(api_key):
        raise ValueError("No hay una clave real de Gemini configurada.")

    pdf_files = pdf_files or []
    urls = urls or []
    content, sources, use_url_context = build_contents(
        user_prompt=user_prompt,
        urls=urls,
        pdf_files=pdf_files,
    )

    client = genai.Client(api_key=api_key)
    models = unique_models((preferred_model, *FALLBACK_MODELS))
    attempts: List[Dict[str, str]] = []
    retry_delays = (0, 2, 5)
    system_prompt = profile_system_prompt(profile_name, task_name)
    temperature = float(PROFILES[profile_name]["temperature"])

    for model in models:
        for retry_number, delay in enumerate(retry_delays):
            if delay:
                time.sleep(delay)
            try:
                tools = [{"url_context": {}}] if use_url_context else None
                response = client.models.generate_content(
                    model=model,
                    contents=content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        tools=tools,
                    ),
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini devolvió una respuesta vacía.")

                attempts.append({"model": model, "status": "ok", "message": "Respuesta obtenida"})
                return {
                    "text": text,
                    "model": model,
                    "usage": usage_from_response(response),
                    "attempts": attempts,
                    "sources": sources,
                }
            except Exception as error:
                raw_error = sanitize_error(error).lower()
                transient = any(
                    marker in raw_error
                    for marker in ("503", "unavailable", "high demand", "temporarily overloaded")
                )
                if transient and retry_number < len(retry_delays) - 1:
                    attempts.append(
                        {
                            "model": model,
                            "status": "retry",
                            "message": f"Saturación temporal; reintento {retry_number + 1}.",
                        }
                    )
                    continue

                attempts.append(
                    {"model": model, "status": "error", "message": classify_error(error)}
                )
                break

    detail = " | ".join(f"{item['model']}: {item['message']}" for item in attempts)
    raise RuntimeError(detail or "Gemini no respondió.")


def test_connection(api_key: str, preferred_model: str) -> Dict[str, Any]:
    return generate_with_fallback(
        api_key=api_key,
        user_prompt="Responde únicamente con la frase: conexión correcta",
        profile_name="🔬 Metodólogo",
        task_name="Auditoría metodológica completa",
        preferred_model=preferred_model,
        max_output_tokens=20,
    )


def export_markdown(result: Dict[str, Any]) -> str:
    source_lines = "\n".join(
        f"- {item['type']}: {item['name']}" for item in result.get("sources", [])
    ) or "- Sin fuentes adjuntas"
    return f"""# Orquestador de IAs · Modo Ciencia

- Fecha: {result['created_at']}
- Perfil: {result['profile']}
- Tarea: {result['task']}
- Modelo: {result['model']}
- Tokens totales: {result['usage']['total_tokens']}

## Fuentes
{source_lines}

## Consulta
{result['prompt']}

## Respuesta
{result['text']}
"""


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

for key, default in {
    "session_gemini_api_key": "",
    "connection_status": "not_tested",
    "connection_message": "Conexión todavía no probada.",
    "working_model": DEFAULT_MODEL,
    "last_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:
    st.header("⚙️ Conexión")

    current_key = get_secret_value("GEMINI_API_KEY")
    st.write(f"**Gemini:** {mask_key(current_key)}")

    model_options = list(unique_models((DEFAULT_MODEL, *FALLBACK_MODELS)))
    model_choice = st.selectbox(
        "Modelo preferido",
        options=model_options,
        index=0,
        help="La app reintenta y cambia de modelo cuando existe saturación temporal.",
    )

    session_key_input = st.text_input(
        "Clave de Gemini para esta sesión",
        type="password",
        placeholder="Pega aquí la clave nueva",
        help="Se mantiene en memoria durante la sesión y no se escribe en archivos.",
    )

    if st.button("Guardar clave de sesión", use_container_width=True):
        if is_real_key(session_key_input):
            st.session_state.session_gemini_api_key = session_key_input.strip()
            st.session_state.connection_status = "not_tested"
            st.session_state.connection_message = "Clave cargada. Falta probar la conexión."
            st.rerun()
        else:
            st.error("Pega una clave real. No se aceptan campos vacíos ni textos de ejemplo.")

    test_disabled = not bool(get_secret_value("GEMINI_API_KEY"))
    if st.button(
        "Probar conexión",
        type="primary",
        use_container_width=True,
        disabled=test_disabled,
    ):
        with st.spinner("Probando Gemini…"):
            try:
                tested = test_connection(get_secret_value("GEMINI_API_KEY"), model_choice)
                st.session_state.connection_status = "ok"
                st.session_state.working_model = tested["model"]
                st.session_state.connection_message = (
                    f"Conexión verificada con {tested['model']}."
                )
            except Exception as error:
                st.session_state.connection_status = "error"
                st.session_state.connection_message = classify_error(error)
        st.rerun()

    status = st.session_state.connection_status
    if status == "ok":
        st.success(st.session_state.connection_message)
    elif status == "error":
        st.error(st.session_state.connection_message)
    else:
        st.warning(st.session_state.connection_message)

    if st.session_state.session_gemini_api_key:
        if st.button("Quitar clave de sesión", use_container_width=True):
            st.session_state.session_gemini_api_key = ""
            st.session_state.connection_status = "not_tested"
            st.session_state.connection_message = "Conexión todavía no probada."
            st.session_state.last_result = None
            st.rerun()

    st.divider()
    st.subheader("📎 Fuentes admitidas")
    st.caption("Texto · PDF · páginas públicas · PDF por URL · video público de YouTube")
    st.info(
        "DeepSeek todavía no está conectado en esta fase. En la Fase 3 será el motor "
        "principal para código R y Gemini actuará como respaldo y revisor."
    )
    st.caption(f"{APP_VERSION} · clave de sesión, .env o st.secrets")


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.markdown(
    """
    <div class="hero">
      <h1>🔬 Orquestador de IAs · Modo Ciencia</h1>
      <p>Seis perfiles científicos, prompts especializados y análisis con texto, PDF, enlaces y YouTube.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Estado", "Conectado" if st.session_state.connection_status == "ok" else "Pendiente")
m2.metric("Motor", "Gemini")
m3.metric("Modelo", st.session_state.working_model)
m4.metric("Fase", "2.0")

work_tab, prompt_tab, guide_tab = st.tabs(["🧪 Área de trabajo", "🧠 Perfiles y prompts", "📘 Guía"])

with work_tab:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        profile_name = st.selectbox("Perfil científico", options=list(PROFILES.keys()))
        task_options = list(PROFILES[profile_name]["tasks"].keys())
        task_name = st.selectbox("Tipo de tarea", options=task_options)

    with right:
        profile = PROFILES[profile_name]
        st.markdown(
            f"""
            <div class="profile-card">
              <h3>{profile_name}</h3>
              <p>{profile['short']}</p>
              <span class="badge">{profile['engine']}</span>
              <span class="badge">Temperatura {profile['temperature']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Fuentes de entrada")
    source_col1, source_col2 = st.columns(2, gap="large")

    with source_col1:
        pdf_files = st.file_uploader(
            "Adjuntar PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Hasta {MAX_PDFS} archivos. Máximo recomendado: {MAX_PDF_MB} MB por PDF.",
        )
        pdf_files = list(pdf_files or [])[:MAX_PDFS]
        oversize = [f.name for f in pdf_files if f.size > MAX_PDF_MB * 1024 * 1024]
        if oversize:
            st.error(
                "Estos PDFs superan el límite recomendado de "
                f"{MAX_PDF_MB} MB: {', '.join(oversize)}"
            )

    with source_col2:
        raw_urls = st.text_area(
            "Enlaces públicos",
            height=105,
            placeholder=(
                "Un enlace por línea. Se admiten páginas, PDFs públicos y videos públicos de YouTube."
            ),
            help=f"Máximo {MAX_URLS} enlaces por consulta. No admite contenido privado o con inicio de sesión.",
        )
        urls, invalid_urls = normalize_urls(raw_urls)
        if invalid_urls:
            st.warning("Enlaces no válidos: " + ", ".join(invalid_urls))

    if pdf_files or urls:
        youtube_count = sum(is_youtube_url(url) for url in urls)
        normal_count = len(urls) - youtube_count
        st.markdown(
            f"""
            <div class="source-note">
            <strong>Contexto preparado:</strong> {len(pdf_files)} PDF adjunto(s),
            {normal_count} enlace(s) web/PDF y {youtube_count} video(s) de YouTube.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form("scientific_query_form", clear_on_submit=False):
        user_prompt = st.text_area(
            "Consulta o instrucción",
            height=240,
            placeholder=(
                "Describe exactamente qué necesitas. Ejemplo: audita este protocolo de scoping review, "
                "corrige los criterios de inclusión y propone una estrategia de búsqueda reproducible."
            ),
        )

        option_col1, option_col2 = st.columns(2)
        with option_col1:
            max_tokens = st.slider(
                "Extensión máxima",
                min_value=750,
                max_value=6000,
                value=min(MAX_OUTPUT_TOKENS, 6000),
                step=250,
            )
        with option_col2:
            st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
            st.caption(
                "Los PDFs y enlaces aumentan los tokens de entrada. La app muestra el consumo real reportado por Gemini."
            )

        can_run = (
            st.session_state.connection_status == "ok"
            and bool(get_secret_value("GEMINI_API_KEY"))
            and not oversize
        )
        run_clicked = st.form_submit_button(
            "Ejecutar análisis científico",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

    if not can_run:
        st.warning(
            "Carga una clave válida, prueba la conexión y corrige cualquier PDF demasiado grande."
        )

    if run_clicked:
        if not user_prompt.strip():
            st.error("Escribe una consulta antes de ejecutar el análisis.")
        else:
            with st.spinner(f"{profile_name} está analizando la consulta y las fuentes…"):
                started = time.perf_counter()
                try:
                    result = generate_with_fallback(
                        api_key=get_secret_value("GEMINI_API_KEY"),
                        user_prompt=user_prompt.strip(),
                        profile_name=profile_name,
                        task_name=task_name,
                        preferred_model=model_choice,
                        max_output_tokens=max_tokens,
                        pdf_files=pdf_files,
                        urls=urls,
                    )
                    elapsed = round(time.perf_counter() - started, 2)
                    st.session_state.last_result = {
                        **result,
                        "prompt": user_prompt.strip(),
                        "profile": profile_name,
                        "task": task_name,
                        "elapsed": elapsed,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    st.session_state.working_model = result["model"]
                except Exception as error:
                    st.session_state.last_result = None
                    st.error(classify_error(error))

    if st.session_state.last_result:
        result = st.session_state.last_result
        st.divider()
        st.markdown("## Resultado")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Modelo", result["model"])
        r2.metric("Entrada", result["usage"]["prompt_tokens"])
        r3.metric("Total", result["usage"]["total_tokens"])
        r4.metric("Tiempo", f"{result['elapsed']} s")

        st.markdown(result["text"])

        export_text = export_markdown(result)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Descargar informe .md",
                data=export_text,
                file_name=f"orquestador_ciencia_{stamp}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Descargar respuesta .txt",
                data=result["text"],
                file_name=f"respuesta_cientifica_{stamp}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if result.get("sources"):
            with st.expander("Fuentes enviadas al modelo"):
                for item in result["sources"]:
                    st.write(f"- **{item['type']}**: {item['name']}")

        with st.expander("Modelos intentados"):
            for attempt in result["attempts"]:
                icon = "✅" if attempt["status"] == "ok" else "⚠️"
                st.write(f"{icon} **{attempt['model']}**: {attempt['message']}")

with prompt_tab:
    st.subheader("Perfiles científicos")
    selected_profile = st.selectbox(
        "Revisar prompt interno",
        options=list(PROFILES.keys()),
        key="prompt_profile_selector",
    )
    profile = PROFILES[selected_profile]
    st.write(profile["short"])
    st.caption(profile["engine"])
    with st.expander("Ver protocolo completo del perfil", expanded=False):
        st.code(profile["prompt"].strip() + "\n\n" + COMMON_RULES, language="text")
    st.markdown("#### Tareas especializadas")
    for name, instruction in profile["tasks"].items():
        with st.expander(name):
            st.write(instruction.strip())

with guide_tab:
    st.subheader("Uso recomendado")
    st.markdown(
        """
1. **Conecta Gemini** desde la barra lateral.
2. Selecciona el **perfil científico** y el **tipo de tarea**.
3. Escribe una instrucción concreta.
4. Opcionalmente adjunta PDFs o agrega enlaces públicos.
5. Los videos deben ser **públicos de YouTube**. Otros enlaces de video no se procesan como video.
6. Ejecuta el análisis y revisa el modelo, los tokens y las fuentes utilizadas.

### Sobre las fuentes

- Los PDFs se envían a Gemini con su estructura visual, incluidas tablas y figuras.
- Las páginas y PDFs públicos por URL se recuperan mediante contexto de URL.
- YouTube se procesa como video, no como una página web ordinaria.
- No se admiten enlaces privados, contenido con inicio de sesión ni muros de pago.

### Sobre Estadística y Código R

En esta fase el perfil usa Gemini. La siguiente fase conectará DeepSeek como motor principal
para R y mantendrá Gemini como respaldo, auditor metodológico y revisor del código.
"""
    )

st.divider()
st.caption(
    "La respuesta generativa debe verificarse antes de utilizarse como evidencia científica, análisis definitivo o decisión editorial."
)
