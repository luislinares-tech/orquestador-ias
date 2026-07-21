
from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from pypdf import PdfReader


# ============================================================
# CONFIGURACIÓN
# ============================================================

load_dotenv()

APP_NAME = "Orquestador Científico de IA"
APP_VERSION = "Fase 3.0"
MAX_URLS = 10
MAX_PDFS = 5
MAX_PDF_MB = 20
MAX_EXTRACTED_CHARS = 140_000
DEFAULT_MAX_OUTPUT = int(os.getenv("MAX_OUTPUT_TOKENS", "3500"))

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "Gemini": {
        "env": "GEMINI_API_KEY",
        "label": "Gemini",
        "icon": "◆",
        "default_model": os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        "fallbacks": ["gemini-3.1-flash-lite", "gemini-3.5-flash"],
        "base_url": "",
        "specialty": "Búsqueda, PDF, enlaces, YouTube y metodología",
    },
    "DeepSeek": {
        "env": "DEEPSEEK_API_KEY",
        "label": "DeepSeek",
        "icon": "⌘",
        "default_model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "fallbacks": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "base_url": "https://api.deepseek.com",
        "specialty": "Código R, Python, estadística y depuración",
    },
    "Kimi": {
        "env": "MOONSHOT_API_KEY",
        "label": "Kimi",
        "icon": "◐",
        "default_model": os.getenv("KIMI_MODEL", "kimi-k2.6"),
        "fallbacks": ["kimi-k2.6", "kimi-k2.5"],
        "base_url": "https://api.moonshot.ai/v1",
        "specialty": "Contexto largo, síntesis y segunda lectura",
    },
    "OpenAI": {
        "env": "OPENAI_API_KEY",
        "label": "OpenAI",
        "icon": "✦",
        "default_model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "fallbacks": ["gpt-5-mini", "gpt-5-nano"],
        "base_url": "https://api.openai.com/v1",
        "specialty": "Auditoría crítica e integración opcional",
    },
}

KEY_ALIASES = {
    "GEMINI": "GEMINI_API_KEY",
    "GEMINI_API_KEY": "GEMINI_API_KEY",
    "DEEPSEEK": "DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
    "KIMI": "MOONSHOT_API_KEY",
    "KIMI_API_KEY": "MOONSHOT_API_KEY",
    "MOONSHOT": "MOONSHOT_API_KEY",
    "MOONSHOT_API_KEY": "MOONSHOT_API_KEY",
    "OPENAI": "OPENAI_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
}

PLACEHOLDER_MARKERS = (
    "pega_aqui",
    "tu_clave",
    "your_api_key",
    "replace_me",
    "xxxxxxxx",
    "sk-xxxx",
)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# REGLAS Y PERFILES
# ============================================================

COMMON_RULES = """
REGLAS TRANSVERSALES OBLIGATORIAS

1. Distingue información proporcionada, hechos corroborados, inferencias, supuestos y datos faltantes.
2. No inventes referencias, DOI, PMID, autores, revistas, volúmenes, páginas, resultados, normas ni decisiones editoriales.
3. Cuando una afirmación requiera evidencia externa, usa solo fuentes efectivamente consultadas y verificables.
4. Si una referencia no puede comprobarse, no la completes por aproximación: márcala como no verificada o exclúyela.
5. Conserva exactamente nombres, cifras, tratamientos, variables y citas aportadas por el usuario, salvo instrucción contraria.
6. No elijas un método estadístico por costumbre. Vincula pregunta, unidad experimental, variable, dependencia, distribución y estimando.
7. Para código R, entrega un script reproducible, comentado, con validaciones, diagnóstico, exportación y manejo de errores.
8. Para revisiones científicas, diferencia protocolo, búsqueda, cribado, extracción, síntesis, bibliometría y metaanálisis.
9. No confundas existencia de una referencia con calidad de la revista. Scopus, Web of Science y SJR requieren verificación separada.
10. Cuando uses fuentes externas, termina con un apartado titulado “Referencias verificadas”. No agregues fuentes no consultadas.
11. Si no se utilizaron fuentes externas, declara: “Referencias: no se utilizaron fuentes externas para esta respuesta”.
12. Responde en el idioma principal del usuario y finaliza con una acción prioritaria y ejecutable.
""".strip()

PROFILES = {'📚 Revisor de Literatura': {'short': 'Diseña, audita y mejora revisiones científicas sin fabricar referencias.',
                             'engine': 'Gemini principal',
                             'temperature': 0.15,
                             'tasks': {'Protocolo PRISMA-P / registro': '\n'
                                                                        'Construye o audita un protocolo de revisión. '
                                                                        'Determina primero el tipo de revisión y su\n'
                                                                        'compatibilidad con PRISMA-P, PRISMA 2020, '
                                                                        'PRISMA-ScR, JBI u otra guía pertinente.\n'
                                                                        'Incluye: título, justificación, pregunta, '
                                                                        'objetivos, criterios de elegibilidad, '
                                                                        'fuentes,\n'
                                                                        'estrategia de búsqueda reproducible, gestión '
                                                                        'de registros y duplicados, cribado por '
                                                                        'pares,\n'
                                                                        'extracción, evaluación crítica, síntesis, '
                                                                        'manejo de discrepancias, enmiendas, ética,\n'
                                                                        'registro y plan de difusión. No asumas '
                                                                        'elegibilidad para PROSPERO: evalúala y '
                                                                        'advierte\n'
                                                                        'cuando el registro adecuado sea OSF u otro '
                                                                        'repositorio.\n',
                                       'Revisión sistemática': '\n'
                                                               'Diseña o audita una revisión sistemática. Formula la '
                                                               'pregunta con el marco adecuado\n'
                                                               '(PICO, PECO, PEO, SPIDER u otro), define elegibilidad, '
                                                               'estrategia multibase, proceso\n'
                                                               'PRISMA 2020, evaluación del riesgo de sesgo, '
                                                               'extracción, síntesis narrativa o cuantitativa,\n'
                                                               'certeza de la evidencia y trazabilidad de '
                                                               'exclusiones.\n',
                                       'Scoping review': '\n'
                                                         'Diseña o audita una scoping review conforme al propósito de '
                                                         'mapear conceptos, evidencia,\n'
                                                         'actores, métodos y vacíos. Usa PCC cuando corresponda, '
                                                         'delimita alcance, fuentes grises si\n'
                                                         'son pertinentes, proceso de selección, charting de datos y '
                                                         'reporte compatible con PRISMA-ScR.\n'
                                                         'No conviertas una scoping review en una revisión sistemática '
                                                         'de efectividad disfrazada.\n',
                                       'Revisión bibliométrica': '\n'
                                                                 'Diseña o audita una revisión bibliométrica '
                                                                 'reproducible. Define bases, periodo, tipos\n'
                                                                 'documentales, deduplicación, normalización de '
                                                                 'autores y afiliaciones, indicadores de\n'
                                                                 'producción, impacto y colaboración, coautoría, '
                                                                 'cocitación, acoplamiento bibliográfico,\n'
                                                                 'coocurrencia de términos, sensibilidad a la base y '
                                                                 'software. Separa bibliometría de\n'
                                                                 'síntesis temática y evita inferir calidad científica '
                                                                 'solo por citas.\n',
                                       'Revisión integrativa': '\n'
                                                               'Estructura una revisión integrativa con pregunta '
                                                               'explícita, búsqueda amplia, evaluación\n'
                                                               'metodológica compatible con diseños heterogéneos, '
                                                               'extracción estandarizada, reducción,\n'
                                                               'comparación y síntesis de datos. Justifica por qué es '
                                                               'integrativa y no narrativa o sistemática.\n',
                                       'Revisión narrativa crítica': '\n'
                                                                     'Construye una revisión narrativa crítica con '
                                                                     'tesis central, criterios transparentes de\n'
                                                                     'selección, organización conceptual, contraste de '
                                                                     'escuelas, identificación de controversias\n'
                                                                     'y límites. Evita presentarla como exhaustiva si '
                                                                     'no existe búsqueda reproducible.\n',
                                       'Meta-análisis': '\n'
                                                        'Evalúa la factibilidad de meta-análisis antes de proponerlo. '
                                                        'Revisa comparabilidad clínica\n'
                                                        'y metodológica, medida de efecto, unidad de análisis, '
                                                        'dependencia, heterogeneidad,\n'
                                                        'modelo de efectos, análisis de sensibilidad, sesgo de '
                                                        'publicación y certeza. No inventes\n'
                                                        'resultados ni combines estudios incompatibles por mera '
                                                        'disponibilidad numérica.\n',
                                       'Rapid review': '\n'
                                                       'Diseña una revisión rápida justificando qué etapas se '
                                                       'abreviarán, qué sesgos puede introducir\n'
                                                       'la simplificación, cómo se mantendrá transparencia y qué '
                                                       'decisiones requieren doble revisión.\n'
                                                       'No la presentes como equivalente a una revisión sistemática '
                                                       'completa.\n',
                                       'Umbrella review': '\n'
                                                          'Diseña o audita una revisión de revisiones. Define unidad '
                                                          'de inclusión, solapamiento de estudios\n'
                                                          'primarios, calidad metodológica de las revisiones, '
                                                          'comparabilidad de desenlaces, recencia,\n'
                                                          'discordancias y método para sintetizar sin contar dos veces '
                                                          'la misma evidencia.\n',
                                       'Estrategia de búsqueda': '\n'
                                                                 'Construye una estrategia de búsqueda reproducible. '
                                                                 'Extrae conceptos, sinónimos,\n'
                                                                 'variantes ortográficas, vocabulario controlado y '
                                                                 'texto libre; adapta la sintaxis por base;\n'
                                                                 'incluye campos, operadores, proximidad, truncamiento '
                                                                 'y fecha de ejecución. Entrega una\n'
                                                                 'cadena maestra y versiones específicas. Prioriza '
                                                                 'sensibilidad sin producir ruido absurdo.\n',
                                       'Cribado y extracción': '\n'
                                                               'Diseña formularios y reglas para deduplicación, '
                                                               'cribado por título/resumen, texto completo,\n'
                                                               'resolución de discrepancias, razones de exclusión, '
                                                               'extracción piloto, control de calidad y\n'
                                                               'trazabilidad. Diferencia datos bibliográficos, '
                                                               'metodológicos, resultados y variables para síntesis.\n',
                                       'Síntesis y vacíos de evidencia': '\n'
                                                                         'Sintetiza evidencia sin limitarse a resumir '
                                                                         'artículo por artículo. Agrupa por pregunta,\n'
                                                                         'población, exposición/intervención, método y '
                                                                         'resultado; compara consistencia, magnitud,\n'
                                                                         'calidad y transferibilidad; identifica '
                                                                         'vacíos reales y no simples temas poco '
                                                                         'publicados.\n'},
                             'prompt': '\n'
                                       'Eres un metodólogo de revisiones de evidencia y especialista en síntesis '
                                       'científica,\n'
                                       'bibliometría y reporte transparente. Tu función es diseñar, auditar y mejorar '
                                       'protocolos,\n'
                                       'revisiones y estrategias de búsqueda.\n'
                                       '\n'
                                       'PROTOCOLO DE RAZONAMIENTO\n'
                                       '1. Identifica el tipo de revisión solicitado y verifica si corresponde al '
                                       'objetivo.\n'
                                       '2. Formula o corrige la pregunta con el marco más apropiado.\n'
                                       '3. Verifica alineación entre pregunta, objetivos, elegibilidad, búsqueda, '
                                       'cribado,\n'
                                       '   extracción, evaluación crítica y síntesis.\n'
                                       '4. Diferencia claramente revisión sistemática, scoping, bibliométrica, '
                                       'integrativa,\n'
                                       '   narrativa, rápida, umbrella y meta-análisis.\n'
                                       '5. Evalúa bases, literatura gris, cobertura temporal, idioma, tipos '
                                       'documentales,\n'
                                       '   deduplicación, doble cribado y razones de exclusión.\n'
                                       '6. Propone estrategias de búsqueda reproducibles y adaptadas a cada base.\n'
                                       '7. Selecciona herramientas de evaluación crítica según diseño, sin usar una '
                                       'sola lista\n'
                                       '   para todos los estudios.\n'
                                       '8. En bibliometría, exige limpieza y normalización antes de interpretar '
                                       'redes.\n'
                                       '9. En síntesis cuantitativa, comprueba independencia, comparabilidad y '
                                       'heterogeneidad.\n'
                                       '10. Señala riesgos de sesgo, limitaciones de cobertura y decisiones que '
                                       'requieren protocolo.\n'
                                       '\n'
                                       'FORMATO DE SALIDA\n'
                                       '1. Tipo de revisión y justificación.\n'
                                       '2. Diagnóstico de coherencia.\n'
                                       '3. Errores críticos.\n'
                                       '4. Protocolo o estructura corregida.\n'
                                       '5. Estrategia de búsqueda o matriz de trabajo, cuando corresponda.\n'
                                       '6. Información pendiente.\n'
                                       '7. Próxima acción prioritaria.\n'},
 '🔬 Metodólogo': {'short': 'Audita coherencia, diseño, unidades, sesgos, variables y análisis.',
                  'engine': 'Gemini principal',
                  'temperature': 0.12,
                  'tasks': {'Auditoría metodológica completa': 'Revisa integralmente problema, pregunta, objetivos, '
                                                               'hipótesis, diseño, muestra, variables, medición, '
                                                               'sesgos, análisis, ética y reproducibilidad.',
                            'Problema, pregunta y objetivos': 'Evalúa la alineación entre contexto, brecha, problema, '
                                                              'pregunta, objetivo general, objetivos específicos e '
                                                              'hipótesis. Corrige formulaciones circulares o no '
                                                              'medibles.',
                            'Diseño experimental': 'Identifica factores, niveles, control, unidad experimental, '
                                                   'réplica, aleatorización, bloqueo, medidas repetidas, estructura '
                                                   'jerárquica y posibles fuentes de pseudorreplicación.',
                            'Diseño observacional': 'Evalúa población, muestreo, temporalidad, exposición, desenlace, '
                                                    'confusión, selección, medición y validez causal o asociativa.',
                            'Variables e instrumentos': 'Construye o audita operacionalización, escala, indicadores, '
                                                        'instrumentos, calibración, validez, confiabilidad, control de '
                                                        'calidad y trazabilidad.',
                            'Plan de análisis': 'Alinea cada objetivo e hipótesis con variables, unidad de análisis, '
                                                'modelo, supuestos, estimandos, tamaños del efecto, intervalos y '
                                                'sensibilidad.',
                            'Métodos para artículo': 'Transforma la metodología en una sección publicable, '
                                                     'reproducible y suficiente para replicación, sin agregar '
                                                     'procedimientos no realizados.',
                            'Respuesta a revisores': 'Analiza observaciones metodológicas de revisores y propone '
                                                     'respuestas técnicas, cambios verificables y justificaciones '
                                                     'cuando no corresponda modificar.'},
                  'prompt': '\n'
                            'Eres un metodólogo científico senior especializado en ciencias ambientales, biológicas,\n'
                            'agrarias, sociales y de la salud. Auditas proyectos, tesis, protocolos y manuscritos.\n'
                            '\n'
                            'PROTOCOLO DE RAZONAMIENTO\n'
                            '1. Identifica el producto, propósito y alcance real de la consulta.\n'
                            '2. Verifica la cadena lógica: contexto → brecha → problema → pregunta → objetivos →\n'
                            '   hipótesis → variables → diseño → medición → análisis → interpretación.\n'
                            '3. No clasifiques el diseño solo por el número de grupos. Determina asignación,\n'
                            '   temporalidad, factores, bloques, jerarquía, dependencia y medidas repetidas.\n'
                            '4. Distingue población objetivo, accesible, muestra, unidad experimental, unidad de\n'
                            '   observación, unidad de muestreo y unidad de análisis.\n'
                            '5. Detecta pseudorreplicación, confusión, contaminación, pérdidas, sesgo de selección,\n'
                            '   medición, información y análisis.\n'
                            '6. Evalúa validez interna, externa, de constructo y estadística.\n'
                            '7. Revisa operacionalización, instrumentos, calibración, cegamiento cuando sea factible,\n'
                            '   control de calidad, datos faltantes y trazabilidad.\n'
                            '8. Selecciona el análisis según la variable respuesta y la estructura de dependencia,\n'
                            '   no mediante una lista ritual de pruebas.\n'
                            '9. La presencia de control mejora comparabilidad, pero no garantiza poder estadístico.\n'
                            '10. Separa errores críticos, mejoras recomendables y alternativas viables.\n'
                            '\n'
                            'FORMATO DE SALIDA\n'
                            '1. Diagnóstico general.\n'
                            '2. Fortalezas verificables.\n'
                            '3. Problemas críticos y consecuencias.\n'
                            '4. Propuesta corregida paso a paso.\n'
                            '5. Matriz objetivo-variable-análisis, si es pertinente.\n'
                            '6. Información pendiente.\n'
                            '7. Próxima acción prioritaria.\n'},
 '🧭 Asesor Académico': {'short': 'Convierte proyectos complejos en entregables, decisiones y cronogramas.',
                        'engine': 'Gemini principal',
                        'temperature': 0.2,
                        'tasks': {'Plan de tesis o artículo': 'Convierte el objetivo académico en estructura, '
                                                              'entregables, dependencias, responsables, riesgos y '
                                                              'cronograma realista.',
                                  'Hoja de ruta semanal': 'Prioriza tareas según fecha, impacto, dependencia, energía '
                                                          'requerida y evidencia de avance. Evita agendas imposibles.',
                                  'Estructura de capítulo': 'Diseña una estructura argumental con propósito de cada '
                                                            'apartado, evidencia necesaria, transición y criterio de '
                                                            'cierre.',
                                  'Preparación de defensa': 'Organiza narrativa, diapositivas, mensajes centrales, '
                                                            'preguntas difíciles, respuestas y plan de ensayo.',
                                  'Decisión editorial': 'Compara opciones de revista o estrategia editorial usando '
                                                        'criterios proporcionados, sin inventar métricas actuales.',
                                  'Plan de correcciones': 'Convierte comentarios de asesores o revisores en una matriz '
                                                          'de acciones, prioridad, evidencia, cambio y respuesta.'},
                        'prompt': '\n'
                                  'Eres un asesor académico senior orientado a ejecución. Ayudas a transformar tesis,\n'
                                  'artículos, cursos, postulaciones y proyectos en planes realistas y verificables.\n'
                                  '\n'
                                  'PROTOCOLO DE RAZONAMIENTO\n'
                                  '1. Define el resultado final, fecha, restricciones y criterio de aceptación.\n'
                                  '2. Identifica entregables, dependencias, cuellos de botella y decisiones '
                                  'pendientes.\n'
                                  '3. Separa trabajo intelectual, gestión de datos, escritura, revisión y trámites.\n'
                                  '4. Prioriza por impacto, urgencia y dependencia, no por facilidad aparente.\n'
                                  '5. Evita cronogramas que ignoren carga docente, tiempos editoriales o revisión de '
                                  'terceros.\n'
                                  '6. Propone hitos observables y una definición clara de “terminado”.\n'
                                  '7. Identifica riesgos y establece planes de contingencia.\n'
                                  '8. Cuando el usuario entregue un documento, basa el plan en su contenido real.\n'
                                  '\n'
                                  'FORMATO DE SALIDA\n'
                                  '1. Objetivo operativo.\n'
                                  '2. Diagnóstico del estado actual.\n'
                                  '3. Entregables y dependencias.\n'
                                  '4. Plan por etapas o semanas.\n'
                                  '5. Riesgos y contingencias.\n'
                                  '6. Próxima acción prioritaria.\n'},
 '📊 Estadística y Código R': {'short': 'Diseña análisis reproducibles y genera código R auditable.',
                              'engine': 'Gemini en Fase 2 · DeepSeek principal en Fase 3',
                              'temperature': 0.08,
                              'tasks': {'Seleccionar análisis': 'Determina estimando, unidad de análisis, '
                                                                'distribución, efectos fijos/aleatorios, dependencia, '
                                                                'supuestos, contrastes, tamaño del efecto y '
                                                                'sensibilidad.',
                                        'Generar código R completo': 'Entrega un script R reproducible desde '
                                                                     'importación y validación hasta modelo, '
                                                                     'diagnóstico, comparación, gráficos y '
                                                                     'exportación.',
                                        'Depurar error de R': 'Explica la causa probable, reproduce el punto de fallo, '
                                                              'propone una corrección mínima y luego una versión '
                                                              'robusta del bloque afectado.',
                                        'ANOVA y comparaciones': 'Evalúa diseño, independencia, estructura factorial o '
                                                                 'bloques, supuestos sobre residuos, alternativas '
                                                                 'robustas y comparaciones con ajuste apropiado.',
                                        'GLM / GLMM': 'Selecciona familia y enlace según la respuesta; evalúa '
                                                      'sobredispersión, ceros, efectos aleatorios, convergencia, '
                                                      'diagnósticos e interpretación en escala útil.',
                                        'Modelos mixtos y repetidas': 'Representa correctamente sujeto, bloque, '
                                                                      'parcela, tiempo, correlación y jerarquía; evita '
                                                                      'tratar observaciones repetidas como '
                                                                      'independientes.',
                                        'Análisis multivariado': 'Define matriz, transformación, distancia, '
                                                                 'escalamiento, PERMANOVA, dispersión, ordenación y '
                                                                 'relación con variables ambientales.',
                                        'Revisión de resultados': 'Audita tablas, modelos, tamaños del efecto, '
                                                                  'intervalos, gráficos e interpretación. No recalcula '
                                                                  'sin datos suficientes.'},
                              'prompt': '\n'
                                        'Eres un bioestadístico senior y programador científico en R. Tu prioridad es '
                                        'la\n'
                                        'coherencia entre diseño, datos, estimando, modelo, diagnóstico e '
                                        'interpretación.\n'
                                        '\n'
                                        'PROTOCOLO DE RAZONAMIENTO\n'
                                        '1. Identifica pregunta, unidad experimental, unidad de análisis y variable '
                                        'respuesta.\n'
                                        '2. Determina escala, distribución, censura, proporciones, conteos, ceros, '
                                        'repetición,\n'
                                        '   jerarquía, desbalance y datos faltantes.\n'
                                        '3. Define el estimando antes de escoger la prueba.\n'
                                        '4. Justifica familia, enlace, efectos fijos, aleatorios, estructura de '
                                        'correlación y contrastes.\n'
                                        '5. No exijas normalidad de datos crudos cuando el supuesto corresponde a '
                                        'residuos.\n'
                                        '6. Evalúa independencia, sobredispersión, singularidad, convergencia, '
                                        'influencia y ajuste.\n'
                                        '7. Reporta estimaciones, intervalos de confianza, tamaños del efecto y '
                                        'multiplicidad.\n'
                                        '8. Cuando generes R:\n'
                                        '   - entrega un script completo y ejecutable;\n'
                                        '   - declara paquetes y versiones cuando sea relevante;\n'
                                        '   - valida nombres y tipos de columnas;\n'
                                        '   - no inventa rutas ni columnas;\n'
                                        '   - incluye manejo de errores y comentarios útiles;\n'
                                        '   - guarda tablas y figuras con nombres claros;\n'
                                        '   - no imprime resultados numéricos inexistentes.\n'
                                        '9. Si faltan datos, entrega una plantilla marcada y explica qué debe '
                                        'reemplazarse.\n'
                                        '10. Para gráficos, prioriza legibilidad científica, unidades, intervalos y '
                                        'leyendas claras.\n'
                                        '\n'
                                        'FORMATO DE SALIDA\n'
                                        '1. Diagnóstico estadístico.\n'
                                        '2. Modelo recomendado y justificación.\n'
                                        '3. Supuestos y diagnósticos.\n'
                                        '4. Código R reproducible.\n'
                                        '5. Interpretación esperada sin inventar resultados.\n'
                                        '6. Riesgos y alternativas.\n'
                                        '7. Próxima acción prioritaria.\n'},
 '✍️ Editor Académico': {'short': 'Mejora claridad y rigor sin alterar datos, citas ni significado.',
                         'engine': 'Gemini principal',
                         'temperature': 0.18,
                         'tasks': {'Mejorar redacción': 'Reescribe con claridad, precisión, cohesión y registro '
                                                        'académico, preservando contenido, datos y citas.',
                                   'Reducir extensión': 'Reduce redundancias y longitud sin eliminar argumentos, '
                                                        'resultados, limitaciones ni citas esenciales.',
                                   'Fortalecer discusión': 'Organiza hallazgo, comparación, mecanismo, implicación, '
                                                           'límites y proyección; evita sobreinterpretación causal.',
                                   'Introducción científica': 'Construye contexto, problema, brecha, relevancia y '
                                                              'objetivo sin convertirla en revisión enciclopédica.',
                                   'Resumen estructurado': 'Redacta un resumen coherente con objetivo, métodos, '
                                                           'resultados disponibles y conclusión proporcional.',
                                   'Portugués académico': 'Adapta al portugués académico natural, preservando '
                                                          'terminología y referencias.',
                                   'Inglés académico': 'Adapta al inglés académico natural, evitando traducción '
                                                       'literal y sin cambiar contenido científico.',
                                   'APA 7 y consistencia': 'Revisa coherencia formal de citas y referencias '
                                                           'proporcionadas, sin completar datos inexistentes ni '
                                                           'fabricar DOI.'},
                         'prompt': '\n'
                                   'Eres un editor académico senior en español, portugués e inglés. Mejoras textos '
                                   'para\n'
                                   'publicación sin alterar datos, citas, resultados ni significado científico.\n'
                                   '\n'
                                   'PROTOCOLO DE RAZONAMIENTO\n'
                                   '1. Identifica función del fragmento y público objetivo.\n'
                                   '2. Conserva cifras, símbolos, variables, tratamientos, citas y conclusiones '
                                   'sustentadas.\n'
                                   '3. Mejora precisión, cohesión, progresión temática, concordancia y economía '
                                   'verbal.\n'
                                   '4. Elimina redundancia, vaguedad, nominalizaciones innecesarias y conectores '
                                   'mecánicos.\n'
                                   '5. Evita intensificar afirmaciones más allá de la evidencia.\n'
                                   '6. No agregues referencias ni completes metadatos no proporcionados.\n'
                                   '7. En traducción, adapta sintaxis y registro, no traduce palabra por palabra.\n'
                                   '8. Cuando haya problemas científicos, sepáralos de los problemas de estilo.\n'
                                   '\n'
                                   'FORMATO DE SALIDA\n'
                                   '1. Texto revisado listo para usar.\n'
                                   '2. Cambios sustantivos realizados.\n'
                                   '3. Problemas científicos o datos que requieren verificación.\n'},
 '🔎 Revisor Crítico': {'short': 'Simula una revisión por pares exigente, útil y trazable.',
                       'engine': 'Gemini principal',
                       'temperature': 0.12,
                       'tasks': {'Revisión por pares completa': 'Evalúa originalidad, pregunta, métodos, resultados, '
                                                                'discusión, transparencia, reproducibilidad y aporte.',
                                 'Comentarios mayores y menores': 'Separa problemas que afectan validez de mejoras '
                                                                  'editoriales; explica consecuencia y corrección '
                                                                  'esperada.',
                                 'Coherencia resultados-conclusiones': 'Detecta afirmaciones no sustentadas, '
                                                                       'causalidad indebida, generalización, omisiones '
                                                                       'y contradicciones.',
                                 'Reproducibilidad': 'Audita detalle metodológico, disponibilidad de datos/código, '
                                                     'decisiones analíticas, versiones, semillas y trazabilidad.',
                                 'Respuesta a revisores': 'Evalúa si cada respuesta atiende el comentario, identifica '
                                                          'evasiones y propone una versión técnica y cordial.',
                                 'Veredicto editorial simulado': 'Propone un veredicto argumentado y provisional, sin '
                                                                 'suplantar al editor ni inventar políticas de '
                                                                 'revista.'},
                       'prompt': '\n'
                                 'Eres un revisor por pares senior, riguroso y constructivo. Evalúas manuscritos y\n'
                                 'respuestas a revisores con atención a validez, transparencia y aporte real.\n'
                                 '\n'
                                 'PROTOCOLO DE RAZONAMIENTO\n'
                                 '1. Resume la contribución en términos verificables.\n'
                                 '2. Evalúa originalidad y relevancia sin confundir tema novedoso con método válido.\n'
                                 '3. Revisa alineación entre pregunta, diseño, datos, análisis y conclusiones.\n'
                                 '4. Identifica amenazas que cambian la interpretación o invalidan el estudio.\n'
                                 '5. Distingue comentarios mayores de correcciones menores.\n'
                                 '6. Para cada crítica, explica evidencia, consecuencia y cambio esperado.\n'
                                 '7. Detecta causalidad indebida, lenguaje excesivo, selectividad y omisiones.\n'
                                 '8. Evalúa reproducibilidad, ética, disponibilidad y transparencia.\n'
                                 '9. No rechaces por preferencias personales ni inventes requisitos de una revista.\n'
                                 '10. El veredicto debe ser simulado, provisional y coherente con los problemas '
                                 'descritos.\n'
                                 '\n'
                                 'FORMATO DE SALIDA\n'
                                 '1. Resumen del manuscrito.\n'
                                 '2. Evaluación general.\n'
                                 '3. Comentarios mayores numerados.\n'
                                 '4. Comentarios menores numerados.\n'
                                 '5. Recomendación editorial simulada.\n'
                                 '6. Prioridad de corrección.\n'}}

# Fortalecimiento específico de perfiles sin alterar sus tareas originales.
PROFILES["📚 Revisor de Literatura"]["prompt"] += """

CONTROL ESTRICTO DE EVIDENCIA
- Formula preguntas PICO, PECO, PCC, SPIDER u otra estructura solo cuando corresponda.
- Distingue PRISMA-P, PRISMA 2020, PRISMA-ScR, JBI, PROSPERO, OSF y protocolos editoriales.
- No declares elegibilidad para PROSPERO sin comprobar alcance y requisitos vigentes.
- Separa estrategia de búsqueda reproducible, cribado, extracción, evaluación crítica y síntesis.
- Toda referencia final debe proceder de una fuente consultada y ser verificable por DOI, PMID, URL editorial u organismo oficial.
"""

PROFILES["🔬 Metodólogo"]["prompt"] += """

CONTROL ESTRICTO DE DISEÑO
- No clasifiques un diseño solo por el número de tratamientos o grupos.
- Distingue unidad experimental, unidad de observación, réplica biológica, réplica técnica y medida repetida.
- No confundas la presencia de un control con poder estadístico.
- Evalúa aleatorización, bloqueo, cegamiento cuando sea pertinente, pérdidas, dependencia y fuentes de sesgo.
"""

PROFILES["📊 Estadística y Código R"]["prompt"] += """

PROTOCOLO REFORZADO PARA R
- Primero identifica pregunta, unidad experimental, variable respuesta, estructura de dependencia y estimando.
- Explica por qué el modelo es adecuado y qué alternativas se descartan.
- Entrega un script ejecutable desde importación hasta exportación.
- Incluye comprobaciones de tipos, valores perdidos, niveles, duplicados y tamaños por grupo.
- Incluye diagnóstico, tamaño del efecto, intervalos de confianza y visualización cuando corresponda.
- Nunca inventes columnas, resultados, paquetes o funciones.
- Cuando una función o paquete sea relevante y se use búsqueda, prioriza documentación oficial de R, CRAN o Bioconductor.
"""


# ============================================================
# MODELOS DE DATOS
# ============================================================

@dataclass
class ProviderResult:
    provider: str
    model: str
    text: str
    usage: Dict[str, int]
    sources: List[Dict[str, str]]
    elapsed: float
    attempts: List[Dict[str, str]]
    raw_reference_report: Optional[Dict[str, Any]] = None


# ============================================================
# ESTADO DE SESIÓN
# ============================================================

for name, default in {
    "appearance": "Sistema",
    "bulk_keys_input": "",
    "last_result": None,
    "last_mode": "",
    "crossref_mailto": os.getenv("CROSSREF_MAILTO", ""),
    "ncbi_api_key": os.getenv("NCBI_API_KEY", ""),
}.items():
    st.session_state.setdefault(name, default)

for provider_name, meta in PROVIDERS.items():
    st.session_state.setdefault(f"session_{meta['env'].lower()}", "")
    st.session_state.setdefault(f"status_{provider_name}", "not_tested")
    st.session_state.setdefault(f"message_{provider_name}", "Sin probar")
    st.session_state.setdefault(f"model_{provider_name}", meta["default_model"])
    st.session_state.setdefault(f"models_{provider_name}", list(dict.fromkeys([meta["default_model"], *meta["fallbacks"]])))


# ============================================================
# APARIENCIA
# ============================================================

def inject_css(mode: str) -> None:
    light = """
      --oc-bg:#f4f7fb; --oc-panel:#ffffff; --oc-panel-2:#f8fafc; --oc-text:#172033;
      --oc-muted:#617083; --oc-border:#dfe6ef; --oc-primary:#4f46e5; --oc-primary-2:#7c3aed;
      --oc-ok:#0f9d72; --oc-warn:#c27a00; --oc-bad:#c93c55; --oc-shadow:0 18px 45px rgba(20,32,55,.08);
    """
    dark = """
      --oc-bg:#0b1020; --oc-panel:#12192a; --oc-panel-2:#182136; --oc-text:#eef2ff;
      --oc-muted:#aab4c8; --oc-border:#2b3750; --oc-primary:#8b83ff; --oc-primary-2:#b58cff;
      --oc-ok:#4dd4a8; --oc-warn:#f0b44c; --oc-bad:#ff7188; --oc-shadow:0 20px 55px rgba(0,0,0,.32);
    """
    if mode == "Oscuro":
        root = dark
        media = ""
    elif mode == "Claro":
        root = light
        media = ""
    else:
        root = light
        media = f"@media (prefers-color-scheme: dark) {{ :root {{ {dark} }} }}"

    st.markdown(
        f"""
        <style>
        :root {{ {root} }}
        {media}
        .stApp {{ background:var(--oc-bg); color:var(--oc-text); }}
        .block-container {{ padding-top:1rem; padding-bottom:3rem; max-width:1500px; }}
        [data-testid="stSidebar"] {{ background:var(--oc-panel); border-right:1px solid var(--oc-border); }}
        .oc-hero {{
            background:linear-gradient(135deg, rgba(79,70,229,.16), rgba(124,58,237,.08), rgba(15,157,114,.08));
            border:1px solid var(--oc-border); border-radius:22px; padding:1.2rem 1.35rem;
            box-shadow:var(--oc-shadow); margin-bottom:.9rem;
        }}
        .oc-hero h1 {{ margin:0; font-size:2rem; color:var(--oc-text); letter-spacing:-.035em; }}
        .oc-hero p {{ margin:.4rem 0 0; color:var(--oc-muted); }}
        .oc-card {{
            background:var(--oc-panel); border:1px solid var(--oc-border); border-radius:18px;
            padding:1rem 1.05rem; box-shadow:0 8px 24px rgba(20,32,55,.04); height:100%;
        }}
        .oc-card h3 {{ margin:.05rem 0 .35rem; color:var(--oc-text); font-size:1.05rem; }}
        .oc-card p {{ margin:.15rem 0; color:var(--oc-muted); }}
        .oc-chip {{
            display:inline-flex; align-items:center; gap:.4rem; border:1px solid var(--oc-border);
            border-radius:999px; padding:.25rem .62rem; margin:.1rem .2rem .1rem 0;
            background:var(--oc-panel-2); color:var(--oc-text); font-size:.82rem;
        }}
        .oc-dot {{ width:.52rem; height:.52rem; border-radius:50%; display:inline-block; }}
        .oc-dot-ok {{ background:var(--oc-ok); }}
        .oc-dot-off {{ background:#8994a8; }}
        .oc-dot-error {{ background:var(--oc-bad); }}
        .oc-note {{
            border-left:4px solid var(--oc-primary); background:var(--oc-panel);
            border-radius:14px; padding:.8rem 1rem; border-top:1px solid var(--oc-border);
            border-right:1px solid var(--oc-border); border-bottom:1px solid var(--oc-border);
        }}
        div[data-testid="stMetric"] {{
            background:var(--oc-panel); border:1px solid var(--oc-border); border-radius:16px;
            padding:.55rem .75rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            border-radius:12px; font-weight:650;
        }}
        textarea, input {{ border-radius:12px !important; }}
        [data-baseweb="tab-list"] {{ gap:.35rem; }}
        [data-baseweb="tab"] {{ border-radius:10px; padding:.45rem .8rem; }}
        hr {{ border-color:var(--oc-border); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(st.session_state.appearance)


# ============================================================
# SEGURIDAD Y CLAVES
# ============================================================

def is_real_key(value: Optional[str]) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return len(normalized) >= 12 and not any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def shared_secrets_allowed() -> bool:
    raw = str(os.getenv("ALLOW_SHARED_SECRETS", "")).strip().lower()
    env_allowed = raw in {"1", "true", "yes", "si", "sí"}
    try:
        secret_allowed = bool(st.secrets.get("ALLOW_SHARED_SECRETS", False))
    except Exception:
        secret_allowed = False
    return env_allowed or secret_allowed


def get_key_with_source(env_name: str) -> Tuple[str, str]:
    session_value = str(st.session_state.get(f"session_{env_name.lower()}", "")).strip()
    if is_real_key(session_value):
        return session_value, "sesión"

    env_value = str(os.getenv(env_name, "")).strip()
    if is_real_key(env_value):
        return env_value, ".env"

    if shared_secrets_allowed():
        try:
            secret_value = str(st.secrets.get(env_name, "")).strip()
        except Exception:
            secret_value = ""
        if is_real_key(secret_value):
            return secret_value, "st.secrets"

    return "", ""


def get_key(provider_name: str) -> str:
    return get_key_with_source(PROVIDERS[provider_name]["env"])[0]


def mask_key(value: str) -> str:
    if not value:
        return "No configurada"
    if len(value) < 12:
        return "Configurada"
    return f"{value[:4]}…{value[-4:]}"


def sanitize_error(error: Exception) -> str:
    text = str(error).strip() or error.__class__.__name__
    patterns = [
        r"AIza[\w-]+",
        r"AQ\.[\w-]+",
        r"sk-[A-Za-z0-9_\-]{10,}",
        r"key=[^\s,&]+",
        r"Bearer\s+[A-Za-z0-9._\-]+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "[CLAVE_OCULTA]", text, flags=re.IGNORECASE)
    return text[:1600]


def classify_error(provider: str, error: Exception) -> str:
    text = sanitize_error(error)
    lowered = text.lower()
    if any(x in lowered for x in ("401", "invalid api key", "api key not valid", "authentication")):
        return f"{provider}: clave inválida o no habilitada."
    if "403" in lowered or "permission" in lowered:
        return f"{provider}: la cuenta no tiene permiso para ese recurso o modelo."
    if "404" in lowered or "not found" in lowered:
        return f"{provider}: modelo o recurso no disponible."
    if "429" in lowered or "quota" in lowered or "rate limit" in lowered:
        return f"{provider}: cuota o límite temporal alcanzado."
    if any(x in lowered for x in ("503", "unavailable", "high demand", "overloaded")):
        return f"{provider}: servicio temporalmente saturado."
    if "timeout" in lowered or "timed out" in lowered:
        return f"{provider}: tiempo de espera agotado."
    return f"{provider}: {text}"


def parse_key_bundle(raw: str) -> Tuple[Dict[str, str], List[str]]:
    found: Dict[str, str] = {}
    ignored: List[str] = []
    for original_line in (raw or "").splitlines():
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            left, right = line.split("=", 1)
        elif ":" in line:
            left, right = line.split(":", 1)
        else:
            ignored.append(original_line)
            continue
        alias = re.sub(r"\s+", "_", left.strip().upper())
        env_name = KEY_ALIASES.get(alias)
        value = right.strip().strip('"').strip("'")
        if env_name and is_real_key(value):
            found[env_name] = value
        else:
            ignored.append(original_line)
    return found, ignored


def store_session_keys(keys: Dict[str, str]) -> None:
    for env_name, value in keys.items():
        st.session_state[f"session_{env_name.lower()}"] = value
        for provider, meta in PROVIDERS.items():
            if meta["env"] == env_name:
                st.session_state[f"status_{provider}"] = "not_tested"
                st.session_state[f"message_{provider}"] = "Clave cargada; falta probarla."


# ============================================================
# UTILIDADES
# ============================================================

def unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(x.strip() for x in items if x and x.strip()))


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
            valid.append(url)
        else:
            invalid.append(url)
    return unique(valid)[:MAX_URLS], invalid


def is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower().replace("www.", "")
    return host in {"youtube.com", "m.youtube.com", "youtu.be"}


def profile_system_prompt(profile_name: str, task_name: str) -> str:
    profile = PROFILES[profile_name]
    return (
        profile["prompt"].strip()
        + "\n\nTAREA ESPECÍFICA\n"
        + task_name
        + "\n"
        + profile["tasks"][task_name].strip()
        + "\n\n"
        + COMMON_RULES
    )


def extract_pdf_text(pdf_files: Sequence[Any]) -> Tuple[str, List[Dict[str, str]], List[str]]:
    blocks: List[str] = []
    sources: List[Dict[str, str]] = []
    warnings: List[str] = []
    remaining = MAX_EXTRACTED_CHARS
    for uploaded in pdf_files:
        if remaining <= 0:
            warnings.append("Se alcanzó el límite de texto extraído.")
            break
        try:
            reader = PdfReader(io.BytesIO(uploaded.getvalue()))
            page_texts: List[str] = []
            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    page_texts.append(f"[Página {index + 1}]\n{text}")
                if sum(len(x) for x in page_texts) >= remaining:
                    break
            joined = "\n\n".join(page_texts)[:remaining]
            remaining -= len(joined)
            blocks.append(f"\n\n===== PDF: {uploaded.name} =====\n{joined}")
            sources.append({"type": "PDF extraído", "title": uploaded.name, "url": ""})
            if not joined.strip():
                warnings.append(f"{uploaded.name}: no se pudo extraer texto; puede ser un PDF escaneado.")
        except Exception as exc:
            warnings.append(f"{uploaded.name}: {sanitize_error(exc)}")
    return "".join(blocks), sources, warnings


def build_text_prompt(user_prompt: str, pdf_text: str, urls: Sequence[str]) -> str:
    chunks = [user_prompt.strip()]
    if pdf_text:
        chunks.append("DOCUMENTOS ADJUNTOS EXTRAÍDOS:\n" + pdf_text)
    if urls:
        chunks.append(
            "ENLACES PROPORCIONADOS POR EL USUARIO:\n- "
            + "\n- ".join(urls)
            + "\nNo afirmes haberlos consultado si el proveedor no dispone de acceso web."
        )
    return "\n\n".join(chunks)


def usage_dict(prompt: int = 0, output: int = 0, total: int = 0, thoughts: int = 0, tools: int = 0) -> Dict[str, int]:
    return {
        "prompt_tokens": int(prompt or 0),
        "output_tokens": int(output or 0),
        "thoughts_tokens": int(thoughts or 0),
        "tool_tokens": int(tools or 0),
        "total_tokens": int(total or prompt + output + thoughts + tools),
    }


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


# ============================================================
# CONECTORES DE IA
# ============================================================

def gemini_sources(response: Any) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    try:
        candidate = response.candidates[0]
        metadata = get_attr(candidate, "grounding_metadata")
        for chunk in get_attr(metadata, "grounding_chunks", []) or []:
            web = get_attr(chunk, "web")
            if web:
                title = str(get_attr(web, "title", "") or "").strip()
                uri = str(get_attr(web, "uri", "") or "").strip()
                if uri:
                    output.append({"type": "Web verificada por grounding", "title": title or uri, "url": uri})
    except Exception:
        pass
    seen: set[str] = set()
    deduped: List[Dict[str, str]] = []
    for item in output:
        key = item["url"]
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def gemini_usage(response: Any) -> Dict[str, int]:
    usage = get_attr(response, "usage_metadata")
    return usage_dict(
        prompt=get_attr(usage, "prompt_token_count", 0),
        output=get_attr(usage, "candidates_token_count", 0),
        total=get_attr(usage, "total_token_count", 0),
        thoughts=get_attr(usage, "thoughts_token_count", 0),
        tools=get_attr(usage, "tool_use_prompt_token_count", 0),
    )


def call_gemini(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    pdf_files: Sequence[Any],
    urls: Sequence[str],
    max_tokens: int,
    temperature: float,
    google_search: bool,
) -> ProviderResult:
    if not is_real_key(api_key):
        raise ValueError("Gemini no tiene una clave válida.")

    parts: List[types.Part] = []
    sources: List[Dict[str, str]] = []
    normal_urls: List[str] = []
    for uploaded in pdf_files:
        parts.append(types.Part.from_bytes(data=uploaded.getvalue(), mime_type="application/pdf"))
        sources.append({"type": "PDF adjunto", "title": uploaded.name, "url": ""})
    for url in urls:
        if is_youtube_url(url):
            parts.append(types.Part(file_data=types.FileData(file_uri=url)))
            sources.append({"type": "YouTube", "title": url, "url": url})
        else:
            normal_urls.append(url)

    contextual_prompt = user_prompt.strip()
    if normal_urls:
        contextual_prompt += (
            "\n\nURLS QUE DEBES CONSULTAR MEDIANTE URL CONTEXT:\n- "
            + "\n- ".join(normal_urls)
            + "\nIndica cualquier URL no recuperada."
        )
    parts.append(types.Part(text=contextual_prompt))
    content = types.Content(parts=parts)

    tools: List[Any] = []
    if google_search:
        tools.append({"google_search": {}})
    if normal_urls:
        tools.append({"url_context": {}})

    client = genai.Client(api_key=api_key)
    attempts: List[Dict[str, str]] = []
    started = time.perf_counter()
    models = unique([model, *PROVIDERS["Gemini"]["fallbacks"]])
    last_error: Optional[Exception] = None

    for candidate_model in models:
        for delay in (0, 2, 5):
            if delay:
                time.sleep(delay)
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        tools=tools or None,
                    ),
                )
                text = str(response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini devolvió una respuesta vacía.")
                attempts.append({"model": candidate_model, "status": "ok", "message": "Respuesta obtenida"})
                sources.extend(gemini_sources(response))
                return ProviderResult(
                    provider="Gemini",
                    model=candidate_model,
                    text=text,
                    usage=gemini_usage(response),
                    sources=_dedupe_sources(sources),
                    elapsed=round(time.perf_counter() - started, 2),
                    attempts=attempts,
                )
            except Exception as exc:
                last_error = exc
                raw = sanitize_error(exc).lower()
                transient = any(x in raw for x in ("503", "unavailable", "high demand", "overloaded"))
                if transient and delay < 5:
                    attempts.append({"model": candidate_model, "status": "retry", "message": "Saturación temporal"})
                    continue
                attempts.append({"model": candidate_model, "status": "error", "message": classify_error("Gemini", exc)})
                break

    raise RuntimeError(" | ".join(x["message"] for x in attempts) or sanitize_error(last_error or RuntimeError("Sin respuesta")))


def call_openai_compatible(
    *,
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    thinking: bool,
) -> ProviderResult:
    meta = PROVIDERS[provider]
    if not is_real_key(api_key):
        raise ValueError(f"{provider} no tiene una clave válida.")
    client = OpenAI(api_key=api_key, base_url=meta["base_url"], timeout=180.0)
    attempts: List[Dict[str, str]] = []
    started = time.perf_counter()
    last_error: Optional[Exception] = None

    for candidate_model in unique([model, *meta["fallbacks"]]):
        try:
            kwargs: Dict[str, Any] = {
                "model": candidate_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            }
            if provider == "DeepSeek":
                kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
                if thinking:
                    kwargs["reasoning_effort"] = "high"
                else:
                    kwargs["temperature"] = 0.15
            elif provider == "Kimi":
                if candidate_model.startswith("kimi-k2.6"):
                    kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
                if not thinking:
                    kwargs["temperature"] = 0.15

            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            text = str(message.content or "").strip()
            if not text:
                raise RuntimeError(f"{provider} devolvió una respuesta vacía.")
            usage = get_attr(response, "usage")
            attempts.append({"model": candidate_model, "status": "ok", "message": "Respuesta obtenida"})
            return ProviderResult(
                provider=provider,
                model=candidate_model,
                text=text,
                usage=usage_dict(
                    prompt=get_attr(usage, "prompt_tokens", 0),
                    output=get_attr(usage, "completion_tokens", 0),
                    total=get_attr(usage, "total_tokens", 0),
                ),
                sources=[],
                elapsed=round(time.perf_counter() - started, 2),
                attempts=attempts,
            )
        except Exception as exc:
            last_error = exc
            attempts.append({"model": candidate_model, "status": "error", "message": classify_error(provider, exc)})
            continue

    raise RuntimeError(" | ".join(x["message"] for x in attempts) or sanitize_error(last_error or RuntimeError("Sin respuesta")))


def call_openai(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> ProviderResult:
    if not is_real_key(api_key):
        raise ValueError("OpenAI no tiene una clave válida.")
    client = OpenAI(api_key=api_key, timeout=180.0)
    started = time.perf_counter()
    attempts: List[Dict[str, str]] = []
    last_error: Optional[Exception] = None

    for candidate_model in unique([model, *PROVIDERS["OpenAI"]["fallbacks"]]):
        try:
            response = client.responses.create(
                model=candidate_model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=max_tokens,
            )
            text = str(get_attr(response, "output_text", "") or "").strip()
            if not text:
                raise RuntimeError("OpenAI devolvió una respuesta vacía.")
            usage = get_attr(response, "usage")
            attempts.append({"model": candidate_model, "status": "ok", "message": "Respuesta obtenida"})
            return ProviderResult(
                provider="OpenAI",
                model=candidate_model,
                text=text,
                usage=usage_dict(
                    prompt=get_attr(usage, "input_tokens", 0),
                    output=get_attr(usage, "output_tokens", 0),
                    total=get_attr(usage, "total_tokens", 0),
                ),
                sources=[],
                elapsed=round(time.perf_counter() - started, 2),
                attempts=attempts,
            )
        except Exception as exc:
            last_error = exc
            attempts.append({"model": candidate_model, "status": "error", "message": classify_error("OpenAI", exc)})
    raise RuntimeError(" | ".join(x["message"] for x in attempts) or sanitize_error(last_error or RuntimeError("Sin respuesta")))


def call_provider(
    provider: str,
    *,
    system_prompt: str,
    user_prompt: str,
    pdf_files: Sequence[Any],
    urls: Sequence[str],
    max_tokens: int,
    temperature: float,
    thinking: bool = False,
    google_search: bool = False,
) -> ProviderResult:
    model = st.session_state[f"model_{provider}"]
    key = get_key(provider)
    if provider == "Gemini":
        return call_gemini(
            api_key=key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            pdf_files=pdf_files,
            urls=urls,
            max_tokens=max_tokens,
            temperature=temperature,
            google_search=google_search,
        )

    pdf_text, pdf_sources, warnings = extract_pdf_text(pdf_files)
    prompt = build_text_prompt(user_prompt, pdf_text, urls)
    if warnings:
        prompt += "\n\nADVERTENCIAS DE EXTRACCIÓN:\n- " + "\n- ".join(warnings)

    if provider in {"DeepSeek", "Kimi"}:
        result = call_openai_compatible(
            provider=provider,
            api_key=key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=max_tokens,
            thinking=thinking,
        )
    else:
        result = call_openai(
            api_key=key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_tokens=max_tokens,
        )
    result.sources.extend(pdf_sources)
    return result


def list_provider_models(provider: str) -> List[str]:
    key = get_key(provider)
    if not is_real_key(key):
        raise ValueError("Primero configura una clave.")
    if provider == "Gemini":
        client = genai.Client(api_key=key)
        names = []
        for item in client.models.list():
            name = str(get_attr(item, "name", "") or "")
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            if name:
                names.append(name)
        return sorted(unique(names))
    if provider in {"DeepSeek", "Kimi"}:
        client = OpenAI(api_key=key, base_url=PROVIDERS[provider]["base_url"], timeout=60.0)
    else:
        client = OpenAI(api_key=key, timeout=60.0)
    return sorted(unique(str(item.id) for item in client.models.list().data))


def test_provider(provider: str) -> ProviderResult:
    prompt = "Responde únicamente: conexión correcta"
    return call_provider(
        provider,
        system_prompt="Eres un verificador de conexión. Sigue exactamente la instrucción.",
        user_prompt=prompt,
        pdf_files=[],
        urls=[],
        max_tokens=30,
        temperature=0.0,
        thinking=False,
        google_search=False,
    )


# ============================================================
# VERIFICACIÓN DE REFERENCIAS
# ============================================================

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
PMID_RE = re.compile(r"\bPMID\s*[:#]?\s*(\d{5,9})\b", re.IGNORECASE)


def _clean_doi(value: str) -> str:
    return value.rstrip(".,;:)]}").lower()


def extract_dois(text: str, sources: Sequence[Dict[str, str]]) -> List[str]:
    values = [_clean_doi(x) for x in DOI_RE.findall(text or "")]
    for source in sources:
        values.extend(_clean_doi(x) for x in DOI_RE.findall(source.get("url", "")))
    return unique(values)


def extract_pmids(text: str, sources: Sequence[Dict[str, str]]) -> List[str]:
    values = PMID_RE.findall(text or "")
    for source in sources:
        url = source.get("url", "")
        match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
        if match:
            values.append(match.group(1))
    return unique(values)


def crossref_lookup(doi: str, mailto: str) -> Optional[Dict[str, Any]]:
    headers = {"User-Agent": f"OrquestadorCientifico/3.0 (mailto:{mailto or 'not-provided'})"}
    params = {"mailto": mailto} if mailto else {}
    response = requests.get(
        "https://api.crossref.org/works/" + quote(doi, safe=""),
        headers=headers,
        params=params,
        timeout=18,
    )
    if response.status_code != 200:
        return None
    return response.json().get("message") or None


def openalex_lookup(doi: str, mailto: str) -> Optional[Dict[str, Any]]:
    params = {"filter": f"doi:https://doi.org/{doi}", "per-page": 1}
    if mailto:
        params["mailto"] = mailto
    response = requests.get("https://api.openalex.org/works", params=params, timeout=18)
    if response.status_code != 200:
        return None
    results = response.json().get("results") or []
    return results[0] if results else None


def pubmed_lookup(pmids: Sequence[str], api_key: str = "") -> Dict[str, Dict[str, Any]]:
    if not pmids:
        return {}
    params: Dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids[:50]),
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params=params,
        timeout=18,
    )
    if response.status_code != 200:
        return {}
    result = response.json().get("result") or {}
    return {pmid: result.get(pmid, {}) for pmid in pmids if result.get(pmid)}


def _authors_from_crossref(item: Dict[str, Any]) -> str:
    authors = item.get("author") or []
    names = []
    for author in authors[:8]:
        family = str(author.get("family") or "").strip()
        given = str(author.get("given") or "").strip()
        if family:
            names.append(f"{family}, {given[:1]}." if given else family)
    if len(authors) > 8:
        names.append("et al.")
    return "; ".join(names) or "Autoría no disponible"


def _year_from_crossref(item: Dict[str, Any]) -> str:
    for field in ("published-print", "published-online", "issued", "created"):
        parts = (((item.get(field) or {}).get("date-parts") or [[None]])[0])
        if parts and parts[0]:
            return str(parts[0])
    return "s. f."


def format_crossref_reference(item: Dict[str, Any], doi: str) -> str:
    title = " ".join(item.get("title") or []) or "Título no disponible"
    journal = " ".join(item.get("container-title") or []) or "Fuente no disponible"
    volume = str(item.get("volume") or "")
    issue = str(item.get("issue") or "")
    pages = str(item.get("page") or "")
    locator = volume + (f"({issue})" if issue else "")
    if pages:
        locator += (", " if locator else "") + pages
    locator = f", {locator}" if locator else ""
    return (
        f"{_authors_from_crossref(item)} ({_year_from_crossref(item)}). "
        f"{title}. *{journal}*{locator}. https://doi.org/{doi}"
    )


def _dedupe_sources(sources: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    output: List[Dict[str, str]] = []
    for source in sources:
        key = source.get("url") or f"{source.get('type')}::{source.get('title')}"
        if key and key not in seen:
            seen.add(key)
            output.append(source)
    return output


def strip_reference_section(text: str) -> str:
    pattern = re.compile(
        r"\n#{1,6}\s*(referencias(?:\s+verificadas)?|bibliograf[ií]a|references)\s*\n",
        re.IGNORECASE,
    )
    match = pattern.search(text or "")
    return (text[: match.start()] if match else text).rstrip()


def build_reference_report(text: str, sources: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    dois = extract_dois(text, sources)
    pmids = extract_pmids(text, sources)
    verified: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    mailto = str(st.session_state.get("crossref_mailto", "")).strip()
    ncbi_key = str(st.session_state.get("ncbi_api_key", "")).strip()

    for doi in dois[:30]:
        crossref = None
        openalex = None
        try:
            crossref = crossref_lookup(doi, mailto)
        except Exception:
            crossref = None
        try:
            openalex = openalex_lookup(doi, mailto)
        except Exception:
            openalex = None
        if crossref:
            verified.append(
                {
                    "kind": "DOI",
                    "id": doi,
                    "reference": format_crossref_reference(crossref, doi),
                    "status": "Crossref + OpenAlex" if openalex else "Crossref",
                    "url": f"https://doi.org/{doi}",
                }
            )
        elif openalex:
            verified.append(
                {
                    "kind": "DOI",
                    "id": doi,
                    "reference": str(openalex.get("display_name") or doi),
                    "status": "OpenAlex",
                    "url": f"https://doi.org/{doi}",
                }
            )
        else:
            excluded.append({"id": doi, "reason": "DOI no confirmado en Crossref ni OpenAlex"})

    try:
        pubmed = pubmed_lookup(pmids, ncbi_key)
    except Exception:
        pubmed = {}
    for pmid in pmids:
        item = pubmed.get(pmid)
        if item:
            verified.append(
                {
                    "kind": "PMID",
                    "id": pmid,
                    "reference": f"{item.get('title', 'Título no disponible')} {item.get('fulljournalname', '')}".strip(),
                    "status": "PubMed/NCBI",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
            )
        else:
            excluded.append({"id": pmid, "reason": "PMID no confirmado en PubMed"})

    web_sources: List[Dict[str, Any]] = []
    for source in _dedupe_sources(sources):
        url = source.get("url", "")
        if not url:
            continue
        if any(v.get("url") == url for v in verified):
            continue
        web_sources.append(
            {
                "kind": "WEB",
                "id": url,
                "reference": source.get("title") or url,
                "status": source.get("type") or "Fuente web recuperada",
                "url": url,
            }
        )

    return {
        "verified": verified,
        "web_sources": web_sources[:20],
        "excluded": excluded,
        "counts": {
            "verified_bibliographic": len(verified),
            "verified_web": len(web_sources[:20]),
            "excluded": len(excluded),
        },
    }


def references_markdown(report: Dict[str, Any]) -> str:
    lines = ["## Referencias verificadas", ""]
    verified = report.get("verified") or []
    web_sources = report.get("web_sources") or []
    excluded = report.get("excluded") or []

    if verified:
        lines.append("### Registros bibliográficos corroborados")
        for index, item in enumerate(verified, 1):
            lines.append(f"{index}. {item['reference']}")
            lines.append(f"   - Estado: **{item['status']}**")
    if web_sources:
        lines.append("")
        lines.append("### Fuentes web efectivamente recuperadas")
        for index, item in enumerate(web_sources, 1):
            lines.append(f"{index}. [{item['reference']}]({item['url']})")
            lines.append(f"   - Estado: **{item['status']}**")
    if not verified and not web_sources:
        lines.append("No se logró confirmar ninguna referencia externa para esta respuesta.")
    if excluded:
        lines.append("")
        lines.append("### Registros excluidos por falta de confirmación")
        for item in excluded:
            lines.append(f"- `{item['id']}`: {item['reason']}")

    lines.extend(
        [
            "",
            "> La existencia de un DOI, PMID o registro no demuestra por sí sola la indexación o calidad de la revista. "
            "Scopus, Web of Science y SJR deben comprobarse por separado en sus portales oficiales.",
        ]
    )
    return "\n".join(lines)


# ============================================================
# ENRUTAMIENTO Y FLUJOS
# ============================================================

def configured_providers() -> List[str]:
    return [name for name in PROVIDERS if is_real_key(get_key(name))]


def auto_provider(profile_name: str, task_name: str, user_prompt: str, pdf_files: Sequence[Any], urls: Sequence[str]) -> str:
    available = configured_providers()
    if profile_name == "📊 Estadística y Código R" and "DeepSeek" in available:
        return "DeepSeek"
    if pdf_files or urls:
        return "Gemini" if "Gemini" in available else (available[0] if available else "Gemini")
    if len(user_prompt) > 30_000 and "Kimi" in available:
        return "Kimi"
    if "Gemini" in available:
        return "Gemini"
    return available[0] if available else "Gemini"


def run_single(
    provider: str,
    *,
    profile_name: str,
    task_name: str,
    user_prompt: str,
    pdf_files: Sequence[Any],
    urls: Sequence[str],
    max_tokens: int,
    evidence_strict: bool,
) -> ProviderResult:
    system_prompt = profile_system_prompt(profile_name, task_name)
    temperature = float(PROFILES[profile_name]["temperature"])
    result = call_provider(
        provider,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pdf_files=pdf_files,
        urls=urls,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=profile_name == "📊 Estadística y Código R",
        google_search=evidence_strict and provider == "Gemini",
    )
    if evidence_strict and provider == "Gemini":
        report = build_reference_report(result.text, result.sources)
        result.text = strip_reference_section(result.text) + "\n\n" + references_markdown(report)
        result.raw_reference_report = report
    return result


def run_r_audited(
    *,
    profile_name: str,
    task_name: str,
    user_prompt: str,
    pdf_files: Sequence[Any],
    urls: Sequence[str],
    max_tokens: int,
) -> Dict[str, Any]:
    if not get_key("DeepSeek") or not get_key("Gemini"):
        raise ValueError("Código R auditado requiere claves válidas de DeepSeek y Gemini.")

    system_prompt = profile_system_prompt(profile_name, task_name)
    primary = call_provider(
        "DeepSeek",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pdf_files=pdf_files,
        urls=urls,
        max_tokens=max_tokens,
        temperature=0.1,
        thinking=True,
        google_search=False,
    )

    audit_prompt = f"""
AUDITA Y CORRIGE LA PROPUESTA DE DEEPSEEK.

CONSULTA ORIGINAL
{user_prompt}

PROPUESTA DE DEEPSEEK
{primary.text}

REQUISITOS DE AUDITORÍA
1. Verifica la coherencia entre diseño, unidad experimental, variable, distribución y modelo.
2. Revisa sintaxis R, paquetes, funciones, argumentos, objetos y orden de ejecución.
3. Identifica pseudorreplicación, dependencia, sobredispersión, desbalance y supuestos omitidos.
4. Cuando se recomiende un paquete o método, corrobóralo preferentemente con documentación oficial de R, CRAN, Bioconductor o publicación metodológica.
5. Entrega un código final corregido, completo y ejecutable.
6. Separa: diagnóstico, correcciones, script final, interpretación y referencias verificadas.
7. No inventes resultados ni columnas.
"""
    auditor = call_provider(
        "Gemini",
        system_prompt=profile_system_prompt("🔎 Revisor Crítico", "Reproducibilidad"),
        user_prompt=audit_prompt,
        pdf_files=[],
        urls=[],
        max_tokens=max_tokens,
        temperature=0.1,
        thinking=False,
        google_search=True,
    )
    report = build_reference_report(auditor.text, auditor.sources)
    auditor.text = strip_reference_section(auditor.text) + "\n\n" + references_markdown(report)
    auditor.raw_reference_report = report

    return {
        "kind": "r_audited",
        "primary": asdict(primary),
        "auditor": asdict(auditor),
        "final_text": auditor.text,
    }


def audit_with_provider(
    auditor_provider: str,
    *,
    original_prompt: str,
    primary: ProviderResult,
    max_tokens: int,
) -> ProviderResult:
    source_text = "\n".join(
        f"- {item.get('title')}: {item.get('url')}" for item in primary.sources if item.get("url")
    ) or "No hay fuentes estructuradas registradas."
    prompt = f"""
Eres auditor de evidencia. Revisa la respuesta principal sin crear fuentes nuevas.

CONSULTA ORIGINAL
{original_prompt}

RESPUESTA PRINCIPAL
{primary.text}

FUENTES REGISTRADAS POR LA APLICACIÓN
{source_text}

INSTRUCCIONES
1. Identifica afirmaciones que no estén suficientemente respaldadas.
2. Comprueba correspondencia entre afirmación y fuente, sin asumir que una URL respalda todo el texto.
3. Señala referencias incompletas, inconsistentes o dudosas.
4. No agregues autores, DOI, PMID ni URLs nuevos.
5. Entrega: fortalezas, problemas, correcciones obligatorias y veredicto de confiabilidad.
"""
    return call_provider(
        auditor_provider,
        system_prompt=profile_system_prompt("🔎 Revisor Crítico", "Reproducibilidad"),
        user_prompt=prompt,
        pdf_files=[],
        urls=[],
        max_tokens=min(max_tokens, 2500),
        temperature=0.1,
        thinking=auditor_provider in {"DeepSeek", "Kimi"},
        google_search=False,
    )


def run_evidence_strict(
    *,
    profile_name: str,
    task_name: str,
    user_prompt: str,
    pdf_files: Sequence[Any],
    urls: Sequence[str],
    max_tokens: int,
    auditor_provider: str,
) -> Dict[str, Any]:
    if not get_key("Gemini"):
        raise ValueError("Evidencia estricta requiere una clave válida de Gemini.")
    primary = run_single(
        "Gemini",
        profile_name=profile_name,
        task_name=task_name,
        user_prompt=user_prompt,
        pdf_files=pdf_files,
        urls=urls,
        max_tokens=max_tokens,
        evidence_strict=True,
    )
    audit = None
    if auditor_provider != "Ninguno" and get_key(auditor_provider):
        audit = audit_with_provider(
            auditor_provider,
            original_prompt=user_prompt,
            primary=primary,
            max_tokens=max_tokens,
        )
    return {
        "kind": "evidence_strict",
        "primary": asdict(primary),
        "audit": asdict(audit) if audit else None,
        "final_text": primary.text,
    }


def run_compare(
    providers: Sequence[str],
    *,
    profile_name: str,
    task_name: str,
    user_prompt: str,
    pdf_files: Sequence[Any],
    urls: Sequence[str],
    max_tokens: int,
) -> Dict[str, Any]:
    results = []
    for provider in providers:
        result = run_single(
            provider,
            profile_name=profile_name,
            task_name=task_name,
            user_prompt=user_prompt,
            pdf_files=pdf_files,
            urls=urls,
            max_tokens=max_tokens,
            evidence_strict=False,
        )
        results.append(asdict(result))
    return {"kind": "compare", "results": results, "final_text": ""}


def run_max_quality(
    *,
    profile_name: str,
    task_name: str,
    user_prompt: str,
    pdf_files: Sequence[Any],
    urls: Sequence[str],
    max_tokens: int,
) -> Dict[str, Any]:
    primary_provider = auto_provider(profile_name, task_name, user_prompt, pdf_files, urls)
    primary = run_single(
        primary_provider,
        profile_name=profile_name,
        task_name=task_name,
        user_prompt=user_prompt,
        pdf_files=pdf_files,
        urls=urls,
        max_tokens=max_tokens,
        evidence_strict=primary_provider == "Gemini",
    )

    candidates = [p for p in ("OpenAI", "Kimi", "Gemini", "DeepSeek") if p != primary_provider and get_key(p)]
    if not candidates:
        return {"kind": "max_quality", "primary": asdict(primary), "audit": None, "final": None, "final_text": primary.text}

    audit = audit_with_provider(candidates[0], original_prompt=user_prompt, primary=primary, max_tokens=max_tokens)

    integrator_provider = "Gemini" if get_key("Gemini") else candidates[0]
    integration_prompt = f"""
INTEGRA UNA RESPUESTA FINAL CIENTÍFICA.

CONSULTA
{user_prompt}

RESPUESTA PRINCIPAL
{primary.text}

AUDITORÍA
{audit.text}

REGLAS
- Conserva solo afirmaciones sostenibles.
- Corrige contradicciones y errores detectados.
- No agregues referencias nuevas.
- Separa información confirmada, limitaciones y acción prioritaria.
"""
    final = call_provider(
        integrator_provider,
        system_prompt=profile_system_prompt(profile_name, task_name),
        user_prompt=integration_prompt,
        pdf_files=[],
        urls=[],
        max_tokens=max_tokens,
        temperature=0.1,
        thinking=integrator_provider in {"DeepSeek", "Kimi"},
        google_search=False,
    )
    if primary.raw_reference_report:
        final.text = strip_reference_section(final.text) + "\n\n" + references_markdown(primary.raw_reference_report)
    return {
        "kind": "max_quality",
        "primary": asdict(primary),
        "audit": asdict(audit),
        "final": asdict(final),
        "final_text": final.text,
    }


# ============================================================
# EXPORTACIÓN
# ============================================================

def result_markdown(payload: Dict[str, Any], metadata: Dict[str, str]) -> str:
    header = f"""# {APP_NAME}

- Fecha: {datetime.now().isoformat(timespec='seconds')}
- Perfil: {metadata['profile']}
- Tarea: {metadata['task']}
- Modo: {metadata['mode']}

## Consulta

{metadata['prompt']}

"""
    kind = payload.get("kind")
    if kind == "single":
        item = payload["result"]
        return header + f"## Respuesta\n\n{item['text']}\n"
    if kind == "compare":
        chunks = [header]
        for item in payload["results"]:
            chunks.append(f"## {item['provider']} · {item['model']}\n\n{item['text']}\n")
        return "\n".join(chunks)
    if kind == "r_audited":
        return (
            header
            + "## Propuesta original de DeepSeek\n\n"
            + payload["primary"]["text"]
            + "\n\n## Auditoría y versión final de Gemini\n\n"
            + payload["auditor"]["text"]
        )
    if kind == "evidence_strict":
        text = header + "## Respuesta fundamentada\n\n" + payload["primary"]["text"]
        if payload.get("audit"):
            text += "\n\n## Auditoría independiente\n\n" + payload["audit"]["text"]
        return text
    if kind == "max_quality":
        return header + "## Respuesta final integrada\n\n" + payload["final_text"]
    return header + payload.get("final_text", "")


# ============================================================
# BARRA LATERAL
# ============================================================

with st.sidebar:
    st.markdown("## 🔬 Orquestador")
    st.caption(f"{APP_VERSION} · cuatro proveedores opcionales")

    st.selectbox(
        "Apariencia",
        ["Sistema", "Claro", "Oscuro"],
        key="appearance",
        help="El cambio se aplica al recargar la interfaz.",
    )

    st.markdown("### Conexiones")
    for provider, meta in PROVIDERS.items():
        key, source = get_key_with_source(meta["env"])
        status = st.session_state[f"status_{provider}"]
        dot_class = "oc-dot-ok" if status == "ok" else "oc-dot-error" if status == "error" else "oc-dot-off"
        source_text = f" · {source}" if source else ""
        st.markdown(
            f'<span class="oc-chip"><span class="oc-dot {dot_class}"></span>{provider}{source_text}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info("Las claves pegadas en la app se conservan solo durante la sesión y no se escriben en GitHub.")

    with st.expander("Modelos y parámetros"):
        for provider in PROVIDERS:
            options = st.session_state[f"models_{provider}"]
            current = st.session_state[f"model_{provider}"]
            if current not in options:
                options = [current, *options]
            st.selectbox(f"{provider}", options=options, key=f"model_{provider}")
        st.number_input(
            "Máximo de tokens de salida",
            min_value=500,
            max_value=12000,
            value=DEFAULT_MAX_OUTPUT,
            step=250,
            key="max_output_tokens",
        )


# ============================================================
# CABECERA
# ============================================================

status_chips = []
for provider in PROVIDERS:
    configured = bool(get_key(provider))
    status_chips.append(
        f'<span class="oc-chip"><span class="oc-dot {"oc-dot-ok" if configured else "oc-dot-off"}"></span>{provider}</span>'
    )

st.markdown(
    f"""
    <div class="oc-hero">
      <h1>🔬 Orquestador Científico de IA</h1>
      <p>Gemini para evidencia y documentos · DeepSeek para R y estadística · Kimi para contexto largo · OpenAI para auditoría.</p>
      <div style="margin-top:.65rem">{''.join(status_chips)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

main_tab, connections_tab, profiles_tab, guide_tab = st.tabs(
    ["🧪 Área de trabajo", "🔑 Conexiones", "🧠 Perfiles", "📘 Guía"]
)


# ============================================================
# TAB CONEXIONES
# ============================================================

with connections_tab:
    st.subheader("Conexiones y manejo seguro de claves")
    st.markdown(
        """
        <div class="oc-note">
        <strong>Recomendación:</strong> usa el importador múltiple para pegar todas tus claves una sola vez.
        Se ordenan automáticamente por proveedor y quedan solo en la sesión. Para una app pública,
        no conviene guardar las claves del propietario porque todos los visitantes consumirían su cuota.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("### Importar varias claves de una vez")
        st.text_area(
            "Pega un bloque con formato CLAVE=valor",
            key="bulk_keys_input",
            height=180,
            placeholder=(
                "GEMINI_API_KEY=...\n"
                "DEEPSEEK_API_KEY=...\n"
                "MOONSHOT_API_KEY=...\n"
                "OPENAI_API_KEY=..."
            ),
            help="No pegues este bloque en GitHub ni en capturas.",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Importar y ordenar", type="primary", use_container_width=True):
                parsed, ignored = parse_key_bundle(st.session_state.bulk_keys_input)
                if parsed:
                    store_session_keys(parsed)
                    st.success(f"Se importaron {len(parsed)} clave(s).")
                    if ignored:
                        st.warning(f"Se ignoraron {len(ignored)} línea(s) no reconocidas.")
                    st.rerun()
                else:
                    st.error("No se reconocieron claves válidas.")
        with c2:
            if st.button("Quitar claves de sesión", use_container_width=True):
                for meta in PROVIDERS.values():
                    st.session_state[f"session_{meta['env'].lower()}"] = ""
                for provider in PROVIDERS:
                    st.session_state[f"status_{provider}"] = "not_tested"
                    st.session_state[f"message_{provider}"] = "Sin probar"
                st.rerun()

        with st.expander("Formato para Streamlit Secrets"):
            st.code(
                """# Settings → Secrets
ALLOW_SHARED_SECRETS = false

GEMINI_API_KEY = "..."
DEEPSEEK_API_KEY = "..."
MOONSHOT_API_KEY = "..."
OPENAI_API_KEY = "..."
""",
                language="toml",
            )
            st.caption(
                "Mantén ALLOW_SHARED_SECRETS=false en una app pública. Actívalo solo en una app privada o de uso personal."
            )

    with right:
        st.markdown("### Estado por proveedor")
        for provider, meta in PROVIDERS.items():
            key, source = get_key_with_source(meta["env"])
            with st.container(border=True):
                top1, top2 = st.columns([1, 1])
                with top1:
                    st.markdown(f"**{meta['icon']} {provider}**")
                    st.caption(meta["specialty"])
                with top2:
                    st.caption(f"{mask_key(key)}" + (f" · {source}" if source else ""))
                b1, b2 = st.columns(2)
                with b1:
                    if st.button(
                        "Probar",
                        key=f"test_{provider}",
                        use_container_width=True,
                        disabled=not bool(key),
                    ):
                        with st.spinner(f"Probando {provider}…"):
                            try:
                                result = test_provider(provider)
                                st.session_state[f"status_{provider}"] = "ok"
                                st.session_state[f"message_{provider}"] = f"Conectado con {result.model}"
                                st.success(st.session_state[f"message_{provider}"])
                            except Exception as exc:
                                st.session_state[f"status_{provider}"] = "error"
                                st.session_state[f"message_{provider}"] = classify_error(provider, exc)
                                st.error(st.session_state[f"message_{provider}"])
                with b2:
                    if st.button(
                        "Sincronizar modelos",
                        key=f"models_button_{provider}",
                        use_container_width=True,
                        disabled=not bool(key),
                    ):
                        with st.spinner("Consultando modelos disponibles…"):
                            try:
                                models = list_provider_models(provider)
                                if models:
                                    st.session_state[f"models_{provider}"] = models
                                    st.success(f"{len(models)} modelo(s) disponibles.")
                                    st.rerun()
                            except Exception as exc:
                                st.error(classify_error(provider, exc))
                status_message = st.session_state[f"message_{provider}"]
                st.caption(status_message)


# ============================================================
# TAB PERFILES
# ============================================================

with profiles_tab:
    st.subheader("Seis perfiles científicos potenciados")
    for name, profile in PROFILES.items():
        with st.expander(name):
            st.write(profile["short"])
            st.caption(profile["engine"])
            st.markdown("**Tareas disponibles:**")
            st.write(" · ".join(profile["tasks"].keys()))
            with st.expander("Ver prompt profesional"):
                st.code(profile["prompt"].strip() + "\n\n" + COMMON_RULES, language="text")


# ============================================================
# TAB GUÍA
# ============================================================

with guide_tab:
    st.subheader("Arquitectura recomendada")
    st.markdown(
        """
        **Automático:** la app elige el motor según la tarea y las claves disponibles.

        **Código R auditado:** DeepSeek genera y Gemini revisa diseño, sintaxis, paquetes, supuestos e interpretación.

        **Evidencia estricta:** Gemini usa Google Search y URL Context; luego la app verifica DOI/PMID en
        Crossref, OpenAlex y PubMed. Una segunda IA puede auditar la correspondencia entre afirmaciones y fuentes.

        **Comparar dos IAs:** muestra respuestas independientes, sin ocultar desacuerdos.

        **Máxima calidad:** genera, audita e integra. Consume más cuota.
        """
    )
    st.warning(
        "Una segunda IA no convierte automáticamente una referencia en verdadera. "
        "La comprobación bibliográfica se realiza con portales estructurados y se separa de la calidad/indexación de la revista."
    )


# ============================================================
# ÁREA DE TRABAJO
# ============================================================

with main_tab:
    row1, row2 = st.columns([1, 1], gap="large")
    with row1:
        profile_name = st.selectbox("Perfil científico", list(PROFILES.keys()))
    with row2:
        task_name = st.selectbox("Tipo de tarea", list(PROFILES[profile_name]["tasks"].keys()))

    profile = PROFILES[profile_name]
    st.markdown(
        f"""
        <div class="oc-card">
          <h3>{profile_name}</h3>
          <p>{profile['short']}</p>
          <span class="oc-chip">{profile['engine']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Modo de ejecución")
    mode = st.radio(
        "Selecciona el flujo",
        [
            "Automático",
            "Una IA",
            "Código R auditado",
            "Evidencia estricta",
            "Comparar dos IAs",
            "Máxima calidad",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_provider = None
    compare_providers: List[str] = []
    auditor_provider = "Ninguno"
    configured = configured_providers()

    if mode == "Una IA":
        selected_provider = st.selectbox("Proveedor", list(PROVIDERS.keys()))
    elif mode == "Comparar dos IAs":
        compare_providers = st.multiselect(
            "Selecciona exactamente dos proveedores",
            list(PROVIDERS.keys()),
            default=configured[:2],
            max_selections=2,
        )
    elif mode == "Evidencia estricta":
        audit_options = ["Ninguno"] + [p for p in ("OpenAI", "Kimi", "DeepSeek") if p in configured]
        auditor_provider = st.selectbox("Auditor independiente opcional", audit_options)

    st.markdown("### Fuentes de entrada")
    source1, source2 = st.columns(2, gap="large")
    with source1:
        pdf_files = st.file_uploader(
            "Adjuntar PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Hasta {MAX_PDFS} archivos; máximo recomendado {MAX_PDF_MB} MB por archivo.",
        )
    with source2:
        raw_urls = st.text_area(
            "Enlaces públicos o YouTube",
            height=120,
            placeholder="https://...\nhttps://youtu.be/...",
        )
        urls, invalid_urls = normalize_urls(raw_urls)
        if invalid_urls:
            st.warning("Enlaces inválidos: " + ", ".join(invalid_urls))

    pdf_files = list(pdf_files or [])[:MAX_PDFS]
    oversize = [f.name for f in pdf_files if len(f.getvalue()) > MAX_PDF_MB * 1024 * 1024]
    if oversize:
        st.error("Archivos demasiado grandes: " + ", ".join(oversize))

    user_prompt = st.text_area(
        "Consulta científica",
        height=240,
        placeholder="Describe el objetivo, datos disponibles, restricciones y formato esperado.",
    )

    with st.expander("Verificación bibliográfica avanzada"):
        st.text_input(
            "Correo para el polite pool de Crossref",
            key="crossref_mailto",
            placeholder="correo@institucion.edu",
            help="Opcional. Crossref recomienda identificar solicitudes automatizadas.",
        )
        st.text_input(
            "NCBI API key",
            key="ncbi_api_key",
            type="password",
            help="Opcional. No es necesaria para consultas pequeñas de PubMed.",
        )

    run_disabled = not bool(user_prompt.strip()) or bool(oversize)
    if mode == "Código R auditado":
        run_disabled = run_disabled or not (get_key("DeepSeek") and get_key("Gemini"))
    elif mode == "Evidencia estricta":
        run_disabled = run_disabled or not get_key("Gemini")
    elif mode == "Comparar dos IAs":
        run_disabled = run_disabled or len(compare_providers) != 2 or any(not get_key(p) for p in compare_providers)
    elif mode == "Una IA":
        run_disabled = run_disabled or not get_key(selected_provider or "")
    elif mode in {"Automático", "Máxima calidad"}:
        run_disabled = run_disabled or not bool(configured)

    if st.button(
        "Ejecutar análisis científico",
        type="primary",
        use_container_width=True,
        disabled=run_disabled,
    ):
        metadata = {
            "profile": profile_name,
            "task": task_name,
            "mode": mode,
            "prompt": user_prompt,
        }
        try:
            with st.spinner("Analizando, verificando y preparando trazabilidad…"):
                max_tokens = int(st.session_state.max_output_tokens)
                if mode == "Código R auditado":
                    payload = run_r_audited(
                        profile_name=profile_name,
                        task_name=task_name,
                        user_prompt=user_prompt,
                        pdf_files=pdf_files,
                        urls=urls,
                        max_tokens=max_tokens,
                    )
                elif mode == "Evidencia estricta":
                    payload = run_evidence_strict(
                        profile_name=profile_name,
                        task_name=task_name,
                        user_prompt=user_prompt,
                        pdf_files=pdf_files,
                        urls=urls,
                        max_tokens=max_tokens,
                        auditor_provider=auditor_provider,
                    )
                elif mode == "Comparar dos IAs":
                    payload = run_compare(
                        compare_providers,
                        profile_name=profile_name,
                        task_name=task_name,
                        user_prompt=user_prompt,
                        pdf_files=pdf_files,
                        urls=urls,
                        max_tokens=max_tokens,
                    )
                elif mode == "Máxima calidad":
                    payload = run_max_quality(
                        profile_name=profile_name,
                        task_name=task_name,
                        user_prompt=user_prompt,
                        pdf_files=pdf_files,
                        urls=urls,
                        max_tokens=max_tokens,
                    )
                else:
                    provider = selected_provider if mode == "Una IA" else auto_provider(
                        profile_name, task_name, user_prompt, pdf_files, urls
                    )
                    strict = mode == "Automático" and profile_name == "📚 Revisor de Literatura"
                    result = run_single(
                        provider,
                        profile_name=profile_name,
                        task_name=task_name,
                        user_prompt=user_prompt,
                        pdf_files=pdf_files,
                        urls=urls,
                        max_tokens=max_tokens,
                        evidence_strict=strict,
                    )
                    payload = {"kind": "single", "result": asdict(result), "final_text": result.text}
                st.session_state.last_result = {"payload": payload, "metadata": metadata}
                st.success("Análisis completado.")
        except Exception as exc:
            st.error(sanitize_error(exc))

    if st.session_state.last_result:
        saved = st.session_state.last_result
        payload = saved["payload"]
        metadata = saved["metadata"]
        st.divider()
        st.markdown("## Resultado")

        result_tab, sources_tab, trace_tab, export_tab = st.tabs(
            ["Respuesta final", "Fuentes", "Trazabilidad", "Exportar"]
        )

        with result_tab:
            kind = payload.get("kind")
            if kind == "single":
                st.markdown(payload["result"]["text"])
            elif kind == "compare":
                cols = st.columns(len(payload["results"]))
                for col, item in zip(cols, payload["results"]):
                    with col:
                        st.markdown(f"### {item['provider']}")
                        st.caption(item["model"])
                        st.markdown(item["text"])
            elif kind == "r_audited":
                st.markdown(payload["auditor"]["text"])
                with st.expander("Ver propuesta original de DeepSeek"):
                    st.markdown(payload["primary"]["text"])
            elif kind == "evidence_strict":
                st.markdown(payload["primary"]["text"])
                if payload.get("audit"):
                    with st.expander("Auditoría independiente"):
                        st.markdown(payload["audit"]["text"])
            elif kind == "max_quality":
                st.markdown(payload["final_text"])
                with st.expander("Ver generación y auditoría"):
                    st.markdown("### Respuesta principal")
                    st.markdown(payload["primary"]["text"])
                    if payload.get("audit"):
                        st.markdown("### Auditoría")
                        st.markdown(payload["audit"]["text"])

        with sources_tab:
            source_items: List[Dict[str, str]] = []
            if payload.get("kind") == "single":
                source_items = payload["result"].get("sources", [])
            elif payload.get("primary"):
                source_items = payload["primary"].get("sources", [])
            elif payload.get("results"):
                for item in payload["results"]:
                    source_items.extend(item.get("sources", []))
            source_items = _dedupe_sources(source_items)
            if source_items:
                for item in source_items:
                    url = item.get("url")
                    if url:
                        st.markdown(f"- **{item.get('type')}**: [{item.get('title')}]({url})")
                    else:
                        st.markdown(f"- **{item.get('type')}**: {item.get('title')}")
            else:
                st.info("No se registraron fuentes externas estructuradas para este resultado.")

        with trace_tab:
            items: List[Dict[str, Any]] = []
            if payload.get("kind") == "single":
                items = [payload["result"]]
            elif payload.get("kind") == "compare":
                items = payload["results"]
            else:
                for key in ("primary", "auditor", "audit", "final"):
                    if payload.get(key):
                        items.append(payload[key])
            for item in items:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Proveedor", item.get("provider", ""))
                c2.metric("Modelo", item.get("model", ""))
                c3.metric("Tokens", item.get("usage", {}).get("total_tokens", 0))
                c4.metric("Tiempo", f"{item.get('elapsed', 0)} s")
                with st.expander(f"Intentos · {item.get('provider', '')}"):
                    st.json(item.get("attempts", []))

        with export_tab:
            md = result_markdown(payload, metadata)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Descargar Markdown",
                    data=md.encode("utf-8"),
                    file_name=f"orquestador_{stamp}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Descargar JSON de trazabilidad",
                    data=json.dumps(saved, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"orquestador_{stamp}.json",
                    mime="application/json",
                    use_container_width=True,
                )

st.caption(
    "La aplicación organiza y audita trabajo científico, pero no sustituye la lectura de las fuentes, "
    "la revisión por pares ni el criterio metodológico profesional."
)
