# Orquestador de IAs · Modo Ciencia · Fase 2

## Incluye

- Seis perfiles científicos especializados.
- Gemini mediante el SDK oficial `google-genai`.
- Fallback entre `gemini-3.1-flash-lite` y `gemini-3.5-flash`.
- Entrada por texto, PDFs, URLs públicas, PDFs por URL y videos públicos de YouTube.
- Tareas específicas para protocolos PRISMA-P, revisión sistemática, scoping review,
  bibliometría, metodología, asesoría, estadística/R, edición y revisión por pares.
- Exportación en Markdown y TXT.

## Actualizar desde la Fase 1

No crees otra carpeta ni otro entorno virtual. Detén Streamlit con `Ctrl + C`, reemplaza
`app.py` y vuelve a ejecutar:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Las dependencias son las mismas que en la Fase 1. Si una función nueva produjera un error
por una versión antigua del SDK, actualiza una sola vez:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "google-genai>=1.0,<2.0" "streamlit>=1.41,<2.0"
```

## Seguridad

No compartas claves API. La clave de sesión se mantiene temporalmente en memoria. Para uso
local persistente, copia `.env.example` como `.env` y coloca allí una clave nueva.

## Fase 3

DeepSeek será el motor principal para código R y Gemini funcionará como respaldo y revisor.
