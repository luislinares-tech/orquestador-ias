# Orquestador Científico de IA · Fase 3

Aplicación Streamlit para trabajo científico con cuatro proveedores opcionales:

- **Gemini**: búsqueda fundamentada, PDFs, enlaces, YouTube y metodología.
- **DeepSeek**: código R, Python, estadística y depuración.
- **Kimi**: contexto largo, síntesis y segunda lectura.
- **OpenAI**: auditoría crítica e integración opcional.

## Funciones principales

- Seis perfiles científicos especializados.
- Interfaz profesional con modo claro, oscuro y sistema.
- Importación múltiple de claves en un solo bloque.
- Claves de sesión, `.env` local y `st.secrets` opcional.
- Enrutamiento automático.
- Flujo **Código R auditado**: DeepSeek genera y Gemini revisa.
- Flujo **Evidencia estricta**: Gemini usa Google Search/URL Context y la app verifica DOI/PMID.
- Verificación en Crossref, OpenAlex y PubMed/NCBI.
- Comparación de dos IAs.
- Modo de máxima calidad.
- PDF, enlaces públicos y YouTube.
- Exportación Markdown y JSON de trazabilidad.

## Instalación local

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Importar varias claves en la interfaz

Pega en **Conexiones → Importar varias claves de una vez**:

```text
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
MOONSHOT_API_KEY=...
OPENAI_API_KEY=...
```

La aplicación reconoce los nombres, ordena las claves por proveedor y las conserva únicamente durante la sesión.

## Persistencia segura

### Uso local

Copia `.env.example` como `.env`. `.env` ya está excluido por `.gitignore`.

### Streamlit Community Cloud

Usa **App settings → Secrets**. Nunca subas `secrets.toml` a GitHub.

Para una aplicación pública se recomienda:

```toml
ALLOW_SHARED_SECRETS = false
```

Con esa configuración, cada usuario debe ingresar sus propias claves. Si activas claves compartidas, cualquier visitante podría consumir la cuota del propietario.

## Actualización desde Fase 2

Reemplaza en el mismo repositorio:

- `app.py`
- `requirements.txt`
- `README.md`
- `.env.example`
- `.gitignore`
- `.streamlit/config.toml`

Streamlit Community Cloud detectará los cambios en GitHub y reconstruirá la app porque cambió `requirements.txt`.

## Verificación de referencias

La aplicación distingue:

- existencia bibliográfica, verificada por DOI/PMID;
- fuente web recuperada;
- indexación y calidad de revista.

La presencia de un DOI no demuestra que una revista esté indexada en Scopus o Web of Science ni cuál sea su SJR. Esos indicadores deben comprobarse por separado en los portales oficiales.
