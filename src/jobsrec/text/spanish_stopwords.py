"""Spanish and project-specific stopwords for Apolo vectorjobs.

This module provides stopword lists for filtering Spanish connector words and
common job-ad boilerplate, helping prevent them from leaking into TF-IDF
cluster labels and top terms.
"""

from __future__ import annotations

# Common Spanish grammatical connectors, pronouns, prepositions, etc.
SPANISH_STOPWORDS: frozenset[str] = frozenset([
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "y", "o", "para", "por", "con", "en", "al", "a", "ante", "bajo", "cabe", "contra",
    "desde", "durante", "entre", "hacia", "hasta", "mediante", "según", "sin", "so", "sobre", "tras", "versus", "vía",
    "se", "que", "como", "sus", "su", "tu", "tus", "mi", "mis", "nuestro", "nuestra", "nuestros", "nuestras",
    "es", "son", "será", "ser", "estar", "tener", "ha", "han", "hay", "he", "hemos", "había", "habían", "este", "esta",
    "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas",
    "lo", "le", "les", "me", "te", "nos", "os", "yo", "tú", "él", "ella", "ellos", "ellas", "nosotros", "nosotras", "vosotros", "vosotras",
    "sí", "no", "pero", "más", "muy", "también", "tampoco", "otro", "otra", "otros", "otras",
    "todo", "toda", "todos", "todas", "cada", "mismo", "misma", "mismos", "mismas",
    "cual", "cuales", "quien", "quienes", "cuyo", "cuya", "cuyos", "cuyas",
    "donde", "cuando", "como", "cuanto", "cuanta", "cuantos", "cuantas",
    "entonces", "así", "luego", "después", "antes", "ahora", "hoy", "ayer", "mañana",
    "aquí", "allí", "allá", "ahí", "cerca", "lejos", "arriba", "abajo", "dentro", "fuera",
    "bien", "mal", "ya", "todavía", "aun", "aunque", "sino", "porque", "pues", "mientras",
])

# Job posting specific boilerplate words (e.g. search, requirements, profile, etc.)
APOLO_DOMAIN_STOPWORDS: frozenset[str] = frozenset([
    "buscamos", "requiere", "ofrecemos", "empresa", "cargo", "funciones", "requisitos",
    "vacante", "puesto", "empleo", "trabajo", "postularse", "postular", "selección", "reclutamiento",
    "contrato", "contratación", "jornada", "completa", "parcial", "remoto", "híbrido", "presencial",
    "experiencia", "año", "años", "mes", "meses", "salario", "sueldo", "beneficios",
    "habilidad", "habilidades", "conocimiento", "conocimientos", "requisito", "perfil", "candidato",
    "candidata", "candidatos", "candidatas", "equipo", "proceso", "proyecto", "proyectos", "cliente",
    "clientes", "servicio", "servicios", "desarrollo", "profesional", "oportunidad", "oportunidades",
    "interesados", "enviar", "cv", "correo", "asunto", "postula", "únete", "cvs", "postulaciones",
    "incorporación", "inmediata", "disponibilidad", "horaria", "nacional", "internacional",
    "zona", "lugar", "ubicación", "ciudad", "país", "oficina", "oficinas", "sector", "área", "departamento",
    "posición", "rol", "tareas", "responsabilidades", "diseñar", "implementar", "mantener", "soporte",
    "colaborar", "participar", "liderar", "gestionar", "administrar", "coordinar", "apoyar", "ayudar",
    "realizar", "ejecutar", "analizar", "evaluar", "reportar", "presentar", "elaborar", "redactar",
])

# Full unified stopword list for Apolo text vectorization
APOLO_STOPWORDS: frozenset[str] = SPANISH_STOPWORDS | APOLO_DOMAIN_STOPWORDS
