# rules.py — LOGICA CONVENCIONAL (no es red neuronal)
CONTENEDORES = {
    "botella_plastica": "Blanco (aprovechables)",
    "lata":             "Blanco (aprovechables)",
    "papel_carton":     "Blanco (aprovechables)",
    "no_identificado":  "Negro (no aprovechables)",
}

def recomendar(clase, confianza, umbral=0.60):
    if confianza < umbral:
        clase = "no_identificado"
    return clase, CONTENEDORES[clase]