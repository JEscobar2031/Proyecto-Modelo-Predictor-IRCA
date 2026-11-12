import os
import joblib
import numpy as np
from pathlib import Path
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent

# Cargar modelos al iniciar la app
MODELS = {}
MODEL_FILES = {
    'rf_cda': 'modelo_rf_cda.pkl',
    'rf_cdb': 'modelo_rf_cdb.pkl',
    'lr_cda': 'modelo_lr_cda.pkl',
    'lr_cdb': 'modelo_lr_cdb.pkl',
}

for key, filename in MODEL_FILES.items():
    model_path = MODELS_DIR / filename
    if model_path.exists():
        MODELS[key] = joblib.load(model_path)

# Variables por modelo
MODEL_VARIABLES = {
    'cda': [
        ('pH', 'pH', 0, 14, 7.0),
        ('Turbiedad_UNT', 'Turbiedad (UNT)', 0, 100, 5.0),
        ('Color_UPC', 'Color (UPC)', 0, 100, 10.0),
        ('Cloro_residual_mgL', 'Cloro residual (mg/L)', 0, 5, 0.5),
        ('Coliformes_totales_UFC_100mL', 'Coliformes totales (UFC/100mL)', 0, 10000, 100),
        ('Ecoli_UFC_100mL', 'E. coli (UFC/100mL)', 0, 10000, 50),
    ],
    'cdb': [
        ('pH', 'pH', 0, 14, 7.0),
        ('Turbiedad_UNT', 'Turbiedad (UNT)', 0, 100, 5.0),
        ('Color_UPC', 'Color (UPC)', 0, 100, 10.0),
        ('Cloro_residual_mgL', 'Cloro residual (mg/L)', 0, 5, 0.5),
        ('Coliformes_totales_UFC_100mL', 'Coliformes totales (UFC/100mL)', 0, 10000, 100),
        ('Ecoli_UFC_100mL', 'E. coli (UFC/100mL)', 0, 10000, 50),
        ('Temperatura_C', 'Temperatura (°C)', 0, 50, 20.0),
        ('Conductividad_uS_cm', 'Conductividad (µS/cm)', 0, 2000, 500),
        ('THMs_ppb', 'THMs (ppb)', 0, 500, 50),
    ],
}

RISK_LEVELS = ['Sin riesgo', 'Riesgo bajo', 'Riesgo medio', 'Riesgo alto', 'Inviable sanitariamente']

MODEL_INFO = {
    'rf_cda': {'name': 'Random Forest CDA', 'desc': 'Modelo reducido con 6 variables', 'vars': 'cda', 'accuracy': 0.999975},
    'rf_cdb': {'name': 'Random Forest CDB', 'desc': 'Modelo ampliado con 9 variables', 'vars': 'cdb', 'accuracy': 0.9999625},
    'lr_cda': {'name': 'Regresión Logística CDA', 'desc': 'Modelo reducido con 6 variables', 'vars': 'cda', 'accuracy': 0.8663375},
    'lr_cdb': {'name': 'Regresión Logística CDB', 'desc': 'Modelo ampliado con 9 variables', 'vars': 'cdb', 'accuracy': 0.8663375},
}


@login_required
def predictor_home(request):
    """Vista principal del predictor IRCA."""
    return render(request, 'predictor/home.html', {
        'models': MODEL_INFO,
    })


@login_required
def predict(request):
    """Vista para realizar predicciones."""
    if request.method != 'POST':
        selected = request.GET.get('model', 'rf_cda')
        variables = MODEL_VARIABLES[MODEL_INFO[selected]['vars']] if selected in MODEL_INFO else []
        return render(request, 'predictor/predict.html', {
            'models': MODEL_INFO,
            'variables': variables,
            'selected_model': selected,
        })

    model_key = request.POST.get('model', 'rf_cda')
    if model_key not in MODELS:
        return render(request, 'predictor/predict.html', {
            'models': MODEL_INFO,
            'variables': {},
            'selected_model': None,
            'error': 'Modelo no encontrado',
        })

    model = MODELS[model_key]
    var_names = [v[0] for v in MODEL_VARIABLES[MODEL_INFO[model_key]['vars']]]

    try:
        values = []
        for var in var_names:
            val = float(request.POST.get(var, 0))
            values.append(val)

        X = np.array([values])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Obtener feature importances si está disponible
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(
                zip(var_names, importances),
                key=lambda x: x[1],
                reverse=True
            )

        # Calcular IRCA aproximado
        irca = calculate_irca(values, MODEL_INFO[model_key]['vars'])

        # Manejar predicciones que son strings directamente
        if isinstance(prediction, str):
            pred_label = prediction
            # Obtener probabilidad de la clase predicha
            class_idx = list(model.classes_).index(prediction)
            pred_confidence = probabilities[class_idx] * 100
        else:
            pred_label = RISK_LEVELS[prediction] if prediction < len(RISK_LEVELS) else f'Clase {prediction}'
            pred_confidence = max(probabilities) * 100

        result = {
            'prediction': pred_label,
            'confidence': pred_confidence,
            'all_probabilities': [
                {'level': cls, 'prob': p * 100}
                for cls, p in zip(model.classes_, probabilities)
            ],
            'feature_importance': feature_importance,
            'irca': irca,
            'values': {var: val for var, val in zip(var_names, values)},
        }

    except Exception as e:
        result = {'error': str(e)}

    variables = MODEL_VARIABLES[MODEL_INFO[model_key]['vars']]
    return render(request, 'predictor/predict.html', {
        'models': MODEL_INFO,
        'variables': variables,
        'selected_model': model_key,
        'result': result,
    })


def calculate_irca(values, var_type):
    """Calcula un porcentaje de riesgo basado en los valores ingresados."""
    # Cálculo simplificado basado en rangos típicos
    risk_score = 0
    max_score = 0

    if var_type == 'cda':
        checks = [
            (values[0] < 6.5 or values[0] > 8.5, 15),  # pH fuera de rango
            (values[1] > 5, 20),  # Turbiedad
            (values[2] > 15, 15),  # Color
            (values[3] < 0.2 or values[3] > 2, 15),  # Cloro
            (values[4] > 0, 20),  # Coliformes
            (values[5] > 0, 15),  # E. coli
        ]
    else:
        checks = [
            (values[0] < 6.5 or values[0] > 8.5, 12),  # pH
            (values[1] > 5, 16),  # Turbiedad
            (values[2] > 15, 12),  # Color
            (values[3] < 0.2 or values[3] > 2, 12),  # Cloro
            (values[4] > 0, 16),  # Coliformes
            (values[5] > 0, 12),  # E. coli
            (values[6] > 30, 8),  # Temperatura
            (values[7] > 1000, 6),  # Conductividad
            (values[8] > 80, 6),  # THMs
        ]

    for condition, score in checks:
        max_score += score
        if condition:
            risk_score += score

    return round((risk_score / max_score) * 100, 2) if max_score > 0 else 0
