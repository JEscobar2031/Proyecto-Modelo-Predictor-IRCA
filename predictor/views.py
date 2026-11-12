"""
Vistas de la aplicación de predicción IRCA.
Este archivo maneja la lógica de negocio principal:
1. Carga inicial de modelos entrenados de Machine Learning (.pkl) usando Joblib.
2. Definición de conjuntos de variables para CDA (6 variables básicas) y CDB (9 variables ampliadas).
3. Renderizado de vistas protegidas por autenticación (home y formulario de predicción).
4. Procesamiento de predicciones en tiempo real, cálculo de probabilidades, importancia de características y estimación de IRCA.
"""
import joblib
import numpy as np
from pathlib import Path
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent

# Diccionario para almacenar en memoria los modelos de Machine Learning cargados desde archivos .pkl
MODELS = {}
MODEL_FILES = {
    'rf_cda': 'modelo_rf_cda.pkl',
    'rf_cdb': 'modelo_rf_cdb.pkl',
    'lr_cda': 'modelo_lr_cda.pkl',
    'lr_cdb': 'modelo_lr_cdb.pkl',
}

# Carga de modelos al iniciar la aplicación Django
for key, filename in MODEL_FILES.items():
    model_path = MODELS_DIR / filename
    if model_path.exists():
        MODELS[key] = joblib.load(model_path)

# Definición de variables de entrada requeridas por cada modelo (CDA: conjunto básico de 6 variables, CDB: conjunto ampliado de 9 variables)
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

# Metadatos descriptivos de los modelos para la interfaz web
MODEL_INFO = {
    'rf_cda': {'name': 'Random Forest CDA', 'desc': 'Modelo reducido con 6 variables', 'vars': 'cda', 'accuracy': 0.999975},
    'rf_cdb': {'name': 'Random Forest CDB', 'desc': 'Modelo ampliado con 9 variables', 'vars': 'cdb', 'accuracy': 0.9999625},
    'lr_cda': {'name': 'Regresión Logística CDA', 'desc': 'Modelo reducido con 6 variables', 'vars': 'cda', 'accuracy': 0.8663375},
    'lr_cdb': {'name': 'Regresión Logística CDB', 'desc': 'Modelo ampliado con 9 variables', 'vars': 'cdb', 'accuracy': 0.8663375},
}


def register(request):
    """Permite crear una cuenta local para acceder al predictor."""
    if request.user.is_authenticated:
        return redirect('predictor:home')

    form = UserCreationForm(request.POST or None)
    form.fields['username'].label = 'Usuario'
    form.fields['password1'].label = 'Contraseña'
    form.fields['password2'].label = 'Confirmar contraseña'
    form.fields['password1'].help_text = ''
    form.fields['password2'].help_text = ''
    for field in form.fields.values():
        field.widget.attrs['class'] = 'form-control'

    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('predictor:login')

    return render(request, 'predictor/register.html', {'form': form})


@login_required
def predictor_home(request):
    """Vista principal del panel de control del predictor IRCA (requiere autenticación)."""
    return render(request, 'predictor/home.html', {
        'models': MODEL_INFO,
    })


@login_required
def predict(request):
    """
    Vista encargada de gestionar el formulario de predicción y procesar los datos ingresados.
    - Método GET: Muestra el formulario con los campos correspondientes al modelo seleccionado.
    - Método POST: Recoge los valores, ejecuta la inferencia con el modelo seleccionado, calcula probabilidades,
      extrae la importancia de características (si aplica) y calcula el IRCA normativo.
    """
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

        # Construcción del arreglo numpy para la inferencia del modelo
        X = np.array([values])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Extracción de importancia de características (Feature Importances) para modelos basados en árboles (Random Forest)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = sorted(
                zip(var_names, importances),
                key=lambda x: x[1],
                reverse=True
            )

        # Cálculo complementario del IRCA basado en umbrales de la Resolución 2115 de 2007
        irca = calculate_irca(values, MODEL_INFO[model_key]['vars'])

        # Mapeo de la etiqueta de predicción y cálculo del porcentaje de confianza
        if isinstance(prediction, str):
            pred_label = prediction
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
    """
    Función auxiliar para calcular el porcentaje aproximado de riesgo IRCA 
    según los criterios y ponderaciones establecidos en la Resolución 2115 de 2007.
    Evalúa si cada parámetro excede los límites permisibles y acumula puntos de riesgo.
    """
    risk_score = 0
    max_score = 0

    if var_type == 'cda':
        checks = [
            (values[0] < 6.5 or values[0] > 8.5, 15),
            (values[1] > 5, 20),
            (values[2] > 15, 15),
            (values[3] < 0.2 or values[3] > 2, 15),
            (values[4] > 0, 20),
            (values[5] > 0, 15),
        ]
    else:
        checks = [
            (values[0] < 6.5 or values[0] > 8.5, 12),
            (values[1] > 5, 16),
            (values[2] > 15, 12),
            (values[3] < 0.2 or values[3] > 2, 12),
            (values[4] > 0, 16),
            (values[5] > 0, 12),
            (values[6] > 30, 8),
            (values[7] > 1000, 6),
            (values[8] > 80, 6),
        ]

    for condition, score in checks:
        max_score += score
        if condition:
            risk_score += score

    return round((risk_score / max_score) * 100, 2) if max_score > 0 else 0
