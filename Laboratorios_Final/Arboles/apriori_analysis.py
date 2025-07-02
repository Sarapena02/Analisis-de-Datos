# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df = pd.read_csv('CBC.csv')

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
print(df.head())

# Preparar los datos para el algoritmo Apriori
# Seleccionar solo las columnas de libros
book_columns = ['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 
                'ArtBks', 'GeoBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence']

# Convertir los datos a formato binario (0 y 1)
df_binary = df[book_columns].copy()
df_binary = df_binary.astype(bool).astype(int)

# Mostrar las primeras filas del dataset binario
print("\nDataset binario para análisis de reglas de asociación:")
print(df_binary.head())

# Aplicar el algoritmo Apriori con soporte mínimo de 0.05
frequent_itemsets = apriori(df_binary, min_support=0.05, use_colnames=True)

# Mostrar los itemsets frecuentes
print("\nITEMSETS FRECUENTES")
print("\n" + "="*100)
print("CONJUNTO DE PRODUCTOS | SOPORTE")
print("-"*100)
print(frequent_itemsets.to_string())
print("="*100)

# Generar reglas de asociación con confianza mínima de 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Configurar opciones de visualización de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Mostrar las reglas de asociación en formato tabla
print("\nREGLAS DE ASOCIACIÓN")
print("\n" + "="*120)
print("ANTECEDENTES (SI COMPRA) -> CONSECUENTES (ENTONCES COMPRA) | SOPORTE | CONFIANZA | LIFT")
print("-"*120)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())
print("="*120)

# Identificar reglas inútiles usando múltiples criterios
print("\nANÁLISIS DE REGLAS INÚTILES")
print("\n" + "="*120)

# 1. Reglas con lift <= 1 (no hay asociación positiva)
lift_useless = rules[rules['lift'] <= 1]
print("\n1. Reglas sin asociación positiva (lift <= 1):")
print("-"*120)
print(lift_useless[['antecedents', 'consequents', 'lift']].to_string())

# 2. Reglas con soporte muy bajo (menos del 5%)
low_support = rules[rules['support'] < 0.05]
print("\n2. Reglas con soporte muy bajo (< 5%):")
print("-"*120)
print(low_support[['antecedents', 'consequents', 'support']].to_string())

# 3. Reglas con confianza muy baja (menos del 50%)
low_confidence = rules[rules['confidence'] < 0.5]
print("\n3. Reglas con confianza muy baja (< 50%):")
print("-"*120)
print(low_confidence[['antecedents', 'consequents', 'confidence']].to_string())

# 4. Reglas duplicadas o redundantes
duplicate_rules = rules[rules.duplicated(['antecedents', 'consequents'], keep=False)]
print("\n4. Reglas duplicadas o redundantes:")
print("-"*120)
print(duplicate_rules[['antecedents', 'consequents']].to_string())

# Combinar todas las reglas inútiles
all_useless_rules = pd.concat([
    lift_useless,
    low_support,
    low_confidence,
    duplicate_rules
]).drop_duplicates()

print("\nRESUMEN DE REGLAS INÚTILES:")
print("-"*120)
print(f"Total de reglas inútiles encontradas: {len(all_useless_rules)}")
print(f"Razones de descarte:")
print(f"- Reglas sin asociación positiva (lift <= 1): {len(lift_useless)}")
print(f"- Reglas con soporte muy bajo: {len(low_support)}")
print(f"- Reglas con confianza muy baja: {len(low_confidence)}")
print(f"- Reglas duplicadas: {len(duplicate_rules)}")

# Filtrar para obtener solo las reglas útiles
useful_rules = rules[~rules.index.isin(all_useless_rules.index)]

# Ordenar por lift para ver las reglas más fuertes
useful_rules = useful_rules.sort_values('lift', ascending=False)

print("\nANÁLISIS DE REGLAS ÚTILES Y RECOMENDACIONES DE NEGOCIO")
print("\n" + "="*120)
print("REGLAS MÁS FUERTES (ORDENADAS POR LIFT):")
print("-"*120)
print(useful_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())
print("="*120)

print("\nRECOMENDACIONES DE NEGOCIO:")
print("\n1. ESTRATEGIAS DE CROSS-SELLING:")
for idx, rule in useful_rules.head(5).iterrows():
    print(f"   • Si un cliente compra {rule['antecedents']}, recomendar {rule['consequents']}")
    print(f"     (Confianza: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f})")

print("\n2. UBICACIÓN EN TIENDA:")
for idx, rule in useful_rules.head(3).iterrows():
    print(f"   • Colocar {rule['antecedents']} cerca de {rule['consequents']}")
    print(f"     (Soporte: {rule['support']:.2%})")

print("\n3. PAQUETES Y PROMOCIONES:")
for idx, rule in useful_rules.head(3).iterrows():
    print(f"   • Crear paquete combinando {rule['antecedents']} con {rule['consequents']}")
    print(f"     (Lift: {rule['lift']:.2f})")

print("\n4. RECOMENDACIONES PERSONALIZADAS:")
print("   • Implementar un sistema de recomendación basado en las reglas anteriores")
print("   • Priorizar las recomendaciones con mayor lift y confianza")

# Guardar las reglas útiles en un archivo CSV
useful_rules.to_csv('reglas_utiles.csv', index=False)

print("\nLas reglas útiles han sido guardadas en 'reglas_utiles.csv'") 