// =================================================================
// REGRESIÓN LINEAL EN SCILAB - DATASET ENERGY EFFICIENCY
// Predecir la Carga de Calefacción (Heating Load)
// =================================================================

clc;
clear;

// 1. CARGA DE DATOS (Ruta absoluta de tu archivo CSV)
path_csv = 'C:\Users\USUARIO\Downloads\datasets_redes_neuronales-main\datasets_redes_neuronales\energy_efficiency\energy_efficiency.csv'; 

// Leer el archivo CSV (delimitador coma)
datos = csvRead(path_csv, ',', '.', 'double');

// 2. SEPARAR CARACTERÍSTICAS (X) Y VARIABLE OBJETIVO (y)
X_raw = datos(:, 1:8); 
y = datos(:, 9);       // Carga de Calefacción (Heating Load)

// Limpieza de datos usando sintaxis compatible con Scilab
filas_con_nan = isnan(y) | or(isnan(X_raw), 'c');
X_raw = X_raw(~filas_con_nan, :);
y = y(~filas_con_nan);

[n, m] = size(X_raw);

// 3. NORMALIZACIÓN Z-SCORE (Media 0, Desviación Estándar 1)
mu = mean(X_raw, 'r');
sigma = stdev(X_raw, 'r');
sigma(sigma == 0) = 1; // Prevenir división por cero

X_norm = (X_raw - ones(n, 1) * mu) ./ (ones(n, 1) * sigma);

// Agregar término de sesgo (columna de 1s)
X = [ones(n, 1), X_norm];

// 4. DIVISIÓN DE DATOS (80% Entrenamiento, 20% Prueba)
n_train = round(0.8 * n);

X_train = X(1:n_train, :);
y_train = y(1:n_train);

X_test = X(n_train+1:$, :);
y_test = y(n_train+1:$);

// 5. CÁLCULO DE COEFICIENTES POR MÍNIMOS CUADRADOS
beta = X_train \ y_train;

disp("--- COEFICIENTES DEL MODELO (BETA) ---");
disp(beta);

// 6. EVALUACIÓN Y PREDICCIÓN EN CONJUNTO DE PRUEBA
y_pred = X_test * beta;
errores = y_test - y_pred;

MAE = mean(abs(errores));
MSE = mean(errores .^ 2);
RMSE = sqrt(MSE);

// Coeficiente de determinación R^2
SS_res = sum(errores .^ 2);
SS_tot = sum((y_test - mean(y_test)) .^ 2);
R2 = 1 - (SS_res / SS_tot);

printf("\n=========================================\n");
printf("  MÉTRICAS EN CONJUNTO DE PRUEBA\n");
printf("=========================================\n");
printf("Error Absoluto Medio (MAE):   %.4f kW\n", MAE);
printf("Error Cuadrático Medio (MSE):  %.4f\n", MSE);
printf("Raíz del MSE (RMSE):           %.4f kW\n", RMSE);
printf("Coeficiente R^2:               %.4f\n", R2);
printf("=========================================\n");

// 7. GRAFICAR RESULTADOS REALES VS PREDICHOS
scf(0);
clf();
plot(y_test, 'r-o');
plot(y_pred, 'b-x');
xtitle("Energy Efficiency: Carga de Calefaccion Reales vs Predichos", "Muestras de Prueba", "Carga (kW)");
legend(["Valor Real (y_test)", "Prediccion (y_pred)"]);
grid();