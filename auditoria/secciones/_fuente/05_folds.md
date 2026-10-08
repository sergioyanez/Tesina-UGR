# 5. Folds, entrenamiento interno, validación interna y test

## 5.1 Cómo se construyen

1. **Folds externos (test).** `particion.construir`: los 133 caballos se ordenan por cantidad de clips (descendente, desempate por id) y se asignan, uno a uno, al fold con menos clips acumulados. Es una *réplica del reparto voraz de `GroupKFold`*, **sin semilla y sin estratificar por clase**. **No coincide con `sklearn.model_selection.GroupKFold`**: comparadas, las dos particiones asignan el mismo fold al {{VAL:balance_folds_fuga.json:reproduccion_particion.fraccion_de_clips_con_el_mismo_fold_en_sklearn}} de los clips (los tamaños de fold sí son los mismos: 90/90/90/89). El docstring del módulo habla de «réplica determinista de `GroupKFold`»: es cierto en el algoritmo general, pero no en el reparto exacto. No afecta la validez, sí la afirmación de equivalencia.
2. **Corte interno.** Dentro de cada fold, `dividir_por_ejemplar(train_val, 0,25)` separa **train interno** y **validación interna** por caballo, también de forma determinista.
3. El test se usa **una sola vez**, tras cargar el mejor checkpoint (§7).

## 5.2 Composición de cada subconjunto

{{TABLA:folds_conjuntos}}

Proporción de cada subconjunto sobre los 359 clips: test ≈ 25 % (89–90 clips), validación interna ≈ 19–20 % (70–73 clips), train interno ≈ 55 % (197–199 clips); en caballos, ≈ 33 de test, 7 de validación y ≈ 93 de train interno por fold.

## 5.3 Hallazgo: la validación interna son los caballos con más clips

`dividir_por_ejemplar` ordena los caballos del `train_val` por cantidad de clips y **asigna a validación los primeros hasta cubrir el 25 % de los clips**. El resultado (`datos/validacion_interna_composicion.json`):

- La validación interna son **los 7 caballos con más clips** de cada fold, con **10 clips por caballo en promedio** (7 a 14), frente a 2,1 clips por caballo en el train interno y 2,7 en el conjunto completo.
- Por eso el train interno **pierde justamente a los caballos más representados** y la validación pesa por igual a esos 7 individuos, mientras que el test sí es una muestra de todos los caballos del fold (≈ 33).
- Cinco o seis de los 7 caballos de validación de cada fold tienen clips en más de una clase.
- La validación interna de cada fold queda **desbalanceada** (IR de 1,47 a 4,18; 65,7 % `Sereno` en el fold 0).

Es la base sobre la que se elige la época y los hiperparámetros (dropout 0,3, pesos `raiz`). **Es una debilidad real del diseño**, no una fuga: el test no interviene (§7). Sus efectos medibles son la elección de época ruidosa (20 a 40 entre folds con la misma semilla; 13 a 50 entre semillas) y una métrica de validación que no es un buen proxy: el test fue **igual o mayor** que la validación en los 4 folds (diferencia media +0,027).

{{FIG:fig_gaps.png|Figura 5.1. F1 macro de train interno, validación interna y test en la época elegida (izquierda) y brechas (derecha).}}
