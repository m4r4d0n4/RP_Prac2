# RP_Prac2

## Entrenar modelo
El modelo entrenado se guardará en `./train/model.keras`
```
docker build -t tf_rp2 .
docker run --gpus all -v $(pwd)/train:/usr/src/app/train -it --rm tf_rp2
```
En caso de que no tengamos gpu es necesario quitar `--gpus all` tanto en el entrenamiento como en inferencia

## Inferir con el modelo

```
 docker build -t tf_load_rp2 -f Dockerfile_inferir .
 docker run --gpus all -v $(pwd)/test:/usr/src/app/test -it --rm tf_load_rp2
```
 El archivo competicion se genera en `./test/Competicion2.txt`, por defecto se infiere con inception.keras, para cualquier otro modelo editar el Dockerfile_inferir