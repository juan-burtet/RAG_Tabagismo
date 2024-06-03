# Teste Prático para Cientista de Dados - A3Data (LLM)

Este repositório foi utilizado para desenvolvimento do teste prático para a vaga de Cientista de Dados na A3Data. Para entender o objetivo do teste, acesse o documento na pasta references/

O desenvolvimento do projeto pode ser verificado no notebook encontrado na pasta notebooks. Para rodar o projeto em forma de API, é necessário fazer instalação dos pacotes do requirements, e utilizar os seguintes comandos:

```
ollama serve & ollama pull llama3
```

depois disso, só precisar iniciar a API utilizando o uvicorn:

```
uvicorn app.main:app
```
