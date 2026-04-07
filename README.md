# image-classification
Refactoring PyTorch image classification quick start example.

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/jgabriellacerda/image-classification.git
cd image-classification
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

Para treinar o modelo e executar uma predição, execute o seguinte comando:

```bash
python main.py
```

O script `main.py` irá:
1. Baixar o dataset FashionMNIST (se ainda não tiver sido baixado).
2. Treinar a rede neural por 10 épocas.
3. Salvar os pesos do modelo treinado em `model.pth`.
4. Executar uma predição com uma imagem do conjunto de teste e imprimir o resultado.

Para executar apenas a predição usando um modelo já treinado, você pode adaptar o script `predict.py` para carregar o modelo e uma imagem de sua escolha.

