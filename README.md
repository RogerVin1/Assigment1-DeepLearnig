# PA1 — Segmentação de instâncias de núcleos (BBBC038 / DSB2018)

## **Disciplina:** Aprendizado Profundo

## **Dupla:** Nicholas Costa e Roger Augusto

Segmentação de instâncias com as arquiteturas da aula, sem detectores por proposta de região.

Trilha escolhida (Parte 2): **fronteiras + watershed**. A U-Net prevê três classes por pixel (fundo / interior / fronteira) e as instâncias saem de um watershed com os interiores como marcadores.

## Estrutura

| Arquivo | O que é |
|---|---|
| `assigment1DL.ipynb` | Notebook principal, Partes 0 a 6. Toda tabela, curva e figura da apresentação sai daqui. |
| `nucleos.py` | O pipeline do notebook empacotado (mesmas funções, mesmos nomes) para os comandos abaixo. |
| `treinar.py` | Um comando que treina o modelo final e salva o checkpoint. |
| `avaliar.py` | Um comando que avalia um checkpoint no split de teste (mAP, erro de contagem, por modalidade). |
| `inferencia.ipynb` | Recebe o caminho de uma imagem qualquer, devolve a máscara de instâncias colorida e a contagem. Não retreina. |
| `checkpoints/unet_3classes_bbbc038.pt` | Pesos do modelo final (~0,5 MB; versionado direto no repositório, ver abaixo). |
| `AI_LOG.md` | Como a IA foi usada. |

## Ambiente

Python ≥ 3.10. Dependências: PyTorch, NumPy, SciPy, scikit-image, Pillow, Matplotlib.

```bash
pip install -r requirements.txt
```

No Google Colab tudo já está instalado; basta abrir o notebook. GPU é recomendada (o treino do modelo final leva ~5 min numa T4; na CPU, dezenas de minutos).

## Dados

BBBC038v1 (`stage1_train`, ~670 imagens, um PNG por núcleo). O download é automático na primeira execução de qualquer comando (83 MB, sem conta), a partir de https://bbbc.broadinstitute.org/BBBC038, para `data/BBBC038/`. A leitura dos ~29 mil PNGs acontece uma vez e fica em cache (`data/BBBC038/stage1_train_cache.pkl`, ~300 MB).

Alternativa: `kaggle competitions download -c data-science-bowl-2018` e descompactar `stage1_train.zip` em `data/BBBC038/stage1_train/`.

**Split.** Treino/validação/teste 70/15/15, estratificado por modalidade (fluorescência, histologia, campo claro — atribuída por heurística de cor e brilho do fundo), por imagem, com seed fixa. Recortes 128×128 (4 por imagem no treino, 2 na validação e no teste). O split é recomputado de forma determinística por `nucleos.split_estratificado`, então notebook e scripts usam exatamente as mesmas imagens.

## Comandos

```bash
python treinar.py            # treina 20 épocas -> checkpoints/unet_3classes_bbbc038.pt
python avaliar.py            # mAP (IoU 0,50–0,95), erro de contagem, AP por limiar, por modalidade
```

`python treinar.py --epocas 5` para um teste rápido; `python avaliar.py --checkpoint outro.pt` para avaliar outro arquivo.

Inferência numa imagem qualquer: abrir `inferencia.ipynb`, ajustar `CAMINHO_IMAGEM` e rodar. Imagens maiores que 128×128 são processadas em tiles sobrepostos com fusão das probabilidades antes do watershed (Parte 4).

## Métrica

mAP de instância implementado à mão (`nucleos.aps_de_uma_imagem`): matriz de IoU entre objetos previstos e verdadeiros, **matching guloso por IoU decrescente** (cada objeto entra em no máximo um par), AP<sub>t</sub> = TP / (TP + FP + FN) para t ∈ {0,50, 0,55, …, 0,95}, média por imagem e depois entre imagens. Erro absoluto de contagem por imagem reportado junto.

## Checkpoint

O enunciado pede os pesos do modelo final, "link se for grande". Os nossos são pequenos (a U-Net tem ~120 mil parâmetros, o arquivo tem ~0,5 MB), então o `.pt` está versionado diretamente em `checkpoints/`, sem link externo. O `.gitignore` exclui apenas `data/`.

## O notebook

| Parte | Conteúdo |
|---|---|
| 0 | Teste unitário sintético (elipses 128×128), treina em segundos, Dice/IoU. |
| Dados | Download, mapa de instâncias, normalização de polaridade, modalidade, split estratificado, `DadosReais`. |
| 1 | Baseline: U-Net binária (Dice/IoU), instâncias por limiar + componentes conexos, mAP e erro de contagem, regra de matching, métricas vs densidade. |
| 2 | Rótulo de 3 classes (espessura da fronteira justificada pelo teto do watershed), CE balanceada por frequência mediana, decodificação por watershed, comparação lado a lado com a baseline. |
| 3 | Ablações com 2 seeds: skip connections vs pool indices (SegNet); CE → CE balanceada → focal γ ∈ {1, 2, 5} → focal balanceada, com recall da classe fronteira. |
| 4 | Inferência em mosaico: tiles sobrepostos, objeto partido na emenda, correção por fusão das probabilidades antes do watershed, mAP antes/depois. |
| 5 | Galeria de falhas com diagnóstico por medidas, campo receptivo teórico (32 px encoder / 47 px total) vs tamanho dos núcleos, correção com encoder mais profundo (103 px). |
| 6 | Teste de estresse: treino sem histologia, avaliação nela e nas demais modalidades como controle. |
