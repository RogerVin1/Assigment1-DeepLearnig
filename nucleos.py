"""Pipeline de segmentação de instâncias de núcleos (BBBC038), empacotado a partir do assigment1DL.ipynb.

As funções são as mesmas do notebook (mesmos nomes e mesma implementação) para que `treinar.py`,
`avaliar.py` e `inferencia.ipynb` rodem sem o notebook. Trilha A da Parte 2: a rede prevê
fundo / interior / fronteira e o watershed a partir dos interiores devolve as instâncias."""

import copy, pickle, zipfile, urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import ndimage
from skimage.segmentation import watershed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAIZ_DADOS = Path("data/BBBC038")
URL_TREINO = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
CHECKPOINT = Path("checkpoints/unet_3classes_bbbc038.pt")
ESPESSURA = 2
TILE, PASSO = 128, 96
LIMIARES_IOU = np.arange(0.5, 1.0, 0.05)


# ---------------------------------------------------------------- dados

def baixar_bbbc038(raiz=RAIZ_DADOS, url=URL_TREINO):
    pasta = raiz / "stage1_train"
    if pasta.exists() and any(pasta.iterdir()):
        return pasta

    raiz.mkdir(parents=True, exist_ok=True)
    arquivo_zip = raiz / "stage1_train.zip"
    if not arquivo_zip.exists():
        print("baixando", url)
        urllib.request.urlretrieve(url, arquivo_zip)

    with zipfile.ZipFile(arquivo_zip) as z:
        z.extractall(pasta)

    return pasta


def carregar_imagem_bbbc(pasta_imagem):
    """Devolve (rgb em [0,1], mapa de instâncias) de uma pasta <ImageId>/ do BBBC038."""
    arquivo = next((pasta_imagem / "images").glob("*.png"))
    rgb = np.asarray(Image.open(arquivo).convert("RGB")).astype(np.float32) / 255

    inst_mask = np.zeros(rgb.shape[:2], dtype=np.int32)
    for k, arquivo_mascara in enumerate(sorted((pasta_imagem / "masks").glob("*.png")), start=1):
        mascara = np.asarray(Image.open(arquivo_mascara).convert("L")) > 0
        inst_mask[mascara & (inst_mask == 0)] = k

    return rgb, inst_mask


def modalidade(rgb):
    """A mediana da imagem é o fundo (que domina a área); a saturação separa histologia das demais."""
    saturacao = (rgb.max(axis=2) - rgb.min(axis=2)).mean()
    fundo = np.median(rgb.mean(axis=2))

    if saturacao > 0.05:
        return "histologia"
    if fundo > 0.5:
        return "campo_claro"
    return "fluorescencia"


def para_cinza(rgb):
    """Cinza em [0,1] com núcleos claros sobre fundo escuro: inverte se o fundo (mediana) é claro
       e estica o contraste entre os percentis 1 e 99."""
    cinza = rgb.mean(axis=2)
    if np.median(cinza) > 0.5:
        cinza = 1 - cinza

    p1, p99 = np.percentile(cinza, [1, 99])
    if p99 - p1 > 1e-3:
        cinza = (cinza - p1) / (p99 - p1)

    return np.clip(cinza, 0, 1).astype(np.float32)


def carregar_bbbc038(pasta, cache=RAIZ_DADOS / "stage1_train_cache.pkl"):
    """Lista de dicts {id, imagem uint8 (já em cinza normalizado), inst int16, modalidade}.
       uint8/int16 mantêm o cache em ~300 MB; a leitura dos ~29 mil PNGs só acontece uma vez."""
    if cache.exists():
        return pickle.load(open(cache, "rb"))

    pastas = sorted(p for p in pasta.iterdir() if (p / "images").is_dir())
    itens = []

    for i, pasta_imagem in enumerate(pastas, start=1):
        rgb, inst_mask = carregar_imagem_bbbc(pasta_imagem)
        itens.append({"id": pasta_imagem.name,
                      "imagem": np.round(para_cinza(rgb) * 255).astype(np.uint8),
                      "inst": inst_mask.astype(np.int16),
                      "modalidade": modalidade(rgb)})
        if i % 100 == 0:
            print(f"  {i}/{len(pastas)} imagens lidas")

    pickle.dump(itens, open(cache, "wb"))
    return itens


def split_estratificado(itens, fracoes=(0.70, 0.15, 0.15), seed=0):
    """Embaralha e divide cada modalidade separadamente, na mesma proporção."""
    rng = np.random.default_rng(seed)
    por_modalidade = {}
    for i, item in enumerate(itens):
        por_modalidade.setdefault(item["modalidade"], []).append(i)

    treino, val, teste = [], [], []
    for indices in por_modalidade.values():
        indices = rng.permutation(indices)
        n_treino = round(fracoes[0] * len(indices))
        n_val = round(fracoes[1] * len(indices))
        treino += indices[:n_treino].tolist()
        val += indices[n_treino:n_treino + n_val].tolist()
        teste += indices[n_treino + n_val:].tolist()

    return treino, val, teste


def recorte_aleatorio(imagem, inst_mask, size, rng, tentativas=5):
    """Recorte size×size em posição aleatória; tenta algumas vezes até conter pelo menos um núcleo."""
    imagem = imagem.astype(np.float32) / 255
    inst_mask = inst_mask.astype(np.int32)
    altura, largura = imagem.shape

    for _ in range(tentativas):
        y0 = rng.integers(0, altura - size + 1)
        x0 = rng.integers(0, largura - size + 1)
        recorte_inst = inst_mask[y0:y0 + size, x0:x0 + size]
        if recorte_inst.any():
            break

    return imagem[y0:y0 + size, x0:x0 + size].copy(), recorte_inst.copy()


class DadosReais(Dataset):
    """Recortes 128×128 do BBBC038. `.instancias[i]` e `.modalidades[i]` acompanham cada recorte;
       `.amostra(i)` devolve (imagem, inst_mask) em numpy."""
    def __init__(self, itens, indices, recortes_por_imagem, size=128, seed=0):
        rng = np.random.default_rng(seed)
        self.imagens, self.instancias, self.modalidades = [], [], []

        for i in indices:
            for _ in range(recortes_por_imagem):
                imagem, inst_mask = recorte_aleatorio(itens[i]["imagem"], itens[i]["inst"], size, rng)
                self.imagens.append(imagem)
                self.instancias.append(inst_mask)
                self.modalidades.append(itens[i]["modalidade"])

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.imagens[idx]).unsqueeze(0)
        y = torch.from_numpy((self.instancias[idx] > 0).astype(np.float32)).unsqueeze(0)
        return x, y

    def amostra(self, idx):
        return self.imagens[idx], self.instancias[idx]


def ids_objetos(inst_mask):
    return np.unique(inst_mask[inst_mask != 0])


def n_objetos(inst_mask):
    return len(ids_objetos(inst_mask))


def colorir_instancias(inst_mask, rng=None):
    """Imagem RGB com uma cor aleatória por instância e fundo preto, só para visualização."""
    rng = rng or np.random.default_rng(0)
    rgb = np.zeros((*inst_mask.shape, 3), dtype=np.float32)

    for k in range(1, inst_mask.max() + 1):
        rgb[inst_mask == k] = rng.uniform(0.3, 1.0, size=3)

    return rgb


# ---------------------------------------------------------------- representação de 3 classes

def rotulo_3classes(inst_mask, espessura=ESPESSURA):
    classes = np.zeros(inst_mask.shape, dtype=np.int64)
    for k in ids_objetos(inst_mask):
        nucleo = inst_mask == k
        classes[nucleo] = 2
        classes[ndimage.binary_erosion(nucleo, iterations=espessura)] = 1
    return classes


class DadosTresClasses(Dataset):
    """Os mesmos recortes de um DadosReais, com o rótulo de 3 classes como alvo."""
    def __init__(self, dados, espessura=ESPESSURA):
        self.imagens = dados.imagens
        self.classes = [rotulo_3classes(inst_mask, espessura) for inst_mask in dados.instancias]

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):
        return torch.from_numpy(self.imagens[idx]).unsqueeze(0), torch.from_numpy(self.classes[idx])


def pesos_frequencia_mediana(dados_3c):
    """w_c = mediana(f) / f_c, com f_c a fração de pixels da classe c (Eigen & Fergus, 2015)."""
    frequencia = np.bincount(np.concatenate([classes.ravel() for classes in dados_3c.classes]), minlength=3)
    frequencia = frequencia / frequencia.sum()
    return torch.tensor(np.median(frequencia) / frequencia, dtype=torch.float32, device=device)


def decodificar_watershed(classes):
    interior = classes == 1
    foreground = classes >= 1
    marcadores, _ = ndimage.label(interior)
    relevo = -ndimage.distance_transform_edt(foreground)
    return watershed(relevo, markers=marcadores, mask=foreground).astype(np.int32)


# ---------------------------------------------------------------- rede e treino

def bloco_conv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNetPequena(nn.Module):
    """U-Net com dois níveis de pooling e skip connections por concatenação."""
    def __init__(self, n_saidas=1):
        super().__init__()
        self.enc1 = bloco_conv(1, 16)
        self.enc2 = bloco_conv(16, 32)
        self.meio = bloco_conv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = bloco_conv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = bloco_conv(32, 16)
        self.saida = nn.Conv2d(16, n_saidas, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.meio(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.saida(d1)


def carregar_modelo(caminho=CHECKPOINT):
    modelo = UNetPequena(n_saidas=3).to(device)
    modelo.load_state_dict(torch.load(caminho, map_location=device))
    return modelo.eval()


def flip_aleatorio(x, y, rng):
    """O mesmo flip (horizontal e/ou vertical) na imagem e na máscara."""
    if rng.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if rng.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    return x, y


def treinar(modelo, treino, val, epocas, criterio, metricas, lr=1e-3, batch_size=8, augmentar=None, seed=0, verboso=True):
    """Adam + `criterio`; a cada época avalia `metricas(modelo, val)` (dict) e guarda tudo no histórico.
       O modelo fica com os pesos da época em que a primeira métrica foi máxima."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    carregador = DataLoader(treino, batch_size=batch_size, shuffle=True)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    historico = {"perda": []}
    melhor = (-np.inf, None)

    for epoca in range(epocas):
        modelo.train()
        perdas = []

        for x, y in carregador:
            if augmentar is not None:
                x, y = augmentar(x, y, rng)
            x, y = x.to(device), y.to(device)
            otimizador.zero_grad()
            perda = criterio(modelo(x), y)
            perda.backward()
            otimizador.step()
            perdas.append(perda.item())

        valores = metricas(modelo, val)
        historico["perda"].append(np.mean(perdas))
        for nome, valor in valores.items():
            historico.setdefault(nome, []).append(valor)

        principal = next(iter(valores.values()))
        if principal > melhor[0]:
            melhor = (principal, copy.deepcopy(modelo.state_dict()))

        if verboso:
            print(f"época {epoca + 1:2d} | perda {np.mean(perdas):.3f} | " + " | ".join(f"{nome} {valor:.3f}" for nome, valor in valores.items()))

    modelo.load_state_dict(melhor[1])
    return historico


# ---------------------------------------------------------------- inferência

def prever_classes(modelo, imagem):
    modelo.eval()
    x = torch.from_numpy(imagem)[None, None].to(device)
    with torch.no_grad():
        return modelo(x).argmax(dim=1)[0].cpu().numpy()


def prob_classes(modelo, imagem):
    modelo.eval()
    x = torch.from_numpy(imagem)[None, None].to(device)
    with torch.no_grad():
        return torch.softmax(modelo(x), dim=1)[0].cpu().numpy()


def inicios(tamanho, tile=TILE, passo=PASSO):
    posicoes = list(range(0, tamanho - tile + 1, passo))
    if posicoes[-1] != tamanho - tile:
        posicoes.append(tamanho - tile)
    return posicoes


def tiles(imagem):
    for y in inicios(imagem.shape[0]):
        for x in inicios(imagem.shape[1]):
            yield slice(y, y + TILE), slice(x, x + TILE)


def inferir_fundido(modelo, imagem):
    """Média das probabilidades das 3 classes na sobreposição dos tiles; um único watershed no fim."""
    soma = np.zeros((3, *imagem.shape), dtype=np.float32)
    cobertura = np.zeros(imagem.shape, dtype=np.float32)
    for janela in tiles(imagem):
        soma[(slice(None), *janela)] += prob_classes(modelo, imagem[janela])
        cobertura[janela] += 1
    return decodificar_watershed((soma / cobertura).argmax(0))


def segmentar_arquivo(modelo, caminho):
    """Qualquer imagem (PNG/JPG/TIF, cinza ou colorida, qualquer tamanho) -> (cinza normalizado, mapa de instâncias)."""
    rgb = np.asarray(Image.open(caminho).convert("RGB")).astype(np.float32) / 255
    imagem = para_cinza(rgb)

    altura, largura = imagem.shape
    faltam = max(0, TILE - altura), max(0, TILE - largura)
    if any(faltam):
        imagem = np.pad(imagem, ((0, faltam[0]), (0, faltam[1])), mode="reflect")

    inst_mask = inferir_fundido(modelo, imagem)[:altura, :largura]
    return imagem[:altura, :largura], inst_mask


# ---------------------------------------------------------------- métricas de instância

def matriz_iou(pred_labels, gt_labels):
    """IoU[i, j] entre o i-ésimo objeto previsto e o j-ésimo verdadeiro."""
    ids_pred = np.unique(pred_labels[pred_labels != 0])
    ids_gt = np.unique(gt_labels[gt_labels != 0])
    iou = np.zeros((len(ids_pred), len(ids_gt)), dtype=np.float32)
    coluna = {j: c for c, j in enumerate(ids_gt)}

    for linha, i in enumerate(ids_pred):
        objeto = pred_labels == i
        vizinhos = np.unique(gt_labels[objeto])
        for j in vizinhos[vizinhos != 0]:
            intersecao = np.logical_and(objeto, gt_labels == j).sum()
            uniao = objeto.sum() + (gt_labels == j).sum() - intersecao
            iou[linha, coluna[j]] = intersecao / uniao

    return iou


def matching_guloso(iou, limiar):
    """Casa os pares de maior IoU primeiro, cada objeto no máximo uma vez. Devolve (TP, FP, FN)."""
    n_pred, n_gt = iou.shape
    pares = sorted(((iou[i, j], i, j) for i in range(n_pred) for j in range(n_gt) if iou[i, j] >= limiar), reverse=True)
    usados_pred, usados_gt = set(), set()

    for _, i, j in pares:
        if i not in usados_pred and j not in usados_gt:
            usados_pred.add(i)
            usados_gt.add(j)

    tp = len(usados_pred)
    return tp, n_pred - tp, n_gt - tp


def aps_de_uma_imagem(pred_labels, gt_labels):
    """AP em cada limiar de IoU para um par (previsão, gabarito)."""
    iou = matriz_iou(pred_labels, gt_labels)
    aps = []
    for limiar in LIMIARES_IOU:
        tp, fp, fn = matching_guloso(iou, limiar)
        aps.append(tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0)
    return np.array(aps)


def avaliar_instancias(segmentar, dados):
    """mAP, erro de contagem e densidade por recorte. `segmentar(imagem) -> labels` é rede + pós-processamento."""
    resultado = {"aps": [], "densidades": [], "erros": []}

    for k in range(len(dados)):
        imagem, gt_labels = dados.amostra(k)
        pred_labels = segmentar(imagem)
        resultado["aps"].append(aps_de_uma_imagem(pred_labels, gt_labels))
        resultado["densidades"].append(n_objetos(gt_labels))
        resultado["erros"].append(abs(n_objetos(pred_labels) - n_objetos(gt_labels)))

    resultado = {chave: np.array(valor) for chave, valor in resultado.items()}
    resultado["maps"] = resultado["aps"].mean(axis=1)
    resultado["mAP"] = resultado["maps"].mean()
    resultado["erro_contagem_medio"] = resultado["erros"].mean()
    resultado["modalidades"] = np.array(dados.modalidades)
    return resultado


def metrica_watershed(modelo, val):
    segmentar = lambda imagem: decodificar_watershed(prever_classes(modelo, imagem))
    return {"val_mAP": avaliar_instancias(segmentar, val)["mAP"]}
