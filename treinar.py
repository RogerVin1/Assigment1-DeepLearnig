"""Treina o modelo final (U-Net de 3 classes + watershed) no BBBC038 e salva o checkpoint.

    python treinar.py                # 20 épocas -> checkpoints/unet_3classes_bbbc038.pt
    python treinar.py --epocas 5     # teste rápido

Baixa o dataset na primeira execução. O split treino/val/teste é o mesmo do notebook (seed 0)."""

import argparse, time
from pathlib import Path

import torch
import torch.nn as nn

import nucleos as nu

parser = argparse.ArgumentParser()
parser.add_argument("--epocas", type=int, default=20)
parser.add_argument("--saida", default=str(nu.CHECKPOINT))
args = parser.parse_args()

itens = nu.carregar_bbbc038(nu.baixar_bbbc038())
idx_treino, idx_val, _ = nu.split_estratificado(itens)

treino = nu.DadosTresClasses(nu.DadosReais(itens, idx_treino, recortes_por_imagem=4, seed=1))
val = nu.DadosReais(itens, idx_val, recortes_por_imagem=2, seed=2)
print(f"recortes 128×128 -> treino {len(treino)} | val {len(val)} | dispositivo {nu.device}")

modelo = nu.UNetPequena(n_saidas=3).to(nu.device)
criterio = nn.CrossEntropyLoss(weight=nu.pesos_frequencia_mediana(treino))

inicio = time.time()
nu.treinar(modelo, treino, val, args.epocas, criterio, nu.metrica_watershed, augmentar=nu.flip_aleatorio)
print(f"treinou em {(time.time() - inicio) / 60:.1f} min")

Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
torch.save(modelo.state_dict(), args.saida)
print("checkpoint salvo em", args.saida)
