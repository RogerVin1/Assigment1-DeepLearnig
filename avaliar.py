"""Avalia um checkpoint no split de teste do BBBC038 como segmentador de instâncias.

    python avaliar.py
    python avaliar.py --checkpoint checkpoints/outro.pt

Reporta mAP (IoU 0,50 a 0,95, matching guloso), erro absoluto de contagem, AP por limiar e a quebra por modalidade."""

import argparse

import numpy as np

import nucleos as nu

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=str(nu.CHECKPOINT))
args = parser.parse_args()

itens = nu.carregar_bbbc038(nu.baixar_bbbc038())
_, _, idx_teste = nu.split_estratificado(itens)
teste = nu.DadosReais(itens, idx_teste, recortes_por_imagem=2, seed=3)

modelo = nu.carregar_modelo(args.checkpoint)
res = nu.avaliar_instancias(lambda imagem: nu.decodificar_watershed(nu.prever_classes(modelo, imagem)), teste)

print(f"=== {args.checkpoint} | teste: {len(teste)} recortes de {len(idx_teste)} imagens ===")
print(f"mAP (IoU 0.50 a 0.95): {res['mAP']:.3f}")
print(f"erro absoluto de contagem por recorte: {res['erro_contagem_medio']:.2f}\n")

print("AP por limiar de IoU:")
for limiar, ap in zip(nu.LIMIARES_IOU, res["aps"].mean(axis=0)):
    print(f"  IoU >= {limiar:.2f}: {ap:.3f}")

print(f"\n{'modalidade':15s} | n   | mAP   | erro de contagem")
for mod in np.unique(res["modalidades"]):
    sel = res["modalidades"] == mod
    print(f"{mod:15s} | {sel.sum():3d} | {res['maps'][sel].mean():.3f} | {res['erros'][sel].mean():.2f}")
