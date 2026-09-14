# AI_LOG — como usamos IA neste assignment

Usamos o Claude (Claude Code, no terminal) como par de programação ao longo do trabalho. O que segue é um resumo honesto de onde a IA entrou e de como o trabalho foi conduzido.

## O caminho: sintético primeiro, real depois

No começo tivemos dificuldade em lidar com os dados reais — o BBBC038 tem um PNG por núcleo, imagens de tamanhos e modalidades diferentes (fluorescência com fundo escuro, histologia colorida com fundo claro, campo claro), e não estava claro como transformar isso em algo que a rede da aula pudesse consumir. Em vez de travar aí, seguimos a ordem do enunciado ao pé da letra: fizemos **todas as partes primeiro com o dataset sintético de elipses** da Parte 0. Isso deixou o pipeline inteiro pronto (U-Net, matching guloso, mAP, rótulo de 3 classes, watershed, ablações, inferência em mosaico, campo receptivo) sem depender dos dados reais.

Depois, com o pipeline funcionando, **adaptamos parte por parte para o BBBC038**. Os principais episódios em que recorremos à IA:

- **Carregar os dados reais.** Pedimos ajuda para montar o `DadosReais` com a mesma interface do `DadosSinteticos`: juntar os PNGs de máscara num mapa de instâncias, normalizar a polaridade (inverter histologia e campo claro para que o núcleo fique sempre claro sobre fundo escuro), classificar a modalidade por heurística de cor/brilho e fazer o split estratificado por modalidade que o enunciado pede. A IA baixou o dataset e validou a heurística (100% das imagens ficaram com núcleo mais claro que o fundo depois da normalização) antes de propor o código.
- **Adaptar as Partes 1 a 6.** Cada parte foi reescrita em cima dos splits reais, mantendo as funções já testadas no sintético. O ganho de ter feito o sintético antes foi grande: sabíamos que o matching e o watershed estavam certos, então as diferenças de resultado eram atribuíveis aos dados.
- **Limpeza e organização.** Pedimos uma passada de simplificação: um único `treinar` para todos os modelos (binário, 3 classes, ablações), `avaliar_instancias` recebendo qualquer `segmentar(imagem) -> labels`, nomes de variáveis claros, sem comentários óbvios, uma célula de markdown por item do enunciado. Uma dessas alterações mexeu em células anteriores e nos obrigou a reexecutar o notebook; a partir daí a regra passou a ser não tocar em células já executadas.
- **Decisões de projeto discutidas com a IA** (e decididas por nós): anel de fronteira inteiro em vez de só no contato, espessura escolhida pelo "teto" do watershed aplicado ao gabarito, pesos por frequência mediana para a CE balanceada, seleção da época pelo mAP de validação depois do watershed (e não pela perda), fusão das probabilidades antes do watershed como correção da inferência em mosaico, encoder mais profundo como correção da Parte 5, histologia como modalidade retirada na Parte 6.
- **Entregáveis.** `nucleos.py`, `treinar.py`, `avaliar.py`, `inferencia.ipynb` e este README foram gerados a partir do notebook e testados nos dados reais.

## O que a IA não fez

As escolhas de trilha, de eixos de ablação, de modalidade retirada e de correção foram nossas; os resultados foram rodados e lidos por nós; e revisamos cada célula gerada antes de manter. Todo código do repositório é reproduzível a partir dele e entendemos o que cada função faz — o notebook está organizado justamente para que cada decisão tenha a justificativa ao lado do código.
