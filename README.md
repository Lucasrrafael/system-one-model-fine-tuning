# laya-treino

Fine-tuning de modelos de decisão "System One" numa pergunta `score` (escala ordinal):
**"quão urgente é este ticket de suporte?"** (`low` → `medium` → `high`), com o dataset
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) em inglês.
Dois modelos suportados: o [Laya](https://huggingface.co/convaiinnovations/laya) (roda na RTX 3050) e o
[clef-flash](https://huggingface.co/Cloudflare/clef-flash) (9B, roda no Colab; ver "clef-flash no Colab").
Cada experimento vai para um MLflow local rodando em Docker.

```
.env.example            padrões de todos os parâmetros (commitado); .env sobrescreve no seu PC
configs/clef-colab.env  perfil do clef-flash no Colab (ENV_FILE=configs/clef-colab.env)
docker-compose.yml      servidor MLflow (dados em D:\laya-treino\mlflow)
prepare_data.py         dataset -> train/val/test.jsonl (+ hash de cada arquivo)
run.py                  baixa o modelo, mede o antes, treina, mede o depois, loga no MLflow, apaga o modelo
sync_mlflow.py          importa para o seu MLflow os runs feitos no Colab
colab/clef_flash.ipynb  notebook do Colab: clona este repo e roda o run.py
src/families/           o que muda por modelo: laya.py e clef.py (download, previsão, treino)
src/core/               comum: dados, alvo por option_id, loop de treino, calibração
src/                    config, download, métricas, gráficos
tests/                  python -m pytest (inclui o caminho do clef num Qwen3.5 minúsculo)
```

A configuração vem, em ordem de prioridade: variável do terminal → arquivo em `ENV_FILE` → `.env` → `.env.example`.

O que é pesado fica em `D:\laya-treino` (`WORK_DIR`), fora do OneDrive:

| Pasta | Conteúdo |
|---|---|
| `.venv` | ambiente Python (PyTorch CUDA ≈ 5 GB) |
| `data/en` | JSONL gerados e `data_stats.json` |
| `models` | modelo baixado do Hugging Face (apagado no fim do run) |
| `runs/<run>` | checkpoint ajustado, previsões em CSV, tabela de comparação |
| `mlflow` | banco e artefatos do MLflow |
| `hf-home`, `hf-cache` | cache do Hugging Face (dataset, tokenizer) |

## Como rodar

1. Abra o Docker Desktop e suba o MLflow (na pasta do projeto):
   ```powershell
   docker compose up -d
   ```
   Interface em http://localhost:5000.
2. Ative o ambiente:
   ```powershell
   D:\laya-treino\.venv\Scripts\Activate.ps1
   ```
3. Gere os dados (uma vez, ou quando mudar o dataset ou os splits no `.env`):
   ```powershell
   python prepare_data.py
   ```
4. Rode um experimento:
   ```powershell
   python run.py
   ```

Para testar outro modelo, troque `MODEL_FAMILY`, `HF_MODEL_ID` e `HF_MODEL_SUBFOLDER` no `.env` e rode `run.py` de novo.
O modelo é baixado para `D:\laya-treino\models`, usado e apagado no fim (`DELETE_MODEL_AFTER_RUN=true`).
- `MODEL_FAMILY=laya`: checkpoints no formato do Laya (`rl_agent_config.json`): `convaiinnovations/laya`
  (raiz e subpasta `multilingual`) e `convaiinnovations/laya-typed-decisions`.
- `MODEL_FAMILY=clef`: `Cloudflare/clef-flash` (precisa de GPU grande, ver abaixo).

Para rodar os testes: `python -m pytest`. Os do clef baixam só o código e o tokenizer do modelo (~20 MB).

Para um teste rápido ponta a ponta, use `MAX_TRAIN_SAMPLES=500`, `MAX_EVAL_SAMPLES=200` e `EPOCHS=1`,
rode `prepare_data.py` de novo e depois `run.py`.

## O dataset

- 28.261 tickets em inglês nos dois CSVs, mas eles se sobrepõem: `prepare_data.py` remove repetidos
  (e textos iguais com prioridades diferentes) antes de dividir, senão o mesmo ticket cairia no
  treino e no teste.
- Em inglês só existem 3 prioridades: `low`, `medium`, `high`. (`very_low` e `critical` só aparecem no CSV alemão.)
- Divisão 80/10/10 estratificada por prioridade (`SPLIT_SEED=42`), para os três conjuntos terem a mesma proporção de níveis.
- É quase certamente um dataset sintético (o autor vende um gerador de tickets) e a licença é CC BY-NC 4.0.
  Bom para estudar, mas os rótulos não foram validados por humanos, então o teto de acurácia é desconhecido.

## Resultados

### Laya, treino completo (2.375 tickets de teste)

Run `laya-lora-20261030-214114` no experimento `system-one-model-priority`: `convaiinnovations/laya`, `TRAIN_MODE=lora`,
4 épocas, configuração padrão do `.env.example`.

| Métrica | Antes | Depois |
|---|---|---|
| Acurácia | 0,447 | **0,533** |
| F1 macro | 0,296 | **0,495** |
| Recall de `high` | 0,146 | **0,566** |
| QWK | 0,128 | **0,385** |
| ECE (calibração) | 0,268 | **0,028** |
| Precision@10 nas caixas de 100 tickets | 0,688 | **0,815** |

Levou 3,3 h na RTX 3050 6 GB, com pico de 2,8 GB de VRAM. Para comparar: chutar sempre `medium` acerta 41% no
mesmo teste.

### clef-flash, treino completo (mesmos 2.375 tickets)

Run `clef-flash-lora-20261031-152945` no experimento `system-one-model-priority`: `Cloudflare/clef-flash` na revisão
fixada, LoRA r=32 + cabeça, 2 épocas, `configs/clef-colab.env`, na G4 do Colab.

| Métrica | Antes | Depois |
|---|---|---|
| Acurácia | 0,383 | **0,589** |
| F1 macro | 0,357 | **0,550** |
| Recall de `high` | 0,595 | **0,617** |
| QWK | 0,134 | **0,451** |
| ECE (calibração) | 0,377 | **0,026** |
| Precision@10 nas caixas de 100 tickets | 0,608 | **0,863** |

O run inteiro levou 59 min (56 min de treino, 1.164 passos de otimização), com pico de 23,2 GB de VRAM. Treinou
202M dos 9,6B parâmetros (2,1%), e a temperatura calibrada ficou em 1,26. Sem treino, o clef fica abaixo do Laya
(acurácia 0,383 contra 0,447) e é bem mais confiante do que deveria (ECE 0,377).

### clef 27B, treino completo (mesmos 2.375 tickets)

Run `clef-lora-20261031-170923` no experimento `system-one-model-priority`: `Cloudflare/clef` (Qwen3.8-27B por baixo),
com o mesmo perfil e os mesmos hiperparâmetros do flash, na G4 do Colab. O código é o mesmo da família clef; só mudam
duas variáveis, definidas numa célula do notebook antes da sondagem:

```python
import os
os.environ["HF_MODEL_ID"] = "Cloudflare/clef"
os.environ["HF_MODEL_REVISION"] = "cfddda043dd190ef3b01cff449d5ee8cff1e8ec6"
```

| Métrica | Antes | Depois |
|---|---|---|
| Acurácia | 0,396 | **0,585** |
| F1 macro | 0,386 | **0,542** |
| Recall de `high` | 0,472 | **0,614** |
| QWK | 0,157 | **0,459** |
| ECE (calibração) | 0,414 | **0,028** |
| Precision@10 nas caixas de 100 tickets | 0,551 | **0,862** |

O run inteiro levou 2 h 54 (2 h 48 de treino, os mesmos 1.164 passos de otimização), com pico de 60,9 GB de VRAM.
Treinou 346M dos 27,7B parâmetros (1,2%), e a temperatura calibrada ficou em 1,07. A config do modelo tem um campo
`output_gate_type: "swish"` que o transformers não usa; a versão com que a Cloudflare testou o modelo (5.10.2)
também não, então o cálculo é o mesmo.

### Comparação: Laya × clef-flash × clef 27B (os três ajustados, mesmo teste)

| Métrica | Laya | clef-flash (9B) | clef (27B) |
|---|---|---|---|
| Acurácia | 0,533 | **0,589** | 0,585 |
| Balanced accuracy | 0,489 | **0,540** | 0,534 |
| F1 macro | 0,495 | **0,550** | 0,542 |
| Recall de `high` | 0,566 | **0,617** | 0,614 |
| Recall de `low` | 0,264 | **0,291** | 0,276 |
| QWK | 0,385 | 0,451 | **0,459** |
| Macro-MAE ↓ | 0,570 | 0,513 | **0,509** |
| Spearman | 0,462 | 0,509 | **0,519** |
| ECE ↓ | 0,028 | **0,026** | 0,028 |
| NLL ↓ | 0,929 | **0,878** | 0,882 |
| AP (`high` × resto) | 0,685 | **0,729** | 0,727 |
| Precision@10 | 0,815 | **0,863** | 0,862 |
| NDCG@10 | 0,893 | 0,919 | **0,921** |
| Parâmetros do modelo | 0,42B | 9,6B | 27,7B |
| Tempo / GPU | 3,3 h na RTX 3050 | 59 min na G4 | 2 h 54 na G4 |
| Pico de VRAM | 2,8 GB | 23,2 GB | 60,9 GB |

↓ = quanto menor, melhor.

- **clef × Laya:** os dois clef ganham do Laya em quase tudo, mais nas métricas ordinais (QWK +0,066 e +0,074) e no
  ranking (Precision@10 +0,048 e +0,047). Nas caixas de 100 tickets, o 1º colocado é `high` em 98,2% das vezes no
  flash e em 97,8% no 27B, contra 96,8% no Laya. A calibração empata: os três terminam com ECE ~0,03.
- **clef-flash × clef 27B:** empate. As diferenças vão de 0,000 a 0,015, ora a favor de um, ora do outro. Sem treino
  o 27B começa um pouco melhor (F1 0,386 contra 0,357), mas o fine-tuning apaga a diferença.
- **Custo:** o 27B levou ~3× o tempo e ~2,6× a VRAM do flash para chegar no mesmo lugar, então nesta tarefa o flash é
  a escolha. Só o Laya cabe na 3050.
- **Quanto confiar:** com 2.375 tickets, o erro-padrão da acurácia é ~0,01; o do recall de `low` é maior, porque só
  conta os tickets `low`. A vantagem dos clef sobre o Laya (~0,05 de acurácia) não é ruído do teste; as diferenças
  entre os dois clef e as de ECE são. Cada modelo rodou uma vez, com uma semente só, então a variação entre sementes
  não foi medida.

## Parâmetros

### Modo de treino (`TRAIN_MODE`)

A RTX 3050 de notebook tem 6 GB. O fine-tuning completo do Laya inglês (421M parâmetros) com AdamW
precisa de ~16 bytes por parâmetro (peso fp32 + gradiente + 2 momentos do Adam) ≈ **6,7 GB antes das
ativações**, então não cabe. Os modos:

| Modo | O que treina | Memória | Quando usar |
|---|---|---|---|
| `lora` (padrão) | adaptadores LoRA em todas as camadas lineares do encoder + cabeça de decisão inteira | cabe em 6 GB | na 3050 |
| `head` | só a cabeça de decisão; encoder congelado | cabe em 6 GB | linha de base rápida |
| `full` | tudo | ~16 GB | Colab/Kaggle T4 |

- **LoRA** (Hu et al., 2022) congela o peso original e aprende uma correção de posto baixo; no paper ele
  empata com o fine-tuning completo no GLUE treinando ~0,1–1% dos parâmetros. O QLoRA (Dettmers et al., 2023)
  mostrou que aplicar LoRA em **todas** as camadas lineares, e não só na atenção, é o que faz o resultado
  igualar ao fine-tuning completo. Por isso `LORA_TARGET_MODULES=Wqkv,Wo,Wi`, que no ModernBERT cobre atenção e MLP.
- **Head** é o "linear probing" do Kumar et al. (2022): treinar só a cabeça preserva as features do
  pré-treino. Serve como piso de comparação para os outros modos.
- No fim do modo `lora`, os adaptadores são fundidos nos pesos (`merge_and_unload`). O checkpoint salvo é um Laya
  comum, que `laya.load` abre sem o `peft`.

### Medido nesta RTX 3050 6 GB (notebook na bateria, GPU limitada a 35 W)

| | Valor |
|---|---|
| Parâmetros treináveis no modo `lora` | 30,1 M de 424,9 M (7%: adaptadores + cabeça de decisão) |
| Pico de VRAM no treino `lora`, micro-batch 8 | **2,6 GB** |
| Mesmo treino com `GRADIENT_CHECKPOINTING=false` | 6,5 GB: passa dos 6 GB, o Windows joga o excesso para a RAM e tudo fica **mais lento** |
| Tempo por micro-batch de 8 tickets (~190 tokens) | ~2,5–2,8 s |
| Validação por época (fp32 + TF32) | ~94 ms por ticket, ~4 min para os 2.375 de validação |
| Inferência em lote do checkpoint ajustado | ~31 ms por ticket |
| Treino completo estimado (18,6 mil tickets × 4 épocas) | **~7 h na bateria**. Ligado na tomada e no modo "Melhor desempenho" do Windows deve ser mais rápido (não medido). |

Para um resultado mais rápido: `MAX_TRAIN_SAMPLES=6000` (~2 h) ou o modo `full` num T4 do Colab/Kaggle.

### Hiperparâmetros

| Parâmetro | Valor | Por quê |
|---|---|---|
| `EPOCHS` | 4 | O BERT recomenda 2–4 épocas (Devlin et al., 2019). O ModernBERT varreu 1–3 nos datasets grandes do GLUE e 2–10 nos pequenos (Warner et al., 2024). Mosbach et al. (2021) mostram que treinar mais (20 épocas) só compensa em datasets pequenos (~2,5 mil exemplos). Com ~19 mil exemplos, 4 fica no meio e é o padrão do notebook do Laya. A validação a cada época no MLflow mostra se vale subir ou descer. |
| `MICRO_BATCH` × `GRAD_ACCUM` | 8 × 8 = **64** | O que importa é o batch efetivo de 64 (padrão do Laya). O micro-batch de 8 é o que cabe na memória; com acumulação o gradiente é o mesmo de um batch de 64, porque o modelo não tem BatchNorm. O BERT usou 16/32. O termo de policy gradient do RLCD é ruidoso, e um batch maior reduz essa variância. Se der falta de memória (OOM), use 4 × 16. |
| `ENCODER_LR` | 2.5e-5 | Só no modo `full`. Fica dentro da faixa 2e-5–5e-5 do BERT e da varredura 1e-5–8e-5 do ModernBERT. Mosbach et al. usam 2e-5 como base estável. Sun et al. (2019) mostram que taxas altas (4e-4) fazem o BERT "esquecer" o pré-treino e não convergir. |
| `HEAD_LR` | 1e-4 | 4× a do encoder: camadas mais próximas da saída mudam mais. É o "discriminative fine-tuning" do ULMFiT (Howard & Ruder, 2018) e a taxa decrescente por camada de Sun et al. (2019). Padrão do Laya. |
| `LORA_LR` | 3e-4 | Adaptadores começam do zero e pedem taxa maior. O paper do LoRA usou 2e-4 a 4e-4 no RoBERTa-large; 3e-4 é o meio. |
| `LORA_R` / `LORA_ALPHA` | 8 / 16 | Valores do paper do LoRA para o RoBERTa-large (escala α/r = 2). Lá, postos entre 4 e 64 deram resultados parecidos. |
| `LORA_DROPOUT` | 0.1 | QLoRA usa 0,1 em modelos menores. |
| `WEIGHT_DECAY` | 0.01 | AdamW com decaimento separado (Loshchilov & Hutter, 2019); 0,01 é o do BERT. |
| `GRAD_CLIP` | 1.0 | Padrão do BERT; evita passos enormes quando um batch tem gradiente atípico. |
| `MIN_LR` | 1e-6 | Fim do cosseno: o Laya decai a taxa em cosseno (Loshchilov & Hutter, 2017) até esse valor. |
| `LOSS` | rlcd | Objetivo do Laya: soft cross-entropy + policy gradient com recompensa de regras de pontuação próprias. Inclui o RPS (Epstein, 1969), feito para escalas ordinais, que é exatamente o caso do `score`. A linha de base de grupo vem do GRPO (Shao et al., 2024). `soft-ce` fica disponível para comparação; o autor do Laya não mediu ganho do termo extra em outro conjunto. |
| `RL_SAMPLES`, `SIGMA_*`, `W_SPH`, `W_RPS` | 4, 0.4→0.1, 0.75, 1.0 | Padrões do notebook oficial do Laya; sem motivo para mudar antes de ter um resultado. |
| `LABEL_SMOOTHING` | 0.0 | Label smoothing mexe na calibração (Müller et al., 2019), e o Laya já ajusta temperatura depois. Deixar em 0 mantém uma fonte de calibração só. |
| `MAX_LEN` / `HEAD_MAX_LEN` | 512 / 192 | Orçamento do checkpoint inglês. As 3 opções + instrução usam bem menos que 192 tokens; sobram 320 para o ticket. O ticket p95 tem ~700 caracteres (~170 tokens), então quase nada é cortado. |
| `AMP` | true | Treino em fp16 com escala de loss (Micikevicius et al., 2018): ~metade da memória de ativações e usa os tensor cores da 3050. |
| `GRADIENT_CHECKPOINTING` | true | Recalcula ativações no backward em vez de guardar (Chen et al., 2016): troca ~30% de tempo por muita memória. |
| `CALIB_MAX` / `CALIB_FRAC` | 400 / 0.1 | Temperatura ajustada num pedaço do treino separado antes de treinar (temperature scaling, Guo et al., 2017). Ajustar em dados já vistos mediria o ajuste, não a calibração. |
| `SEED` | 0 | Só a semente muda bastante o resultado de fine-tuning (Dodge et al., 2020). Para comparar modos, rode 3 sementes (0, 1, 2) e compare as médias. |
| `SPLIT_SEED` | 42 | Fixa a divisão dos dados: todo run compara no mesmo teste. |

**Limitação conhecida:** o loop do Laya não tem warmup. BERT e Mosbach et al. usam 10% de warmup linear;
o cosseno do Laya começa direto na taxa cheia.

## clef-flash no Colab

O clef-flash é o Qwen3.5-9B (18,8 GB em bf16) com uma cabeça que pontua todas as opções de uma vez.
Não cabe na 3050, então roda no Colab (G4, 96 GB). O código continua aqui: o notebook clona este repositório.

### Passo a passo

1. **No PC:** dê push da branch para o GitHub (o notebook clona a branch `clef-flash`).
2. **No Drive:** crie `MyDrive/system-one/data/en/` e envie `train.jsonl`, `val.jsonl`, `test.jsonl` e `data_stats.json`
   de `D:\laya-treino\data\en`. São os mesmos arquivos do teste do Laya; o notebook confere o SHA-256 de cada um.
3. **No Colab:** abra
   `https://colab.research.google.com/github/Lucasrrafael/system-one-model-fine-tuning/blob/clef-flash/colab/clef_flash.ipynb`,
   escolha a GPU G4 e rode as células:
   - **Sondagem:** 200 tickets, 1 época. Mede segundos por passo e VRAM; a célula seguinte converte isso em horas e
     unidades de computação.
   - **Run completo:** avaliação base, treino de 2 épocas, calibração e avaliação final.
   - A cada época o run (checkpoint incluso) e o MLflow são copiados para `MyDrive/system-one/`. Se o Colab cair, a
     seção "Se o Colab desconectar" continua do `checkpoint_latest`.
4. **De volta no PC:** depois do `Pronto.` do run completo, baixe `MyDrive/system-one/mlflow` (o Drive gera um zip),
   extraia em `D:\laya-treino\colab-mlflow` e importe. O `--source` é a pasta que tem o `mlflow.db`:
   ```powershell
   D:\laya-treino\.venv\Scripts\python.exe sync_mlflow.py --source D:\laya-treino\colab-mlflow\mlflow --experiment system-one-model-priority
   ```
   Os runs completos de todos os modelos (Laya e clef) ficam juntos no experimento `system-one-model-priority`. Sem
   `--experiment`, as sondagens (`clef-sondagem`) vêm junto, na aba delas. Um banco antigo, de quando o perfil usava
   `clef-tickets-priority`, entra na mesma aba com `--experiment clef-tickets-priority --into system-one-model-priority`.
   A pasta `runs` do Drive não precisa ser baixada: ela só tem o checkpoint, que não roda na 3050.

### Como a família clef funciona (`src/families/clef.py`)

- **Código do modelo:** usa o `joint_schema_model.py` do próprio repositório, na revisão fixada em
  `HF_MODEL_REVISION` (o arquivo dessa revisão foi lido).
- **Ordem das opções:** o encoder do clef põe `choice` em ordem alfabética e `noul` como `true, false`. O alvo de treino
  é montado pelo `option_id` (`src/core/targets.py`), nunca pela posição.
- **O que treina:** o backbone fica congelado; LoRA em todas as camadas lineares de texto (atenção cheia
  `q/k/v/o_proj`, atenção linear `in_proj_qkv`, `in_proj_z`, `out_proj`, MLP) e a cabeça inteira (122M) em fp32.
  A parte de visão fica de fora.
- **O que salva:** só o adaptador LoRA e a cabeça (~0,5 GB), não os 19 GB do backbone. A avaliação final carrega o
  modelo base e aplica os dois.
- **Calibração:** uma temperatura por tipo de pergunta, com piso 1.

### Hiperparâmetros do clef (`configs/clef-colab.env`)

| Parâmetro | Valor | Por quê |
|---|---|---|
| LoRA | r=32, α=64, dropout 0 | Receita do Inner Jev (Geuter et al., 2026) para Gemma e Qwen. A Cloudflare usou r=256, mas num treino multitarefa com muito mais dados. |
| Camadas do LoRA | todas as lineares de texto | QLoRA (Dettmers et al., 2023) e Inner Jev: LoRA só na atenção fica abaixo do fine-tuning completo. |
| `LORA_LR` / `HEAD_LR` | 1e-4 / 1e-4 | Inner Jev usou 1e-4 no Qwen3.5-4B e 1e-4 na cabeça; o Our Own Jev (PayPal) também 1e-4. |
| `WEIGHT_DECAY` | 0 | Inner Jev. |
| Batch | 8 × 4 = 32 | Our Own Jev usou 32. Com ~18,6 mil itens dá ~580 passos por época. |
| `EPOCHS` | 2 | Our Own Jev treinou 1 época num corpus maior; com uma tarefa só, 2 épocas dão mais passos sem repetir muito. |
| `WARMUP_FRACTION` | 0.05 | O Inner Jev usa warmup (WSD) e o Mosbach et al. mostra que warmup estabiliza o fine-tuning. O loop do Laya não tem. |
| `LOSS` | rlcd | O mesmo do run do Laya, para comparar; a Cloudflare também usa RLCD depois do CE + Brier. `soft-ce` e `soft-ce+brier` ficam disponíveis. |
| `CALIB_MIN_TEMPERATURE` | 1 | Inner Jev restringe a temperatura do Qwen a ≥ 1. O Our Own Jev viu a temperatura < 1 piorar o ECE nos modelos grandes. |
| `CLEF_MAX_LENGTH` | 2048 | O prompt fixo do clef + o ticket (p95 ~170 tokens) ficam bem abaixo; não corta nada. |
| `DTYPE` | bf16 | A G4 tem bf16; o T4 não tem (lá use fp16, com risco de overflow). |

## Métricas

Tudo é calculado no **teste**, duas vezes: `baseline/` (modelo baixado, sem treino) e `finetuned/`.
`val/` é a validação a cada época e `train/loss` é a loss por época. No MLflow, os gráficos ficam em
`plots/`, as previsões por ticket em `predictions/` e o antes × depois em `comparison.csv`.

### Classificação (nível com maior probabilidade)

| Métrica | Por quê |
|---|---|
| `accuracy` | A mais intuitiva, mas engana com classes desbalanceadas: chutar sempre `medium` já acerta ~40%. Compare com essa linha de base. |
| `balanced_accuracy` | Média do recall de cada classe (Brodersen et al., 2010); não premia ignorar a classe rara. |
| `precision_*`, `recall_*`, `f1_*` (macro, weighted e por nível) | Macro dá o mesmo peso a cada nível; weighted pondera pela frequência (Sokolova & Lapalme, 2009). O que mais importa aqui é `recall_high`: deixar passar um ticket urgente custa caro. |
| `mcc` | Usa as quatro células da matriz de confusão e só fica alto se o modelo for bom em todas as classes (Chicco & Jurman, 2020). |
| `cohen_kappa` | Concordância descontando o acaso (Cohen, 1960). |

### Ordinais (a ordem low < medium < high importa)

| Métrica | Por quê |
|---|---|
| `qwk` | Kappa com peso quadrático (Cohen, 1968): errar `low`↔`high` pesa 4× mais que errar por um nível. É a métrica padrão de escalas ordinais (ex.: correção automática de redações). |
| `mae`, `macro_mae` | Erro médio em níveis. O macro (Baccianella et al., 2009) calcula por classe verdadeira e tira a média, para resistir ao desbalanceamento; foi adotado pelas tarefas ordinais do SemEval. |
| `mae_expected` | Mesmo erro, mas usando o `score` contínuo que o Laya devolve (nível esperado). |
| `within_1` | % de previsões a no máximo 1 nível da verdade. |
| `spearman`, `kendall_tau` | Correlação de postos entre o score esperado e o nível real (Spearman, 1904; Kendall, 1938): o modelo ordena bem? |

### Probabilidades e calibração

| Métrica | Por quê |
|---|---|
| `nll`, `brier`, `rps` | Regras de pontuação estritamente próprias (Gneiting & Raftery, 2007): só são minimizadas quando a probabilidade prevista é a verdadeira. Brier (Brier, 1950) é o erro quadrático das probabilidades; o RPS (Epstein, 1969) é o Brier nas distribuições acumuladas, que pune mais o erro distante numa escala ordinal. |
| `ece`, `mce`, `mean_confidence` | Erro de calibração esperado e máximo (Naeini et al., 2015; Guo et al., 2017, com 15 bins): "quando diz 80%, acerta 80%?". O próprio card do Laya diz que os checkpoints publicados são confiantes demais. |

### Top-k

Foram pedidos k = 1, 3, 5, 10, 15, 20. Existem dois sentidos de top-k, e o projeto registra os dois:

1. **Top-k de classe** (`top{k}_accuracy`, Russakovsky et al., 2015): o nível certo está entre os k mais prováveis?
   Com 3 níveis, só k = 1 tem sentido (k ≥ 3 dá 100% sempre), então só ele é registrado.
2. **Ranking dos "mais prováveis"** (`precision_at_k`, `recall_at_k`, `ndcg_at_k`): ordena os tickets pela
   probabilidade de serem `high` (`RANK_BY=prob`) e olha os k primeiros. É o caso de uso de verdade:
   "dos tickets que chegaram, quais atender primeiro?"
   - `precision@k`: dos k primeiros, quantos são realmente `high` (Manning et al., 2008).
   - `recall@k`: dos `high` que existem, quantos apareceram nos k primeiros.
   - `ndcg@k`: qualidade da ordem com relevância graduada, `high` > `medium` > `low` (Järvelin & Kekäläinen, 2002).
   - No teste inteiro (~2 mil tickets, ~40% `high`), os 20 primeiros seriam quase sempre `high` em qualquer modelo razoável.
     Por isso o cálculo simula **500 caixas de entrada de 100 tickets** sorteados (`RANK_POOLS`, `RANK_POOL_SIZE`) e tira a média.
   - `rank_auroc` e `rank_average_precision` resumem o ranking inteiro. AP é mais informativa que AUROC com classes
     desbalanceadas (Saito & Rehmsmeier, 2015).

### Custo

`latency_ms_per_item` (inferência em lote no GPU), `train/seconds`, `train/peak_vram_gb`, `model_download_mb`.

## Referências

- Baccianella, Esuli, Sebastiani (2009). *Evaluation Measures for Ordinal Regression.* ISDA.
- Brier (1950). *Verification of Forecasts Expressed in Terms of Probability.* Monthly Weather Review.
- Brodersen et al. (2010). *The Balanced Accuracy and Its Posterior Distribution.* ICPR.
- Chen et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174.
- Chicco, Jurman (2020). *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy.* BMC Genomics.
- Cohen (1960). *A Coefficient of Agreement for Nominal Scales.* Educational and Psychological Measurement.
- Cohen (1968). *Weighted kappa.* Psychological Bulletin.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Dodge et al. (2020). *Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping.* arXiv:2002.06305.
- Epstein (1969). *A Scoring System for Probability Forecasts of Ranked Categories.* Journal of Applied Meteorology.
- Gneiting, Raftery (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA.
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Howard, Ruder (2018). *Universal Language Model Fine-tuning for Text Classification.* ACL.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- Järvelin, Kekäläinen (2002). *Cumulated Gain-Based Evaluation of IR Techniques.* ACM TOIS.
- Kendall (1938). *A New Measure of Rank Correlation.* Biometrika.
- Kumar et al. (2022). *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR.
- Loshchilov, Hutter (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR.
- Loshchilov, Hutter (2019). *Decoupled Weight Decay Regularization.* ICLR.
- Manning, Raghavan, Schütze (2008). *Introduction to Information Retrieval*, cap. 8.
- Micikevicius et al. (2018). *Mixed Precision Training.* ICLR.
- Mosbach, Andriushchenko, Klakow (2021). *On the Stability of Fine-tuning BERT.* ICLR.
- Müller, Kornblith, Hinton (2019). *When Does Label Smoothing Help?* NeurIPS.
- Naeini, Cooper, Hauskrecht (2015). *Obtaining Well Calibrated Probabilities Using Bayesian Binning.* AAAI.
- Russakovsky et al. (2015). *ImageNet Large Scale Visual Recognition Challenge.* IJCV.
- Saito, Rehmsmeier (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE.
- Shao et al. (2024). *DeepSeekMath* (GRPO). arXiv:2402.03300.
- Sokolova, Lapalme (2009). *A systematic analysis of performance measures for classification tasks.* Information Processing & Management.
- Spearman (1904). *The Proof and Measurement of Association between Two Things.* American Journal of Psychology.
- Sun et al. (2019). *How to Fine-Tune BERT for Text Classification?* CCL.
- Warner et al. (2024). *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder* (ModernBERT). arXiv:2412.13663.
- Geuter, Cheng, Alvarez-Melis (2026). *Releasing the Inner Jev: A Simple Recipe for Open Decision Models.* arXiv:2610.17205.
- Weng, Liu, Chen, Manoharan (2026). *Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models.* arXiv:2610.29052.
- Laya: [model card](https://huggingface.co/convaiinnovations/laya), [guia de fine-tuning](https://nandhakishorm.github.io/laya/finetune/).
- clef-flash: [model card](https://huggingface.co/Cloudflare/clef-flash), [post da Cloudflare](https://blog.cloudflare.com/clef-decision-models).
