# TE-CNN-AAE — Implementação e Reprodução do Experimento

Implementação do pipeline do artigo **"TE-CNN-AAE: Learning Robust Financial Time Series Representations with Trend-Enhanced Adversarial Autoencoders"** (submetido ao IEEE SMC 2025), executado em Google Colab com dados sintéticos (mock) que replicam a estrutura dos dados DJIA originais.

---

## Sobre o artigo

O artigo propõe o **TE-CNN-AAE** (Trend-Enhanced CNN Adversarial Autoencoder), um autoencoder adversarial que aprende representações de baixa dimensão de séries temporais intradiárias de mercado financeiro. A ideia central é treinar o encoder para, além de reconstruir a série de entrada, estimar simultaneamente a direção e força da tendência de preços — usando um **Trend Discriminator** como sinal adversarial.

O modelo é avaliado em dados de 5 minutos dos 30 ativos do índice Dow Jones Industrial Average (DJIA), cobrindo o período de janeiro de 2018 a dezembro de 2022.

---

## Arquitetura do modelo

```
Entrada: X_norm [B, 1, 79]   ← 79 pontos intradiários (5 min, 14:30–20:30 UTC), normalizados por dia

                    ┌─────────────────────────────┐
                    │          ENCODER             │
                    │  Conv1D(1→f1, k1) + ReLU    │
                    │  MaxPool(2)                  │
                    │  Conv1D(f1→f2, k2) + ReLU   │
                    │  MaxPool(2)                  │
                    │  Conv1D(f2→f3, k3) + ReLU   │
                    │  Flatten → Linear → z [B,64] │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
          ┌─────────▼──────────┐     ┌────────────▼──────────┐
          │      DECODER        │     │  TREND DISCRIMINATOR  │
          │  Linear → reshape   │     │  Dropout(entrada)     │
          │  ConvTranspose1d ×3 │     │  Linear(64) + ReLU   │
          │  Sigmoid            │     │  Linear(2) + Sigmoid  │
          │  x_hat [B,1,79]     │     │  [up, down] ∈ [0,1]  │
          └────────────────────┘     └───────────────────────┘

Loss total = λ_recon · MSE(x_hat, x) + λ_adv · MSE(trend_hat, y_trend)
```

O **CNN-AE** (ablação sem Trend Discriminator) usa a mesma estrutura encoder-decoder, mas sem o componente adversarial.

---

## Resultados obtidos (dados mock — 5 tickers)

> Os resultados abaixo são da execução com dados sintéticos (AAPL, MSFT, JPM, CVX, JNJ) e servem para validar o pipeline. Os valores do artigo original usam dados reais de 30 ativos DJIA via Polygon.io.

### Tabela III — Qualidade dos embeddings

| Modelo | Silhouette Score ↑ | Davies-Bouldin Index ↓ |
|---|---|---|
| **TE-CNN-AAE** | **0.3213** | **1.1985** |
| CNN-AE | 0.1403 | 2.4448 |
| PRD (paper original) | −0.0003 | 2.9394 |

O TE-CNN-AAE produziu embeddings com separação de clusters significativamente melhor que a ablação e que os dados brutos, consistente com os resultados reportados no artigo.

### Tabela V — Classificador LSTM (Mann-Whitney, α = 5%)

| Modelo | Supera RC | Supera PRD |
|---|---|---|
| **TE-CNN-AAE** | **100% (5/5)** | **60% (3/5)** |

### Hiperparâmetros encontrados pelo grid search

| Parâmetro | TE-CNN-AAE | CNN-AE (ablação) |
|---|---|---|
| Filtros (L1, L2, L3) | 32, 64, 128 | 32, 32, 128 |
| Kernels (L1, L2, L3) | 13, 7, 7 | 7, 7, 7 |
| z_dim | 64 | 64 |
| Dropout | 0.1 | 0.1 |

---

## Estrutura do repositório

```
.
├── TE_CNN_AAE.ipynb                  ← notebook principal (pipeline completo)
├── README.md
└── exp_tecnn_mock_protocol/          ← gerado ao executar o pipeline
    ├── data/
    │   ├── SUMMARY.csv
    │   └── {TICKER}/data_79.npz     ← séries intradiárias por ticker
    ├── search_autoencoders/
    │   ├── search_te/
    │   │   ├── best_te.json         ← melhor arquitetura TE-CNN-AAE
    │   │   ├── te_best/best_te.pt
    │   │   └── te_trials/           ← JSONs dos 216 trials
    │   └── search_ae/
    │       ├── best_ae.json         ← melhor arquitetura CNN-AE
    │       └── ae_trials/
    ├── models/
    │   ├── tecnn_aae_mock.pt        ← modelo TE-CNN-AAE treinado
    │   └── cnn_ae_mock.pt           ← modelo CNN-AE treinado
    ├── outputs/
    │   ├── embeddings_tecnn_mock.csv
    │   ├── embeddings_cnnae_mock.csv
    │   ├── labels_mock.csv
    │   ├── analyze/
    │   │   ├── tecnn/
    │   │   │   ├── metrics.json     ← Silhouette + Davies-Bouldin
    │   │   │   └── umap_*.png       ← Fig 4a
    │   │   └── cnnae/
    │   │       ├── metrics.json
    │   │       └── umap_*.png       ← Fig 4b
    │   └── lstmstudy_mock/
    │       ├── lstmstudy_summary.csv
    │       └── MannWhitney_Data/
    │           └── {TICKER}_eval.json
    └── state/
        └── manifest.json            ← flags de progresso (resumable)
```

---

## Como executar

### Pré-requisitos

- Google Colab (recomendado, GPU T4) ou ambiente local com Python 3.10+
- Sem dependência de dados externos — o pipeline usa dados sintéticos por padrão

### Dependências

```bash
pip install numpy pandas torch umap-learn scikit-learn matplotlib
```

### Execução rápida (dados mock, ~1–3 horas no Colab)

Execute todas as células do notebook. O ponto de entrada ao final da célula principal é:

```python
DJIA_30 = ["AAPL", "MSFT", "JPM", "CVX", "JNJ"]  # reduza para testar mais rápido

run_mock_pipeline_tecnn_protocol(
    exp_root="./exp_tecnn_mock_protocol",
    restore_zips=True,     # retoma de onde parou após reiniciar a sessão
    enable_bundles=True,   # salva ZIPs de checkpoint a cada etapa
    search_epochs=5,       # 20 no paper
    n_grid_repeats=3,      # 10 no paper
    n_eval_runs=20,        # 100 no paper
)
```

Para reproduzir o protocolo completo do artigo (30 tickers, 20 épocas, 100 runs), remova os parâmetros reduzidos e use a lista `DJIA_30` completa. Estimativa: **36–48 horas** de GPU.

### Resumabilidade

O pipeline é totalmente resumável. Se a sessão do Colab for interrompida, basta reexecutar — ele detecta automaticamente os ZIPs de checkpoint e continua da etapa onde parou, sem reprocessar o que já foi concluído.

### Visualização dos artefatos

Após o pipeline terminar, execute a célula de visualização (última célula do notebook) para gerar e exibir:

- **Fig 2** — Histograma de ângulos de tendência
- **Fig 4a/4b** — UMAP dos embeddings TE-CNN-AAE e CNN-AE
- **Fig 4c** — UMAP dos dados brutos (PRD)
- **Fig 5** — Heatmap de performance por ticker
- **Fig 6** — Boxplot de acurácias por abordagem
- **Tabela III** — Silhouette Score e Davies-Bouldin Index
- **Tabela V** — Win rates do teste Mann-Whitney

---

## Etapas do pipeline

| # | Etapa | Descrição | Saída principal |
|---|---|---|---|
| 1 | `mockdata` | Gera séries sintéticas de 79 pontos por dia | `data/{TICKER}/data_79.npz` |
| 2 | `searchae` | Grid search de 216 arquiteturas × 2 modelos | `search_autoencoders/*/best_*.json` |
| 3 | `train` | Treina TE-CNN-AAE com a melhor arquitetura | `models/tecnn_aae_mock.pt` |
| 4 | `embed` | Extrai embeddings z com o encoder treinado | `outputs/embeddings_tecnn_mock.csv` |
| 5 | `trainae` | Treina CNN-AE (ablação) | `models/cnn_ae_mock.pt` |
| 6 | `embed AE` | Extrai embeddings do CNN-AE | `outputs/embeddings_cnnae_mock.csv` |
| 7 | `analyze` | UMAP + Silhouette + Davies-Bouldin | `outputs/analyze/*/metrics.json` |
| 8 | `labels` | Labels 3-classes por entropia máxima | `outputs/labels_mock.csv` |
| 9 | `lstmstudy` | Grid search LSTM + 100 runs + Mann-Whitney | `outputs/lstmstudy_mock/` |

---

## Detalhes de implementação

**Dados de entrada:** 79 pontos de fechamento em intervalos de 5 minutos (14:30–20:30 UTC), com normalização min-max por dia. Dias com menos de 80% de completude ou sem cotação de abertura (14:30) são descartados.

**Split cronológico:** Os dados são divididos por data, não por amostra — todas as amostras de um mesmo dia ficam no mesmo split. Último 10% de dias únicos → teste holdout; últimos 10% dos dias de treino → validação.

**Trend Encoding:** Regressão linear sobre a série normalizada. O ângulo da reta ajustada é convertido em um vetor de duas componentes `[up, down]` ∈ [0,1], onde apenas uma componente é não-nula por série.

**Classificação downstream:** LSTM de 1 camada com janelas de W dias (W ∈ {5, 15, 30}), classificando o movimento do fechamento do dia seguinte em 3 classes: Increase / NoAction / Decrease. O limiar de classificação é determinado por maximização de entropia, calculado exclusivamente sobre o conjunto de treino.

**Teste estatístico:** Mann-Whitney U unilateral com α = 5%, comparando as distribuições de acurácia de 100 execuções independentes entre pares de abordagens (TE-CNN-AAE vs RC, TE-CNN-AAE vs PRD).

---

## Referências

- Araujo, J. O. A., Oliveira, A. L. I., Zanchettin, C. *TE-CNN-AAE: Learning Robust Financial Time Series Representations with Trend-Enhanced Adversarial Autoencoders*. IEEE SMC 2025 (submetido).
- Repositório oficial do artigo: [osf.io/rwtb3](https://osf.io/rwtb3/?view_only=b5fff19337034a7b849facc7b9d5fd37)
- Yıldırım, D. et al. *Forecasting Directional Movement of Forex Data Using LSTM with Technical and Macroeconomic Indicators*. Financial Innovation, 2021.
- Zeng, A. et al. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.

---

## Notas

- Os dados reais (Polygon.io) requerem assinatura paga. Esta implementação usa dados sintéticos que replicam a estrutura estatística das séries intradiárias para fins de validação do pipeline.
- O ticker DOW é excluído do estudo LSTM por histórico de negociação insuficiente no período coberto, conforme o artigo original.
- A implementação usa `torch`, não `tensorflow`. O decoder usa `ConvTranspose1d` com stride real, sem `F.interpolate`.
