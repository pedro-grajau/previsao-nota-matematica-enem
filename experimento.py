"""Reconstrução do experimento de regressão para concluintes do Maranhão."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pycaret.regression import RegressionExperiment
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preparar import ALVO, carregar, codificar
from selecao import selecionar


def executar(dados, saida):
    saida.mkdir(exist_ok=True, parents=True)
    edicoes = {}
    contagens = []
    for ano in (2021, 2022, 2023):
        print(f"Lendo {ano}", flush=True)
        base, resumo = carregar(dados, ano)
        edicoes[ano] = codificar(base)
        contagens.append(resumo)
    pd.DataFrame(contagens).to_csv(saida / "amostra.csv", index=False)
    base = edicoes[2021]
    nomes = selecionar(base.drop(columns=ALVO), base[ALVO])
    pd.Series(nomes, name="variavel").to_csv(saida / "selecao.csv", index=False)
    colunas = nomes + [ALVO]
    treino = edicoes[2022].reindex(columns=colunas, fill_value=0)
    teste = edicoes[2023].reindex(columns=colunas, fill_value=0)
    exp = RegressionExperiment()
    exp.setup(data=treino, test_data=teste, target=ALVO, index=False,
              imputation_type=None, remove_multicollinearity=True,
              multicollinearity_threshold=0.95, normalize=True, normalize_method="minmax",
              transform_target=False, fold=10, session_id=42, n_jobs=1,
              html=False, verbose=False)
    exp.compare_models(include=["lightgbm", "gbr", "xgboost", "rf", "lr"],
                       fold=10, sort="R2", round=6, errors="raise", verbose=False)
    exp.pull().to_csv(saida / "validacao_cruzada.csv", index_label="id")
    modelo = exp.create_model("lightgbm", fold=10, round=3, verbose=False)
    exp.pull().to_csv(saida / "folds_lightgbm.csv")
    pred = exp.predict_model(modelo, data=teste, verbose=False)["prediction_label"]
    real = teste[ALVO]
    mse = mean_squared_error(real, pred)
    metricas = dict(MAE=mean_absolute_error(real, pred), MSE=mse, RMSE=mse**0.5,
                    R2=r2_score(real, pred))
    pd.DataFrame([metricas]).to_csv(saida / "teste_2023.csv", index=False)
    exp.save_model(modelo, str(saida / "modelo_gbm"), verbose=False)
    fig, ax = plt.subplots()
    pontos = ax.hexbin(pred, real - pred, gridsize=48, bins="log", mincnt=1)
    fig.colorbar(pontos, ax=ax, label="Participantes por célula")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Nota prevista", ylabel="Nota observada − prevista", title="Resíduos no ENEM 2023")
    fig.savefig(saida / "residuos.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    ranking = pd.Series(modelo.feature_importances_, index=exp.get_config("X_train_transformed").columns)
    ranking.sort_values(ascending=False).rename_axis("variavel").rename("divisoes").to_csv(saida / "importancia.csv")
    ax = ranking.nlargest(10).sort_values().plot.barh(color="#166c75")
    ax.set(xlabel="Número de divisões nas árvores", ylabel="", title="Importância no LightGBM")
    ax.figure.savefig(saida / "importancia.png", bbox_inches="tight", dpi=150)
    plt.close(ax.figure)
    print(metricas, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dados", type=Path, required=True)
    cli.add_argument("--saida", type=Path, default=Path("resultados/reproducao"))
    args = cli.parse_args()
    executar(args.dados, args.saida)
