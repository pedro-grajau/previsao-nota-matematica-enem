"""Filtragem dos CSVs nacionais em blocos e codificação do questionário."""
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ALVO = "NU_NOTA_MT"
NOTAS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_REDACAO"]
QUESTIONARIO = [f"Q{i:03}" for i in range(1, 26)]
CATEGORIAS = [
    "TP_SEXO", "TP_ESCOLA", "TP_LINGUA", "TP_COR_RACA", "TP_ESTADO_CIVIL",
    "TP_FAIXA_ETARIA", "TP_LOCALIZACAO_ESC", "TP_DEPENDENCIA_ADM_ESC",
    "TP_NACIONALIDADE", "TP_SIT_FUNC_ESC",
] + [q for q in QUESTIONARIO if q != "Q005"]
FILTROS = ["SG_UF_ESC", "IN_TREINEIRO", "TP_ENSINO", "TP_ST_CONCLUSAO", "NU_ANO"]
COLUNAS = FILTROS + CATEGORIAS + NOTAS + ["Q005", ALVO]


def carregar(pasta, ano):
    arquivo = f"MICRODADOS_ENEM_{ano}.csv"
    csv = pasta / arquivo
    if csv.exists():
        with csv.open("rb") as entrada:
            return filtrar(entrada, ano)
    candidatos = [pasta / f"microdados_enem_{ano}.zip", pasta / f"enem_{ano}.zip"]
    zip_path = next((p for p in candidatos if p.exists()), None)
    if zip_path is None:
        raise FileNotFoundError(f"Forneça {arquivo} ou o ZIP do Inep em {pasta}")
    with ZipFile(zip_path) as pacote:
        membro, = [n for n in pacote.namelist() if Path(n).name.lower() == arquivo.lower()]
        with pacote.open(membro) as entrada:
            return filtrar(entrada, ano)


def filtrar(entrada, ano):
    partes = []
    total = 0
    tipos = {c: "string" for c in CATEGORIAS}
    for bloco in pd.read_csv(entrada, sep=";", encoding="latin1", usecols=COLUNAS,
                             dtype=tipos, chunksize=100_000):
        if not bloco.NU_ANO.eq(ano).all():
            raise ValueError(f"Arquivo incompatível com a edição {ano}")
        total += len(bloco)
        manter = (bloco.SG_UF_ESC.eq("MA") & bloco[ALVO].gt(0) & bloco.TP_ENSINO.eq(1)
                  & bloco.IN_TREINEIRO.eq(0) & bloco.TP_ST_CONCLUSAO.eq(2))
        partes.append(bloco.loc[manter].drop(columns=FILTROS).dropna())
    dados = pd.concat(partes, ignore_index=True)
    if dados.empty:
        raise ValueError(f"Nenhum participante elegível em {ano}")
    return dados, {"ano": ano, "inscritos": total, "elegiveis": len(dados)}


def codificar(dados):
    # As letras preservam o vínculo com o dicionário de cada edição. A troca
    # por índices ordinais antes das dummies produziria os mesmos indicadores.
    return pd.get_dummies(dados, columns=CATEGORIAS, dtype=float)
