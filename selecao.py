"""Stepwise bidirecional por testes t parciais de regressão linear com intercepto."""
import numpy as np
from scipy.stats import t


def selecionar(x, y, alfa=0.05):
    matriz = x.to_numpy(dtype=float, copy=True)
    matriz -= matriz.mean(axis=0)
    normas = np.linalg.norm(matriz, axis=0)
    matriz /= np.where(normas > 0, normas, 1)
    resposta = np.asarray(y, dtype=float) - np.mean(y)
    gram = matriz.T @ matriz
    produto = matriz.T @ resposta
    soma_quadrados = resposta @ resposta
    ativos = []
    visitados = set()
    while True:
        estado = tuple(sorted(ativos))
        if estado in visitados:
            raise RuntimeError("Ciclo na seleção stepwise")
        visitados.add(estado)
        candidatos = np.array([j for j in range(x.shape[1]) if j not in ativos])
        if not len(candidatos):
            break
        variancia = gram.diagonal()[candidatos].copy()
        cov = produto[candidatos].copy()
        erro = soma_quadrados
        if ativos:
            cruzado = gram[np.ix_(ativos, candidatos)]
            ajuste = np.linalg.solve(gram[np.ix_(ativos, ativos)], cruzado)
            variancia -= (cruzado * ajuste).sum(axis=0)
            cov -= produto[ativos] @ ajuste
            beta = np.linalg.solve(gram[np.ix_(ativos, ativos)], produto[ativos])
            erro -= produto[ativos] @ beta
        validos = variancia > 1e-10
        if not validos.any():
            break
        candidatos = candidatos[validos]
        ganho = cov[validos] ** 2 / variancia[validos]
        melhor = np.argmax(ganho)
        gl = len(y) - len(ativos) - 2
        estatistica = np.sqrt(ganho[melhor] * gl / max(erro - ganho[melhor], 1e-15))
        if 2 * t.sf(estatistica, gl) >= alfa:
            break
        ativos.append(candidatos[melhor])
        while ativos:
            inversa = np.linalg.inv(gram[np.ix_(ativos, ativos)])
            beta = inversa @ produto[ativos]
            gl = len(y) - len(ativos) - 1
            mse = max(soma_quadrados - produto[ativos] @ beta, 1e-15) / gl
            p = 2 * t.sf(abs(beta) / np.sqrt(mse * inversa.diagonal()), gl)
            if max(p) < alfa:
                break
            ativos.pop(int(np.argmax(p)))
    return x.columns[ativos].tolist()
