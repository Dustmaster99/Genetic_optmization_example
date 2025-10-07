# Algoritmo Genético 3D para Maximização de Função

Este repositório contém a implementação de um **Algoritmo Genético (AG)** em Python para encontrar os pontos `(x, y)` que maximizam uma função de fitness 3D `Z = f(x, y)`.

O projeto também gera gráficos 3D da função e acompanha a evolução do fitness ao longo das gerações.

---

## Funcionalidades

- Visualização 3D da função `Z = f(x, y)` usando `matplotlib`.
- Inicialização de população aleatória de indivíduos `(x, y)`.
- Seleção de indivíduos por:
  - Roleta (proporcional ao fitness)
  - Torneio
  - Rank
- Crossover:
  - Média simples
  - Crossover aritmético
- Mutação:
  - Ruído uniforme
  - Ruído gaussiano
- Substituição:
  - Total
  - Elitismo
  - Parcial (steady-state)
- Registro histórico de:
  - Máximo de `Z` por geração
  - Média de `Z` por geração
  - Mínimo de `Z` por geração
- Identificação do **melhor indivíduo** da última geração
- Gráfico da evolução do fitness ao longo das gerações

---

## Resultados
Exemplo de resultado: 

![Gráfico 3D da função](https://github.com/user-attachments/assets/ba24fcf2-538e-4557-93a2-7ce287249059)

![Evolução do fitness](https://github.com/user-attachments/assets/8f8b45c1-4258-4665-8001-61398d083b88)

![Melhor indivíduo](https://github.com/user-attachments/assets/def760f8-6573-4c32-a6b0-e738ca7883d4)


## Estrutura da função `main`

```python
def main(
    fitness_function,
    x_range=(-5, 5),
    y_range=(-5, 5),
    num_pontos=100,
    init_pop_size=50,
    n_steps=50,
    n_selecionados=15,
    prob_crossover=0.8,
    prob_mutacao=0.1,
    max_ruido=0.5
):
    """
    Executa o algoritmo genético e plota gráficos.

    Parâmetros:
        fitness_function (callable): função que recebe x, y e retorna z
        x_range (tuple): intervalo de x para o gráfico 3D
        y_range (tuple): intervalo de y para o gráfico 3D
        num_pontos (int): número de pontos para plotagem 3D
        init_pop_size (int): tamanho da população inicial
        n_steps (int): número de gerações
        n_selecionados (int): número de indivíduos selecionados
        prob_crossover (float): probabilidade de crossover
        prob_mutacao (float): probabilidade de mutação
        max_ruido (float): amplitude do ruído da mutação uniforme
    """


