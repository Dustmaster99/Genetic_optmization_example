import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
função main implementa um algoritmo genético (AG) para encontrar os pontos (x, y) que maximizam uma função de fitness Z = f(x, y). Além disso, ela:

Plota a superfície 3D da função.

Inicializa uma população de indivíduos.

Aplica repetidamente seleção, crossover, mutação e substituição.

Mantém histórico das gerações.

Mostra a evolução do fitness ao longo das gerações.

Identifica o melhor indivíduo da última geração.
'''



#------------------------------------------------------------------------------------------------------------------------------------------------#
# Definindo uma função capaz de gerar uma população inicial aleatória de n pontos no plano xy, que são candidatos a solução de maximização do algoritmo

def gerar_populacao_inicial_3d(n, x_range=(-5, 5), y_range=(-5, 5)):
    """
    Gera uma população inicial de n pontos aleatórios no plano XY,
    incluindo o valor da função fitness como coordenada Z.

    Parâmetros:
        n (int): número de indivíduos (pontos)
        x_range (tuple): intervalo (min, max) de x
        y_range (tuple): intervalo (min, max) de y

    Retorna:
        populacao_3d (ndarray): array (n, 3) com coordenadas (x, y, z)
    """
    # Gerando x e y aleatórios
    x_vals = np.random.uniform(x_range[0], x_range[1], n)
    y_vals = np.random.uniform(y_range[0], y_range[1], n)
    
    # Calculando z usando a função fitness
    z_vals = fitness_function(x_vals, y_vals)
    
    # Combinando x, y e z em um array (n, 3)
    populacao_3d = np.column_stack((x_vals, y_vals, z_vals))
    
    return populacao_3d


#------------------------------------------------------------------------------------------------------------------------------------------------#
# Definindo uma função capaz de gerar uma população inicial aleatória de n pontos no plano xy, que são candidatos a solução de maximização do algoritmo
def selecao_roleta(populacao_3d, n_selecionados):
    """
    Seleciona indivíduos da população 3D com probabilidade proporcional ao valor de Z (fitness).

    Parâmetros:
        populacao_3d (ndarray): array (num_individuos, 3) com (x, y, z)
        n_selecionados (int): número de indivíduos a serem selecionados

    Retorna:
        selecionados (ndarray): array (n_selecionados, 3) com indivíduos escolhidos
    """
    z_vals = populacao_3d[:, 2]
    
    # Garantir que os valores sejam positivos para roleta
    z_min = z_vals.min()
    if z_min < 0:
        z_vals = z_vals - z_min  # desloca para que o menor valor seja 0

    # Evitar divisão por zero se todos z forem iguais
    if z_vals.sum() == 0:
        probabilidades = np.ones_like(z_vals) / len(z_vals)
    else:
        probabilidades = z_vals / z_vals.sum()

    # Seleção com base nas probabilidades
    indices_selecionados = np.random.choice(len(populacao_3d), size=n_selecionados, p=probabilidades)
    selecionados = populacao_3d[indices_selecionados]

    return selecionados

#------------------------------------------------------------------------------------------------------------------------------------------------#
def selecao_torneio(populacao_3d, n_selecionados, tamanho_torneio=3):
    """
    Seleciona indivíduos usando torneio baseado no valor de Z (fitness).

    Parâmetros:
        populacao_3d (ndarray): array (num_individuos, 3) com (x, y, z)
        n_selecionados (int): número de indivíduos a serem selecionados
        tamanho_torneio (int): tamanho de cada torneio

    Retorna:
        selecionados (ndarray): array (n_selecionados, 3) com indivíduos escolhidos
    """
    selecionados = []
    n = len(populacao_3d)

    for _ in range(n_selecionados):
        # Escolhe aleatoriamente 'tamanho_torneio' indivíduos
        indices_torneio = np.random.choice(n, size=tamanho_torneio, replace=False)
        torneio = populacao_3d[indices_torneio]
        
        # Seleciona o melhor (maior z)
        melhor_indice = np.argmax(torneio[:, 2])
        selecionados.append(torneio[melhor_indice])
    
    return np.array(selecionados)

#------------------------------------------------------------------------------------------------------------------------------------------------#
def selecao_rank(populacao_3d, n_selecionados):
    """
    Seleciona indivíduos usando seleção por rank.

    Parâmetros:
        populacao_3d (ndarray): array (num_individuos, 3) com (x, y, z)
        n_selecionados (int): número de indivíduos a serem selecionados

    Retorna:
        selecionados (ndarray): array (n_selecionados, 3) com indivíduos escolhidos
    """
    n = len(populacao_3d)
    
    # Ordena pelo valor de Z (maior primeiro)
    indices_ordenados = np.argsort(populacao_3d[:, 2])[::-1]
    populacao_ordenada = populacao_3d[indices_ordenados]

    # Cria probabilidades proporcionais ao rank (maior rank = maior chance)
    ranks = np.arange(1, n+1)  # rank 1 para o menor, n para o maior
    probabilidades = ranks / ranks.sum()
    
    # Seleção com base nas probabilidades de rank
    indices_selecionados = np.random.choice(n, size=n_selecionados, p=probabilidades)
    selecionados = populacao_ordenada[indices_selecionados]
    
    return selecionados
#------------------------------------------------------------------------------------------------------------------------------------------------#
def crossover_media(populacao_3d, fitness_function, prob_crossover= 0.8):
    """
    Aplica crossover usando a média simples entre dois pais com probabilidade prob_crossover.

    Parâmetros:
        populacao_3d (ndarray): array (n, 3) com (x, y, z)
        fitness_function (callable): função que recebe x, y e retorna z
        prob_crossover (float): probabilidade de aplicar crossover a cada indivíduo (0 a 1)

    Retorna:
        nova_populacao (ndarray): array (n, 3) com nova população
    """
    n = len(populacao_3d)
    nova_populacao = np.zeros_like(populacao_3d)
    
    for i in range(n):
        if np.random.rand() < prob_crossover:
            # Aplica crossover
            pais = populacao_3d[np.random.choice(n, size=2, replace=False)]
            x_filho = (pais[0, 0] + pais[1, 0]) / 2
            y_filho = (pais[0, 1] + pais[1, 1]) / 2
        else:
            # Copia indivíduo da população original
            x_filho, y_filho = populacao_3d[i, 0], populacao_3d[i, 1]
        
        z_filho = fitness_function(x_filho, y_filho)
        nova_populacao[i] = [x_filho, y_filho, z_filho]
    
    return nova_populacao


#------------------------------------------------------------------------------------------------------------------------------------------------#
def crossover_aritmetico(populacao_3d, fitness_function, prob_crossover= 0.8):
    """
    Aplica crossover aritmético na população 3D usando alpha aleatório com probabilidade prob_crossover.

    Parâmetros:
        populacao_3d (ndarray): array (n, 3) com (x, y, z)
        fitness_function (callable): função que recebe x, y e retorna z
        prob_crossover (float): probabilidade de aplicar crossover a cada indivíduo (0 a 1)

    Retorna:
        nova_populacao (ndarray): array (n, 3) com nova população
    """
    n = len(populacao_3d)
    nova_populacao = np.zeros_like(populacao_3d)
    
    for i in range(n):
        if np.random.rand() < prob_crossover:
            # Aplica crossover aritmético
            pais = populacao_3d[np.random.choice(n, size=2, replace=False)]
            alpha = np.random.rand()
            x_filho = alpha * pais[0, 0] + (1 - alpha) * pais[1, 0]
            y_filho = alpha * pais[0, 1] + (1 - alpha) * pais[1, 1]
        else:
            # Copia indivíduo da população original
            x_filho, y_filho = populacao_3d[i, 0], populacao_3d[i, 1]
        
        z_filho = fitness_function(x_filho, y_filho)
        nova_populacao[i] = [x_filho, y_filho, z_filho]
    
    return nova_populacao
#------------------------------------------------------------------------------------------------------------------------------------------------#
def mutacao_uniforme(populacao_3d, fitness_function, prob_mutacao=0.1, max_ruido=1.0):
    """
    Aplica mutação real aleatória com ruído uniforme na população 3D.

    Parâmetros:
        populacao_3d (ndarray): array (n, 3) com (x, y, z)
        fitness_function (callable): função que recebe x, y e retorna z
        prob_mutacao (float): probabilidade de cada indivíduo sofrer mutação (0 a 1)
        max_ruido (float): valor máximo do ruído a ser adicionado/subtraído a x e y

    Retorna:
        nova_populacao (ndarray): array (n, 3) com população mutada
    """
    nova_populacao = np.copy(populacao_3d)
    n = len(populacao_3d)

    for i in range(n):
        if np.random.rand() < prob_mutacao:
            ruido_x = np.random.uniform(-max_ruido, max_ruido)
            ruido_y = np.random.uniform(-max_ruido, max_ruido)
            x_novo = nova_populacao[i, 0] + ruido_x
            y_novo = nova_populacao[i, 1] + ruido_y
            z_novo = fitness_function(x_novo, y_novo)
            nova_populacao[i] = [x_novo, y_novo, z_novo]

    return nova_populacao
#------------------------------------------------------------------------------------------------------------------------------------------------#
def mutacao_gaussiana(populacao_3d, fitness_function, prob_mutacao=0.1, sigma=1.0, mean = 0):
    """
    Aplica mutação real aleatória com ruído gaussiano na população 3D.

    Parâmetros:
        populacao_3d (ndarray): array (n, 3) com (x, y, z)
        fitness_function (callable): função que recebe x, y e retorna z
        prob_mutacao (float): probabilidade de cada indivíduo sofrer mutação (0 a 1)
        sigma (float): desvio padrão do ruído gaussiano aplicado a x e y

    Retorna:
        nova_populacao (ndarray): array (n, 3) com população mutada
    """
    nova_populacao = np.copy(populacao_3d)
    n = len(populacao_3d)

    for i in range(n):
        if np.random.rand() < prob_mutacao:
            ruido_x = np.random.normal(mean, sigma)
            ruido_y = np.random.normal(mean, sigma)
            x_novo = nova_populacao[i, 0] + ruido_x
            y_novo = nova_populacao[i, 1] + ruido_y
            z_novo = fitness_function(x_novo, y_novo)
            nova_populacao[i] = [x_novo, y_novo, z_novo]

    return nova_populacao
#------------------------------------------------------------------------------------------------------------------------------------------------#
def substituicao_total(pais, filhos):
    """
    Substituição total: toda a população de pais é substituída pelos filhos.
    """
    return np.copy(filhos)
#------------------------------------------------------------------------------------------------------------------------------------------------#
def substituicao_elitismo(pais, filhos, n_elite=5):
    """
    Mantém os 'n_elite' melhores pais e completa a população com os filhos.
    
    Parâmetros:
        pais (ndarray): população de pais (n, 3)
        filhos (ndarray): população de filhos (n, 3)
        n_elite (int): número de melhores pais a manter
    """
    n_total = len(pais)
    
    # Ordena pais pelo valor de z (maior primeiro)
    indices_ordenados = np.argsort(pais[:, 2])[::-1]
    elite = pais[indices_ordenados[:n_elite]]
    
    # Preenche o restante com filhos
    n_filhos_necessarios = n_total - n_elite
    novos_individuos = filhos[:n_filhos_necessarios]
    
    nova_populacao = np.vstack((elite, novos_individuos))
    return nova_populacao
#------------------------------------------------------------------------------------------------------------------------------------------------#
def substituicao_parcial(pais, filhos, n_substituir= 10):
    """
    Substituição parcial (Steady-State): substitui apenas os piores pais pelos melhores filhos.

    Parâmetros:
        pais (ndarray): população de pais (n, 3)
        filhos (ndarray): população de filhos (n, 3)
        n_substituir (int, opcional): número de indivíduos a substituir. 
                                      Se None, substitui metade da população.

    Retorna:
        nova_populacao (ndarray): população após substituição
    """
    n_total = len(pais)
    if n_substituir is None:
        n_substituir = n_total // 2  # por padrão, metade da população

    nova_populacao = np.copy(pais)

    # Ordena pais pelo valor de z (menor primeiro) - piores
    indices_piores = np.argsort(pais[:, 2])[:n_substituir]

    # Ordena filhos pelo valor de z (maior primeiro) - melhores
    indices_melhores_filhos = np.argsort(filhos[:, 2])[::-1][:n_substituir]

    # Substitui os piores pais pelos melhores filhos
    nova_populacao[indices_piores] = filhos[indices_melhores_filhos]

    return nova_populacao






#------------------------------------------------------------------------------------------------------------------------------------------------#

def main(fitness_function,
         x_range=(-5, 5),
         y_range=(-5, 5),
         num_pontos=100,
         init_pop_size=50,
         n_steps=50,
         n_selecionados=15,
         prob_crossover=0.8,
         prob_mutacao=0.1,
         max_ruido=0.5):
    """
    Função principal para executar algoritmo genético e plotar gráficos.
    
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

    # --- Gráfico 3D da função ---
    x = np.linspace(x_range[0], x_range[1], num_pontos)
    y = np.linspace(y_range[0], y_range[1], num_pontos)
    X, Y = np.meshgrid(x, y)
    Z = fitness_function(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("Superfície 3D da Função Z")
    ax.set_xlabel("Eixo X")
    ax.set_ylabel("Eixo Y")
    ax.set_zlabel("Eixo Z")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.show()

    # Valor máximo da função no grid
    i, j = np.unravel_index(np.argmax(Z), Z.shape)
    x_max, y_max, z_max = X[i, j], Y[i, j], Z[i, j]
    print(f"Máximo da função em grid: x={x_max:.3f}, y={y_max:.3f}, z={z_max:.3f}")

    # --- Algoritmo genético ---
    # Inicializa população
    populacao = np.random.uniform(x_range[0], x_range[1], (init_pop_size, 2))
    populacao_3d = np.column_stack((populacao, fitness_function(populacao[:,0], populacao[:,1])))

    # Histórico
    historico_max_z = []
    historico_media_z = []
    historico_min_z = []

    pais = populacao_3d

    for t in range(n_steps):
        # ----------------------------
        # 1 - Seleção (apenas um ativo, outros comentados)
        filhos = selecao_roleta(pais, n_selecionados)
        #filhos = selecao_torneio(pais, n_selecionados)
        # filhos = selecao_rank(pais, n_selecionados)

        # ----------------------------
        # 2 - Crossover (apenas um ativo, outros comentados)
        #filhos = crossover_media(filhos, fitness_function, prob_crossover=prob_crossover)
        filhos = crossover_aritmetico(filhos, fitness_function, prob_crossover=prob_crossover)

        # ----------------------------
        # 3 - Mutação (apenas uma ativa, outros comentados)
        #filhos = mutacao_uniforme(filhos, fitness_function, prob_mutacao=prob_mutacao, max_ruido=max_ruido)
        filhos = mutacao_gaussiana(filhos, fitness_function, prob_mutacao=prob_mutacao, sigma=max_ruido)

        # ----------------------------
        # 4 - Substituição (comentado para mostrar opções)
        # filhos = substituicao_total(pais, filhos)
        filhos = substituicao_elitismo(pais, filhos)
        #filhos = substituicao_parcial(pais, filhos)

        # Atualiza pais
        pais = filhos

        # ----------------------------
        # Salva histórico
        z_vals = pais[:, 2]
        historico_max_z.append(np.max(z_vals))
        historico_media_z.append(np.mean(z_vals))
        historico_min_z.append(np.min(z_vals))

    # --- Melhor indivíduo da última geração ---
    melhor_individuo = pais[np.argmax(pais[:, 2])]
    x_melhor, y_melhor, z_melhor = melhor_individuo
    print("\nMelhor indivíduo da última geração:")
    print(f"x = {x_melhor:.4f}, y = {y_melhor:.4f}, z = {z_melhor:.4f}")

    # --- Gráfico da evolução do fitness ---
    plt.figure(figsize=(10,6))
    plt.plot(historico_max_z, label='Máximo de Z')
    plt.plot(historico_media_z, label='Média de Z')
    plt.plot(historico_min_z, label='Mínimo de Z')
    plt.xlabel('Geração')
    plt.ylabel('Fitness (Z)')
    plt.title('Evolução do Fitness ao longo das gerações')
    plt.legend()
    plt.grid(True)
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------#

# Definindo a função 3D Z = f(x,y)
def fitness_function(x, y):
    return np.sin(x) * np.cos(y) + 1.5 * np.exp(-(x**2 + y**2) / 10)

main(fitness_function,
     x_range=(-5,5),
     y_range=(-5,5),
     num_pontos=1000,
     init_pop_size=50,
     n_steps=50,
     n_selecionados=15,
     prob_crossover=0.8,
     prob_mutacao=0.1,
     max_ruido=0.5)

