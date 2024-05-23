import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np



def alinha_face(imagem,dets,tamanho=None, return_face=True):
    """
    Alinha a face de uma imagem. Só funciona se tiver apenas 1 face
    Argumentos:
    imagem-> imagem em cinza
    dets->resultado de detector(gray)
    tamanho (tupla, opcional): As dimensões (largura, altura) desejadas para a imagem alinhada. Se não for especificado, a imagem será mantida no tamanho original.
    return_face (bool, opcional): Se True, retorna a imagem alinhada. Se False (padrão), Imprime a imagem e retorna None.

    Retorna:
    lista de dicionarios no seguinte formato
    [{'imagem':img,'coordinates':{'x':int,'y':int,'w':int,'h',int}}]
    Se return_face for True retornara a lista com as informações da imagem
    Se return_face for False, imprime a foto original, o ROI e a imagem alinhada da ultima face encontrada e retorna None.
    """
    #Configura lista para retornar
    faces=[]
    # Carrega o classificador de marcos faciais
    classificador_68_path = r"C:\Users\alber\Documents\Projetos - Dados\computer_vision\Notebooks\files\classificadores\shape_predictor_68_face_landmarks.dat"
    # detector_face_dlib = dlib.get_frontal_face_detector()
    classificador_dlib_68 = dlib.shape_predictor(classificador_68_path)

    # ## print(type(imagem))
    # # Salva o tamanho da imagem
    height, width = imagem.shape[:2]
    # # Detecta rostos na imagem
    # dets = detector_face_dlib(imagem, 1)

   
    # Para C
    for i, det in enumerate(dets):
        # Obtém as coordenadas do retângulo delimitador do rosto
        x = det.left()
        y = det.top()
        w = det.right() - x
        h = det.bottom() - y

        # Recorta o ROI do rosto
        if x < 0: x = 0
        if y < 0: y = 0
        roi = imagem[y:y+h, x:x+w]

        # Detecta os marcos faciais
        shape = classificador_dlib_68(imagem, det)
        left_eye = extrair_olho_centro_esquerdo(shape)
        right_eye = extrair_olho_centro_direito(shape)

        # Obtém a matriz de rotação e aplica a rotação na imagem
        M = get_rotation_matrix(left_eye, right_eye)

        # Define o novo tamanho caso informado pelo usuário
        if tamanho:
            width = tamanho[0]
            height = tamanho[1]
        # Rotaciona imagem
        rotated = cv2.warpAffine(imagem, M, (width, height))
        
        # Recorta a imagem alinhada
        cropped = rotated[y:y+h, x:x+w]
        faces.append({'image':cropped, 'coordinates':{'x':x,'y':y,'w':w,'h':h},'face':det})
    # Retorna imagem caso requisitado
    if return_face:
        return faces
    # Imprime comparação
    plt.subplot(1,3,1)
    
    plt.title("Original")
    plt.imshow(imagem)

    plt.subplot(1,3,2)
    plt.title("Roi")
    plt.imshow(roi)

    plt.subplot(1,3,3)
    plt.title("Alinhado")
    plt.imshow(cropped)
    plt.show()
    
    return None

def numero_de_faces(imagem):
    """ Funcao que retorna o numero de imagens em uma face
    Variavel
    imagem - a imagem em formato cv2
    Ex:
    image_path=r'folder/a.png'
    imagem = cv2.imread(image_path)
    tem_face(imagem)

    """
    classificador_68_path = r"C:\Users\alber\Documents\Projetos - Dados\computer_vision\Notebooks\files\classificadores\shape_predictor_68_face_landmarks.dat"
    detector_face_dlib = dlib.get_frontal_face_detector()

    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    # Detecta rostos na imagem
    dets = detector_face_dlib(imagem, 1)
    return len(dets)




# Função para extrair os pontos do olho
def extrair_olho(shape, eye_indices):
    """
    Extrai os pontos faciais que representam um olho com base nos índices fornecidos.

    Argumentos:
    shape (objeto): Objeto contendo os pontos faciais detectados.
    eye_indices (lista): Índices dos pontos faciais que representam o olho.

    Retorna:
    lista: Lista de pontos faciais que representam o olho.
    """
    # Mapeia os pontos do olho usando os índices fornecidos
    points = map(lambda i: shape.part(i), eye_indices)
    return list(points)

# Função para extrair o centro do olho
def extrair_olho_centro(shape, eye_indices):
    """
    Calcula o centro de um olho com base nos pontos faciais detectados.

    Argumentos:
    shape (objeto): Objeto contendo os pontos faciais detectados.
    eye_indices (lista): Índices dos pontos faciais que representam o olho.

    Retorna:
    tuple: Uma tupla contendo as coordenadas (x, y) do centro do olho.
    """
    # Extrai os pontos do olho
    points = extrair_olho(shape, eye_indices)
    # Extrai os valores x e y dos pontos
    xs = map(lambda p: p.x, points)
    ys = map(lambda p: p.y, points)
    # Calcula a média dos valores x e y para encontrar o centro
    return sum(xs) // 6, sum(ys) // 6

# Funções específicas para extrair o centro do olho esquerdo e direito
def extrair_olho_centro_esquerdo(shape):
    """
    Calcula o centro do olho esquerdo com base nos pontos faciais detectados.

    Argumentos:
    shape (objeto): Objeto contendo os pontos faciais detectados.

    Retorna:
    tuple: Uma tupla contendo as coordenadas (x, y) do centro do olho esquerdo.
    """
    #Define os range do olho esquerdo
    OLHO_ESQUERDO = list(range(42, 48))

    return extrair_olho_centro(shape, OLHO_ESQUERDO)

def extrair_olho_centro_direito(shape):
    """
    Calcula o centro do olho direito com base nos pontos faciais detectados.

    Argumentos:
    shape (objeto): Objeto contendo os pontos faciais detectados.

    Retorna:
    tuple: Uma tupla contendo as coordenadas (x, y) do centro do olho direito.
    """
    #Define o range do olho direito
    OLHO_DIREITO = list(range(36, 42))
    return extrair_olho_centro(shape, OLHO_DIREITO)

# Função para calcular o ângulo entre dois pontos
def angulo_entre_pontos(p1, p2):
    """
    Calcula o ângulo entre dois pontos.

    Argumentos:
    p1 (tuple): Coordenadas (x, y) do primeiro ponto.
    p2 (tuple): Coordenadas (x, y) do segundo ponto.

    Retorna:
    float: O ângulo em graus entre os dois pontos.
    """
    x1, y1 = p1
    x2, y2 = p2
    # Calcula a tangente do ângulo entre os pontos
    tan = (y2 - y1) / (x2 - x1)
    # Converte o ângulo em graus
    return np.degrees(np.arctan(tan))

# Função para obter a matriz de rotação
def get_rotation_matrix(p1, p2):
    """
    Calcula a matriz de rotação para rotacionar um objeto de acordo com a inclinação entre dois pontos.

    Argumentos:
    p1 (tuple): Coordenadas (x, y) do primeiro ponto.
    p2 (tuple): Coordenadas (x, y) do segundo ponto.

    Retorna:
    numpy.ndarray: A matriz de rotação.
    """
    # Calcula o ângulo entre os pontos
    angle = angulo_entre_pontos(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    # Calcula o ponto médio entre os dois pontos
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    # Obtém a matriz de rotação usando o ângulo e o ponto médio
    M = cv2.getRotationMatrix2D((xc, yc), angle, 1)
    return M

if __name__=='__main__':

    image = cv2.imread('notebooks/files/nic1.jpg')
    alinha_face(image,return_face=False)
    print(type(alinha_face('notebooks/files/nic1.jpg')))
