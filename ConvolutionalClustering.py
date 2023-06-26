import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import time
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import label
import seaborn as sns


# Scale the datafram to desired scale and build a matrix for the convolutioin
def transform_df(df, scale_size=100, pad=None, 
                 cols=None, plot_matrix=False):
    if cols is None:
        col1, col2 = df.columns[0], df.columns[1]
    else:
        col1, col2 = cols[0], cols[1]
    if pad == None:
        pad = int(scale_size * 0.1)
    inner_size = int(scale_size * 0.8)
    # Calcolo dei valori minimi e massimi delle colonne x e y
    min_value = df[[f'{col1}', f'{col2}']].min().min()
    max_value = df[[f'{col1}', f'{col2}']].max().max()

    df_s = df.add( - df.min() )
    df_m = df_s.multiply( inner_size / df_s.max().max() )
    
    matrix_shape = tuple([inner_size + 2 * pad] * 2)
    matrix = np.zeros(matrix_shape)

    # Posiziona gli '1' nella matrice in base alle coordinate del DataFrame
    for _, row in df_m.iterrows():
        x = int(row[f'{col1}'])
        y = int(row[f'{col2}'])
        matrix[pad + y-1, pad + x-1] = 255
        
    if plot_matrix:
        plt.imshow(matrix, origin='lower')
        plt.title(f'scale_size={scale_size}  pad={pad}')
        
    return df_m, matrix, pad

# show image
def show_img(x, ax=None, figsize=(5, 5), title=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()

    if x.dtype != np.uint8:
        x = x.astype(np.uint8)

    if len(x.shape) == 3 and x.shape[2] == 3:
        # Immagine a colori (BGR)
        ax.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    else:
        # Immagine in scala di grigi
        ax.imshow(x, cmap='gray')

    if title:
        ax.set_title(title)
        
# Utility Functions
def create_gaussian_kernel(size, sigma):
    # Calcola il filtro gaussiano unidimensionale
    gaussian_1d = cv2.getGaussianKernel(size, sigma)
    
    # Calcola il prodotto esterno per ottenere il filtro gaussiano bidimensionale
    gaussian_kernel = np.outer(gaussian_1d, gaussian_1d)
    
    return gaussian_kernel

def generate_colors(num_clusters):
    colors = [tuple(np.random.randint(0, 255, 3)) for x in range(num_clusters)]
    return np.array(colors).tolist()

# ---------------------- CONVOLUTIONAL CLUSTERING (ONLY VISUALIZATION) ----------------------
def CC(df, cols=None,
       scale_size=100, pad=None, plot_matrix=False,
       kernel_type='mean', k_size=None, sigma=20,
       ratio_scaler=0.9, THR=100, loops=2,
       draw_type='line', radius=2, thickness=2,
       show_clusters=False, plot_scatter=True):
    
    if cols is None:
        col1, col2 = df.columns[0], df.columns[1]
    else:
        col1, col2 = cols[0], cols[1]
    df_scaled, m, pad_value = transform_df(df, cols=cols, scale_size=scale_size, pad=pad, plot_matrix=plot_matrix)
    
    m = np.flipud(m)
    image = np.uint8(m)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # kernel
    if k_size is None:
        k_size = int(np.sqrt(np.min(m.shape)) * 2)
    # kernel (size x size)
    if kernel_type == 'mean':
        kernel = np.ones((k_size, k_size), np.float32) / int(k_size ** 2)
    elif kernel_type == 'gaussian':
        kernel = create_gaussian_kernel(k_size, sigma)
    else:
        raise ValueError('Wrong kernel_type chosen.')
        

    # concolutional filtering loops
    blobs = image.copy()


    for l in range(loops):
        # Apply filter
        blobs = cv2.filter2D(blobs, -1, kernel)
        # Normalize at 255 an THR
        MAX = np.max(blobs)
        ratio = (255 * ratio_scaler) / MAX
        blobs = blobs * ratio
        blobs = np.round(blobs).astype(np.uint8)
        blobs = cv2.convertScaleAbs(blobs)

        # Apply THR
        blobs[blobs < THR] = 0
        
   
    # Utilizzo la funzione label per etichettare le regioni diverse da 0
    labeled_matrix, num_regions = label(blobs)
    label_map = labeled_matrix[::-1,:,0]
    
    # Find Clusters
    original_img = image.copy()
    img_cluster = original_img.copy()

    blobs_4_contours = cv2.cvtColor(blobs, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(blobs_4_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if draw_type == 'line':
        centers = []
        for i, contour in enumerate(contours):
            # Compute center
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                center_x = int(moments['m10'] / moments['m00'])
                center_y = int(moments['m01'] / moments['m00'])
                centers.append([center_x, center_y])
                cv2.circle(img_cluster, (center_x, center_y), radius, (0, 0, 255), -1)
                # Draw contour
                cv2.drawContours(img_cluster, [contour], -1, (0, 255, 0), thickness)

    elif draw_type == 'area':
        colors = generate_colors(len(contours))
        for i, contour in enumerate(contours):
            fill_color = colors[i]
            # Creazione di un'immagine con canale alfa per il riempimento
            filled_area_with_alpha = np.zeros((img_cluster.shape[0], img_cluster.shape[1], 4), dtype=np.uint8)

            # Riempimento dell'area del contorno con il colore e l'opacità desiderati
            cv2.fillPoly(filled_area_with_alpha, [contour], fill_color)

            # Combinazione dell'immagine con trasparenza con l'immagine cluster utilizzando l'operatore OR
            img_cluster = cv2.bitwise_or(img_cluster, filled_area_with_alpha)
    else:
        raise ValueError('Choose a correct draw_type between \'line\' and \'area\'')
    
    if show_clusters:
        show_img(img_cluster, title=f'Convolutional Clustering L={loops}\nkernel: type={kernel_type}  size={k_size}')
    
        
    # df_cc['label']
    L = len(df_scaled.index) 
    labels = []
    for i in range(L):
        x = int(df_scaled[f'{col1}'].iloc[i])
        y = int(df_scaled[f'{col2}'].iloc[i])
        l = label_map[pad_value + y-1, pad_value + x-1]
        labels.append(l) 
        
    df2 = df.copy()
    df2['label'] = labels
    
    if plot_scatter:
        plt.figure(figsize=(6,6))
        plt.scatter(df2[col1], df2[col2], c=df2['label'])
        plt.xlabel(col1), plt.ylabel(col2)
        plt.show()
    return df2