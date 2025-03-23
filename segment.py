from bilateral_solver import bilateral_solver_output
from features_extract import deep_features
from torch_geometric.data import Data
from extractor import ViTExtractor
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import util
import os
from create_graph import wgraph
from scipy.sparse import csr_matrix
import cv2


def GNN_seg(mode, cut, alpha, epoch, K, pretrained_weights, in_dir, out_dir, save, cc, bs, log_bin, res, facet, layer,
            stride, device):
    """
    Segment entire dataset; Get bounding box (k==2 only) or segmentation maps
    bounding boxes will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')
    @param cut: chosen clustering functional: NCut==1, CC==0
    @param epoch: Number of epochs for every step in image
    @param K: Number of segments to search in each image
    @param pretrained_weights: Weights of pretrained images
    @param dir: Directory for chosen dataset
    @param out_dir: Output directory to save results
    @param cc: If k==2 chose the biggest component, and discard the rest (only available for k==2)
    @param b_box: If true will output bounding box (for k==2 only), else segmentation map
    @param log_bin: Apply log binning to the descriptors (correspond to smother image)
    @param device: Device to use ('cuda'/'cpu')
    """
    ##########################################################################################
    # Dino model init
    ##########################################################################################
    extractor = ViTExtractor('dino_vits8', stride, model_dir=pretrained_weights, device=device)
    # VIT small feature dimension, with or without log bin
    if not log_bin:
        feats_dim = 3
    else:
        feats_dim = 6528

    # # if two stage make first stage foreground detection with k == 2
    # if mode == 1 or mode == 2:
    #     foreground_k = K
    #     K = 2

    ##########################################################################################
    # GNN model init
    ##########################################################################################
    # import cutting gnn model if cut == 0 NCut else CC
    if cut == 0: 
        from gnn_pool import GNNpool 
    else: 
        from gnn_pool_cc import GNNpool

    model = GNNpool(feats_dim, 64, 32, K, device).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()


    ###for now we ignore only perform one segmentation
    # if mode == 1 or mode == 2:
    #     model2 = GNNpool(feats_dim, 64, 32, foreground_k, device).to(device)
    #     torch.save(model2.state_dict(), 'model2.pt')
    #     model2.train()
    # if mode == 2:
    #     model3 = GNNpool(feats_dim, 64, 32, 2, device).to(device)
    #     torch.save(model3.state_dict(), 'model3.pt')
    #     model3.train()

    ##########################################################################################
    # Iterate over files in input directory and apply GNN segmentation
    ##########################################################################################
    for filename in tqdm(os.listdir(in_dir)):
        # If not image, skip
        if not filename.endswith((".jpg", ".png", ".jpeg")):
            continue
        # if file already processed
        if os.path.exists(os.path.join(out_dir, filename.split('.')[0] + '.txt')):
            continue
        if os.path.exists(os.path.join(out_dir, filename)):
            continue

        ##########################################################################################
        # Data loading
        ##########################################################################################
        # loading images
        # image_tensor, image = util.load_data_img(os.path.join(in_dir, filename), res)
        image_tensor, image_orig = util.load_data_img(os.path.join(in_dir, filename), res)
        # Redimensionner l'image pour le traitement
        image = cv2.resize(image_orig, res, interpolation=cv2.INTER_AREA)
        # Extract deep features, from the transformer and create an adj matrix
        # F = deep_features(image_tensor, extractor, layer, facet, bin=log_bin, device=device)
        # W = util.create_adj(F, cut, alpha)
        
        
        # # Data to pytorch_geometric format
        # node_feats, edge_index, edge_weight = util.load_data(W, image)
        # data = Data(node_feats, edge_index, edge_weight).to(device)
        def prepare_features_and_graph(image, sigI=0.1/np.sqrt(3), sigX=4, r=7):
            """
            Prépare les caractéristiques d'image et le graphe de poids W
            """
            # Convertir l'image en niveaux de gris si elle est en couleur
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2).astype(np.float32)
            else:
                gray = image.astype(np.float32)
            
            # Normaliser l'image
            gray = gray / 255.0
            
            # Obtenir la matrice W sparse avec la fonction wgraph
            W_sparse = wgraph(gray, sigI=sigI, sigX=sigX, r=r)
            
            # Créer des caractéristiques basées sur les pixels
            pixel_values = gray.flatten().reshape(-1, 1)
            
            h, w = gray.shape
            y_coords = np.repeat(np.arange(h).reshape(-1, 1), w, axis=0) / h
            x_coords = np.tile(np.arange(w).reshape(1, -1), (h, 1)).reshape(-1, 1) / w
            
            # Concaténer pour obtenir [valeur_pixel, y_coord, x_coord]
            F = np.hstack([pixel_values, y_coords, x_coords])
            
            return F, W_sparse
        # Dans GNN_seg, utilisez cette fonction:
        F, W_sparse = prepare_features_and_graph(image,r=3)

# Convertir en tenseur PyTorch Geometric
        W_sparse_coo = W_sparse.tocoo()
        indices = torch.LongTensor(np.vstack([W_sparse_coo.row, W_sparse_coo.col]))
        values = torch.FloatTensor(W_sparse_coo.data)

        # Créer le tenseur sparse pour PyTorch
        W_tensor_sparse = torch.sparse_coo_tensor(
            indices,
            values,
            size=torch.Size(W_sparse.shape)
        ).to(device)

        # Pour le GNN pool, il faut le convertir en dense pour la fonction de perte
        # Attention: cela peut être coûteux en mémoire pour de grandes images
        W_tensor_dense = W_tensor_sparse.to_dense()

        # Données pour PyTorch Geometric
        node_feats = torch.FloatTensor(F)
        data = Data(x=node_feats, 
                    edge_index=indices,
                    edge_attr=values).to(device)
        model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
        opt = optim.AdamW(model.parameters(), lr=0.001)
        # Entraînement
        for _ in range(epoch[0]):
            opt.zero_grad()
            # Utiliser le tenseur dense pour le modèle
            A, S = model(data, W_tensor_dense)
            loss = model.loss(A, S)
            loss.backward()
            opt.step()

        # polled matrix (after softmax, before argmax)
        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)

        ##########################################################################################
        # Post-processing Connected Component/bilateral solver
        ##########################################################################################
        from util import graph_to_mask_pixel

        mask0, S = graph_to_mask_pixel(S, cc, image.shape)

        # apply bilateral solver
        if bs:
            mask0 = bilateral_solver_output(image, mask0)[1]

        if mode == 0:
            util.save_or_show([image, mask0, util.apply_seg_map(image, mask0, 0.7)], filename, out_dir ,save)
            continue
        


if __name__ == '__main__':
    ################################################################################
    # Mode
    ################################################################################
    # mode == 0 Single stage segmentation
    # mode == 1 Two stage segmentation for foreground
    # mode == 2 Two stage segmentation on background and foreground
    mode = 0
    ################################################################################
    # Clustering function
    ################################################################################
    # NCut == 0
    # CC == 1
    # alpha = k-sensetivity paremeter
    cut = 0
    alpha = 3
    ################################################################################
    # GNN parameters
    ################################################################################
    # Numbers of epochs per stage [mode0,mode1,mode2]
    epochs = [400, 100, 10]
    # Number of steps per image
    step = 1
    # Number of clusters
    K = 2
    ################################################################################
    # Processing parameters
    ################################################################################
    # Show only largest component in segmentation map (for k == 2)
    cc = False
    # apply bilateral solver
    bs = False
    # Apply log binning to extracted descriptors (correspond to smoother segmentation maps)
    log_bin = False
    ################################################################################
    # Descriptors extraction parameters
    ################################################################################
    # Directory to pretrained Dino
    pretrained_weights = './dino_deitsmall8_pretrain_full_checkpoint.pth'
    # Resolution for dino input, higher res != better performance as Dino was trained on (224,224) size images
    res = (32, 32)
    # stride for descriptor extraction
    stride = 8
    # facet fo descriptor extraction (key/query/value)
    facet = 'key'
    # layer to extract descriptors from
    layer = 11
    ################################################################################
    # Data parameters
    ################################################################################
    # Directory of image to segment
    in_dir = './images/single/'
    out_dir = './results/'
    save = False
    ################################################################################
    # Check for mistakes in given arguments
    assert not(K != 2 and cc), 'largest connected component only available for k == 2'

    # if CC set maximum number of clusters
    if cut == 1:
        K = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If Directory doesn't exist than download
    if not os.path.exists(pretrained_weights):
        url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
        util.download_url(url, pretrained_weights)

    GNN_seg(mode, cut, alpha, epochs, K, pretrained_weights, in_dir, out_dir, save, cc, bs, log_bin, res, facet, layer, stride,
            device)
