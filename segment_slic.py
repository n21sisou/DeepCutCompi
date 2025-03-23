from torch_geometric.data import Data
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import util
import os
from scipy.sparse import csr_matrix
import cv2
from skimage import data, segmentation, color
from skimage import graph
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt 
def GNN_seg( epochs, K, compactness,n_segments, in_dir, out_dir, save, res,device):
    """
    Segment entire dataset; Get bounding box (k==2 only) or segmentation maps
    bounding boxes will be in the following format: class, confidence, left, top , right, bottom
    (class and confidence not in use for now, set as '1')
    @param epoch: Number of epochs for every step in image
    @param K: Number of segments to search in each image
    @param dir: Directory for chosen dataset
    @param out_dir: Output directory to save results
    @param device: Device to use ('cuda'/'cpu')
    """
    

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Create a tmp directory for temporary files if it doesn't exist
    os.makedirs('./tmp', exist_ok=True)
    #SLIC extractor superpixel [L, a, b, x, y] L, a, b: from the Lab color space (perceptually uniform) x, y: spatial coordinates
    feats_dim = 5
    
    from gnn_pool import GNNpool 
   

    model = GNNpool(feats_dim,256, 128, K, device).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()

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
        labels = segmentation.slic(image, compactness=compactness, n_segments=n_segments,sigma=1, start_label=1)
        slic_boundaries = segmentation.mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), labels) #plot slic boundaries

        rag = graph.rag_mean_color(image, labels, mode='similarity')

        # Dans GNN_seg, utilisez cette fonction:
        A_nx = nx.adjacency_matrix(rag)

        # F, W_sparse = prepare_features_and_graph(image,r=3)
        indices = torch.LongTensor(np.array(A_nx.nonzero()))
        values = torch.FloatTensor(A_nx[A_nx.nonzero()])

        # Create PyTorch sparse tensor for the model
        W_sparse = torch.sparse_coo_tensor(
            indices,
            values,
            size=torch.Size(A_nx.shape)
        ).to(device)

        # Convert to dense for the loss function (be careful with large graphs)
        W_dense = W_sparse.to_dense()
        # Données pour PyTorch Geometric

        node_features = []
        for region in rag.nodes():
            # Skip background region (label 0)
            if region == 0:
                continue
            
            # Get region mask
            mask = labels == region
            
            # Calculate mean color
            mean_color = np.mean(image[mask], axis=0)
            std_color = np.std(image[mask], axis=0)

            # Calculate center of mass (position)
            y_indices, x_indices = np.where(mask)
            center_y = np.mean(y_indices) / image.shape[0]  # Normalize
            center_x = np.mean(x_indices) / image.shape[1]  # Normalize
            
            # Combine features [R, G, B, y, x]
            features = np.concatenate([mean_color/255.0, [center_y, center_x]])
            node_features.append(features)

            # Convert to tensor
        node_features = torch.FloatTensor(node_features).to(device)

    # Create PyTorch Geometric data
        data = Data(
            x=node_features,
            edge_index=indices,
            edge_attr=values
        ).to(device)

        model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
        opt = optim.AdamW(model.parameters(), lr=0.001)
        # Entraînement
        losses = []

        # Créer barre de progression tqdm
        
        progress_bar = tqdm(range(epochs), desc="Training")
        display_freq = max(1, epochs // 10)

        for epoch_ in progress_bar:
            opt.zero_grad()
            # Utiliser le tenseur dense pour le modèle
            A, S = model(data, W_dense)
            loss = model.loss(A, S)
            loss.backward()
            opt.step()
            loss_value = loss.item()
            losses.append(loss_value)
            
            # Mettre à jour la description de la barre de progression
            # Afficher la perte seulement tous les display_freq époques (10% du total)
            if epoch_ % display_freq == 0 or epoch_ == epochs - 1:
                progress_bar.set_postfix({
                    'loss': f"{loss_value:.4f}",
                    'min_loss': f"{min(losses):.4f}"
                })


        # polled matrix (after softmax, before argmax)
        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)

        ##########################################################################################
        # Post-processing Connected Component/bilateral solver
        ##########################################################################################
        pixel_labels = np.zeros_like(labels)
        for i, label in enumerate(S):
            pixel_labels[labels == i+1] = label.item()

        # Post-processing
        mask0 = pixel_labels.astype(float)

        
        fused_image = util.apply_seg_map(image, mask0, 0.7)
        
        # Save the images using matplotlib plots
        base_filename = os.path.splitext(filename)[0]
        result_filename = os.path.join(out_dir, f"{base_filename}_segmentation.png")

        try:
            # Create a figure with 3 subplots
            fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
            
            # Plot original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[ 1].imshow(slic_boundaries)
            axes[ 1].set_title("SLIC Boundaries")
            axes[1].axis('off')
            # Plot segmentation mask
            axes[2].imshow(mask0, cmap='tab20')
            axes[2].set_title('Segmentation Mask')
            axes[2].axis('off')
            
            # Plot fused image
            axes[3].imshow(fused_image)
            axes[3].set_title('Segmentation Overlay')
            axes[3].axis('off')
            
            # Adjust layout and save
            plt.tight_layout()

            if save:
                plt.savefig(result_filename, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved segmentation results for {filename}")
            
            else:
                
                plt.show()
                
        except Exception as e:
            print(f"Error saving results for {filename}: {e}")

if __name__ == '__main__':
    ################################################################################
    # Mode
    ################################################################################
    # mode == 0 Single stage segmentation
    # mode == 1 Two stage segmentation for foreground
    # mode == 2 Two stage segmentation on background and foreground
    
    ################################################################################

    ################################################################################
    # GNN parameters
    ################################################################################
    # Numbers of epochs per stage [mode0,mode1,mode2]
    epochs=1000
    # Number of steps per image
    # Number of clusters
    K = 5
   
    res = (256, 256)
    #parameters for SLIC
    compactness=50
    n_segments=1000
    # Directory of image to segment
    in_dir = './images/single/'
    out_dir = f'./results/K={K}_slic({compactness,n_segments})'
    save = True
    
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    GNN_seg( epochs, K, compactness,n_segments ,in_dir, out_dir, save, res,device)
