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

def GNN_seg(mode, epochs, K, compactness, n_segments, in_dir, out_dir, save, res, device, conv_hidden=1024, mlp_hidden=512, downsample_factor=4, r=3):
    """
    Segment entire dataset using either SLIC superpixels or pixel-based approach
    
    @param mode: 0 for SLIC, 1 for pixel-based
    @param epochs: Number of epochs for training
    @param K: Number of segments to search in each image
    @param compactness: Parameter for SLIC (only used in mode 0)
    @param n_segments: Number of superpixels for SLIC (only used in mode 0)
    @param in_dir: Directory for chosen dataset
    @param out_dir: Output directory to save results
    @param save: Whether to save results
    @param res: Resolution for input images
    @param device: Device to use ('cuda'/'cpu')
    @param downsample_factor: Downsampling factor for pixel mode (mode 1)
    @param r: Radius parameter for create_graph.wgraph (mode 1)
    """
    
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Create a tmp directory for temporary files if it doesn't exist
    os.makedirs('./tmp', exist_ok=True)
    
    if mode == 0:  # SLIC mode
        feats_dim = 5  # [R, G, B, y, x]
    else:  # Pixel mode with RAG
        feats_dim = 5  # [R, G, B, y, x] - même structure que SLIC
    
    from gnn_pool import GNNpool 
    
    model = GNNpool(feats_dim, conv_hidden, mlp_hidden, K, device).to(device)
    torch.save(model.state_dict(), 'model.pt')
    model.train()

    # Iterate over files
    for filename in tqdm(os.listdir(in_dir)):
        # If not image, skip
        if not filename.endswith((".jpg", ".png", ".jpeg")):
            continue
        # if file already processed
        if os.path.exists(os.path.join(out_dir, filename.split('.')[0] + '.txt')):
            continue
        if os.path.exists(os.path.join(out_dir, filename)):
            continue

        # Data loading
        _, image_orig = util.load_data_img(os.path.join(in_dir, filename), res)
        # Resize image
        image = cv2.resize(image_orig, res, interpolation=cv2.INTER_AREA)
        
        if mode == 0:  # SLIC mode
            # Process using SLIC
            labels = segmentation.slic(image, compactness=compactness, n_segments=n_segments, sigma=1, start_label=1)
            slic_boundaries = segmentation.mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), labels)
            
            # Create region adjacency graph
            rag = graph.rag_mean_color(image, labels, mode='similarity')
            A_nx = nx.adjacency_matrix(rag)
            
            # Create PyTorch tensors
            indices = torch.LongTensor(np.array(A_nx.nonzero()))
            values = torch.FloatTensor(A_nx[A_nx.nonzero()])
            
            # Create PyTorch sparse tensor for the model
            W_sparse = torch.sparse_coo_tensor(
                indices,
                values,
                size=torch.Size(A_nx.shape)
            ).to(device)
            
            # Convert to dense for the loss function
            W_dense = W_sparse.to_dense()
            
            # Extract node features
            node_features = []
            for region in rag.nodes():
                # Skip background region (label 0)
                if region == 0:
                    continue
                
                # Get region mask
                mask = labels == region
                
                # Calculate mean color
                mean_color = np.mean(image[mask], axis=0)
                
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
            
        else:  # Pixel mode using RAG directly on pixel grid
            # Downsample the image for the pixel mode to make computation tractable
            h, w = image.shape[:2]
            small_h, small_w = h // downsample_factor, w // downsample_factor
            small_img = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
            
            # Create pixel grid labels (each pixel is its own region)
            pixel_labels = np.arange(small_h * small_w).reshape(small_h, small_w)
            
            # Apply RAG directly on the pixel grid
            # This treats each pixel as a node in the graph
            rag = graph.rag_mean_color(small_img, pixel_labels, mode='similarity')
            
            # Create adjacency matrix from RAG
            A_nx = nx.adjacency_matrix(rag)
            
            # Convert to PyTorch tensors
            indices = torch.LongTensor(np.array(A_nx.nonzero()))
            values = torch.FloatTensor(A_nx[A_nx.nonzero()])
            
            # Create sparse tensor
            W_sparse = torch.sparse_coo_tensor(
                indices,
                values,
                size=torch.Size(A_nx.shape)
            ).to(device)
            
            # Convert to dense for loss function (careful with memory on large images)
            W_dense = W_sparse.to_dense()
            
            # Extract node features for each pixel
            node_features = []
            for i in range(small_h):
                for j in range(small_w):
                    # Use RGB values and normalized coordinates as features
                    pixel_features = small_img[i, j].astype(np.float32) / 255.0  # RGB normalized
                    y_coord = i / small_h  # Normalized y-coordinate
                    x_coord = j / small_w  # Normalized x-coordinate
                    
                    # Combine color and position features
                    if len(small_img.shape) == 3:  # Color image
                        features = np.concatenate([pixel_features, [y_coord, x_coord]])
                    else:  # Grayscale image
                        features = np.array([pixel_features, y_coord, x_coord])
                    
                    node_features.append(features)
            
            # Convert to tensor
            node_features = torch.FloatTensor(node_features).to(device)
            
            # Create PyTorch Geometric data
            data = Data(
                x=node_features,
                edge_index=indices,
                edge_attr=values
            ).to(device)
            
            # Create empty labels for visualization
            labels = pixel_labels
            slic_boundaries = None  # No SLIC boundaries for pixel mode
        
        # Train the model
        model.load_state_dict(torch.load('./model.pt', map_location=torch.device(device)))
        opt = optim.AdamW(model.parameters(), lr=0.001)
        losses = []
        
        # Create progress bar
        progress_bar = tqdm(range(epochs), desc="Training")
        display_freq = max(1, epochs // 10)
        
        for epoch_ in progress_bar:
            opt.zero_grad()
            A, S = model(data, W_dense)
            loss = model.loss(A, S)
            loss.backward()
            opt.step()
            loss_value = loss.item()
            losses.append(loss_value)
            
            # Update progress bar
            if epoch_ % display_freq == 0 or epoch_ == epochs - 1:
                progress_bar.set_postfix({
                    'loss': f"{loss_value:.4f}",
                    'min_loss': f"{min(losses):.4f}"
                })
        
        # Get final segmentation
        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)
        
        # Convert to pixel-wise segmentation
        if mode == 0:  # SLIC mode
            pixel_labels = np.zeros_like(labels)
            for i, label in enumerate(S):
                pixel_labels[labels == i+1] = label.item()
            
            # Final mask
            mask0 = pixel_labels.astype(float)
            
        else:  # Pixel mode
            # Reshape to 2D grid
            seg_map = S.numpy().reshape(small_h, small_w)
            
            # Resize back to original size
            mask0 = cv2.resize(seg_map.astype(float), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Apply segmentation overlay
        fused_image = util.apply_seg_map(image, mask0, 0.7)
        
        # Save/display results
        base_filename = os.path.splitext(filename)[0]
        method_name = "slic" if mode == 0 else "pixel"
        result_filename = os.path.join(out_dir, f"{base_filename}_segmentation_{method_name}.png")
        
        try:
            if mode == 0:  # SLIC mode - 4 subplots
                fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
                
                axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(slic_boundaries)
                axes[1].set_title("SLIC Boundaries")
                axes[1].axis('off')
                
                axes[2].imshow(mask0, cmap='tab20')
                axes[2].set_title('Segmentation Mask')
                axes[2].axis('off')
                
                axes[3].imshow(cv2.cvtColor(fused_image, cv2.COLOR_BGR2RGB))
                axes[3].set_title('Segmentation Overlay')
                axes[3].axis('off')
                
            else:  # Pixel mode - 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
                
                axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Show downsampled segmentation grid before final result
                pixel_grid = np.zeros((small_h, small_w, 3), dtype=np.uint8)
                for i in range(small_h):
                    for j in range(small_w):
                        color = np.random.randint(0, 256, 3)  # Random color for visualization
                        pixel_grid[i, j] = color
                
                axes[1].imshow(mask0, cmap='tab20')
                axes[1].set_title('Segmentation Mask')
                axes[1].axis('off')
                
                axes[2].imshow(cv2.cvtColor(fused_image, cv2.COLOR_BGR2RGB))
                axes[2].set_title('Segmentation Overlay')
                axes[2].axis('off')
            
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
    # mode == 0: SLIC superpixel segmentation
    # mode == 1: Pixel-based segmentation
    mode = 0  # Default to SLIC
    
    # GNN parameters
    epochs = 500
    K = 5  # Number of clusters
    
    # Image parameters
    res = (256, 256)
    
    # SLIC parameters (used if mode == 0)
    compactness = 40
    n_segments = 400
    
    # Pixel mode parameters (used if mode == 1)
    downsample_factor = 4  # Reduce image size by this factor
    r = 3  # Radius for neighborhood in pixel graph
    
    # I/O parameters
    in_dir = './images/single/'
    method_str = "slic" if mode == 0 else f"pixel_ds{downsample_factor}"
    out_dir = f'./results/K={K}_{method_str}'
    save = True
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Run segmentation with selected mode
    GNN_seg(mode, epochs, K, compactness, n_segments, in_dir, out_dir, save, res, device, 
            downsample_factor=downsample_factor, r=r)