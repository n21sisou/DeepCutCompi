# DeepCutCompi

Sisouphanthong Nolan, Charles Anna

This repo showcases our experiments and modifications based on the DeepCut algorithm.

> Based on the paper:  
> **DeepCut: Unsupervised Segmentation Using Graph Neural Networks Clustering**  
> Aflalo, Bagon, Kashti, Eldar – ICCVW 2023  
> [Read the paper](https://openaccess.thecvf.com/content/ICCV2023W/UG2+/papers/Aflalo_DeepCut_Unsupervised_Segmentation_Using_Graph_Neural_Networks_Clustering_ICCVW_2023_paper.pdf)

We took the original codebase and ran additional experiments. 
https://sampl-weizmann.github.io/DeepCut/

---

## 🧪 Our Approach

We explored whether the **Graph Neural Network (GNN)** in DeepCut, trained with the **Normalized Cut (N-cut)** loss, could still learn meaningful image segmentations **without relying on transformer-based features**.

To do this:

- We removed the Vision Transformer (ViT) feature extractor used in the original DeepCut.
- Instead, we applied the **SLIC superpixel algorithm** to obtain image patches.
- These superpixels were used to construct a graph directly from the image.
- The GNN was then trained on this graph using the same N-cut loss as in the original method.

Our goal was to assess the standalone learning capacity of the GNN for unsupervised segmentation when given a purely spatial and color based, low-level representation.
--- 
## ▶️ Run the Code

Clone the repository:

```bash
git clone https://github.com/n21sisou/DeepCutCompi.git
cd DeepCutCompi

python segment_slic.py
```
### example google colab notebook
[<img alt="Open In Colab" height="50" src="https://colab.research.google.com/assets/colab-badge.svg" width="300"/>](https://colab.research.google.com/drive/1gDkU0gCYKW3elZShclzCFyuK68FigISm#scrollTo=FckI1ji0zaFT)



## 🖼️ Example Output

Here we can see an output of the code:
![Example Segmentation](results/K=3_slic%28%2850%2C%20400%29%29/mnist3_segmentation.png)

