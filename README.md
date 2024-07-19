### The official coding of NeSM
### A Segmentation Method suitable for NeRF can perform ultrasound image segmentation.
[NeSM: A NeRF-Based 3D Segmentation Methodfor Ultrasound Images](https://ieeexplore.ieee.org/document/10547922)
Yong-Jun Wen; Jenhui Chen
in Proc. 2024 10th ICASI (Oral Presentation)

# Setup
```javascript
conda env create -f environment.yml
conda activate NeSM
```
# Running code
Training:

`python segment_ultra_nerf_v03.py --config [CHANGE TO ANY conf_ file].txt --expname test_generated --n_iters [>500000] --loss double_ssim --i_embed_gauss 0 --i_img 2000 --i_print 2000  --i_weights 2000 --ft_path /logs/[ANY MODEL NAME]`

Rendering:

`python render_demo_segment[ANY NAME].py --config [CHANGE TO ANY conf_ file].txt --ft_path /logs/[ANY MODEL NAME]`

Testing:

(not yet)

# Acknowledgments
Sincere thanks to wysocki et al. for opening the [Ultra-NeRF original code](https://github.com/magdalena-wysocki/ultra-nerf/tree/main) and part of the dataset.
We sincerely thank [Krönke et al.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0268550) for opening up part of the [clinical dataset](https://www.cs.cit.tum.de/camp/publications/segthy-dataset/).
