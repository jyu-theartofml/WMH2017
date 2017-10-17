## Segmentation of white matter intensities using modified 3D U-net.

![alt text](mask%20overlay.png "Predicted mask contour")

White Matter Hyperintensities (aka., leukoaraiosis) is often linked to high risk of stroke and dementia in older patients[1]. While image 
segmentation is critical for diagnosis and evaluation of treatments, automated segmentation of medical images remains a challenging task. In 
this post, a convolutional neural network called U-net is evaluated on brain MRIs for segmentation of White Matter Intensities (WMH).
The dataset was obtained from the [WMH Segmentation Challenge](http://wmh.isi.uu.nl/), which's organized by UMC Utrecht, VU Amsterdam, and NUHS Singapore. The goal is 
to train the deep learning model to generate binary mask that corresponds to WMH region of the brain MRI. 

### Why U-net
U-net was introduced by Olaf Ronneberger and his team back in 2015 as a refined autoencoder method to target medical image segmentation [2].
It's worth pointing out that while U-net looks very similar to SegNet (commonly used for semantic segmentation), the important difference is U-net's concatenation
step, where high-resolution features in the contractive path (left side of the net)is combined with the more abstract representational features in the expansive path (right side).
This allows the network to learn both localized finer features as well as contextual information, making it a desirable tool for various types of medical images with
high dynamic range and high resolution.

For this project,a modified U-net was compiled to accomodate 3D image arrays and computation cost, and this particular model is primarily based on Cicek et al.'s
published work on volumetric segmentation[3]. Instead of using the Caffe framwork, this model is built and trained using Keras. Due to computational overhead,
no batch normalization was used.

![alt text](U-net.png "U-net")
<p align ='center'><b>Fig 1.</b> 4-layer U-net for volumetric segmentation.(<i>source:https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/</i>)</p>


### Data Processing

There was total of 60 patient sample (20 from each site), and different MRI parameters were applied at different hospitals to generate multiple images for each patient. The images of interest are the pre-processed files that corrected for bias field, and only T1 and Flair images are used. In the data_process.py script, it imports the image files and subsequently reformats the data into numpy arrays. The input to the U-net model is resized to samples of 128x128x16x2 tensor; and training mask files are also converted to binary image. The mask files in the training dataset are annotated manually by radiology experts and used to train the model. The dataset was split into 75% training set and 25% validation set.








<p> </p>







1. Wardlaw, J. M., Valdés Hernández, M. C., & Muñoz-Maniega, S. (2015). What are White Matter Hyperintensities Made of?: Relevance to Vascular Cognitive Impairment. Journal of the American Heart Association: Cardiovascular and Cerebrovascular Disease, 4(6), e001140. http://doi.org/10.1161/JAHA.114.0011402.
2. Ronneberger,O.,Fischer, F., Brox, T. (2015) **U-Net: Convolutional Networks for Biomedical Image Segmentation**. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015 
3. Çiçek, O.,Abdulkadir, A., Lienkamp, S., Brox, T., Ronnebergeer, O.  **3D U-Net: Learning Dense VolumetricSegmentation from Sparse Annotation**. Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9901: 424--432, Oct 2016
