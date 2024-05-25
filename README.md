# Quantitative analysis of the dexamethasone side effect on human-derived young and aged skeletal muscle by myotube and nuclei segmentation using deep learning

![pipeline](https://github.com/tdn02007/QA-skMCs-Seg/img/main.png)


We report an a new method for quantitative analysis of the dexamethasone side effect on human-derived young and aged skeletal muscle by simultaneous myotube and nuclei segmentation using deep learning combined with post-processing techniques. The deep learning model outputs myotube semantic segmentation, nuclei semantic segmentation, and nuclei center, and post-processing applies a watershed algorithm to accurately distinguish overlapped nuclei and identify myotube branches through skeletonization.

 ## Denpendencies
To run the training and inference scrips, several dependencies are required to be installed.
To facilitate the installation process, We have prepared a requirements.txt which contains all the required packages. The requirements can be accessed by running:

    pip install -r requirements.txt

 ## Data
The original data includes fluorescence images of myotubes and nuclei. Nuclei are expressed as inp1 and myotubes are expressed as inp2. There are three types of label data for learning deep learning: nuclei semantic segmentation (cls1), nuclei center (cls2), and myotube semantic segmentation (cls3).

 ## Run Code
 ### Preparing the training data

Data for model learning can be obtained through preprocessing. Preprocessing was performed using the ImageJ program, and the following file is loaded into ImageJ and then execute.

    preprocessing_nuclei.ijm

Myotube data was preprocessed using the following python code.

    preprocessing_myotube.py

By specifying the folder of the original data and the folder where it will be saved, data for learning the model is created.

### Training
When data is ready, we can start training deep learning network by running:

    python model/main.py --mode train

Parameters can be modified in main.py such as batch_size, learning rate, model path, etc.


### Inference
The following script can be used to predict the results of new data when the model has been trained:

    python model/main.py --mode test --batch_size 1

The results will be stored at ./results/.
Finally, post-process the image by running the following in order:

    python post-processing/image_merge.py
    python post-processing/skeletonize_img.py
    python post-processing/watershed.py
    python post-processing/image_concat.py
