# Lung Cancer detection

This project will use CNN to detect abnormal cells at lung level.

Link to dataset: https://github.com/tampapath/lung_colon_image_set

## Abreviations
    
* Lung adenocarcinoma -> lung_aca (Adenocarcinoma of the lung is the most common type of lung cancer, and like other forms of lung cancer, it is characterized by distinct cellular and molecular features)

* Lung squamous cell carcinoma -> lung_scc (Lung squamous cell carcinoma (SCC) is a type of non-small cell (NSCLC) cancer that occurs when abnormal lung cells multiply out of control and form a tumor)

* Lung benign tissue -> lung_n (Lung benign tissue is an abnormal growth of tissue that serves no purpose and is found not to be cancerous.)

## TODO
 
- [ ] increasing the dataset size
- [ ] optimize the learning rate (https://www.jeremyjordan.me/nn-learning-rate/)
- [x] randomizing the training data order
- [x] improve the network design
- [x] Change bianry_crossentropy to loss_mean_squared_logarithmic_error
- [ ] used GANs implementation, slight improvement
- [x] improving the model using VGG19 and a custom LRA implementation
