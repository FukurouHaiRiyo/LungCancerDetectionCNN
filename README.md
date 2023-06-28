# Proiect licenta
Acest proiect se va folosi de retelele neuronale convolutionale pentru detectia celuleor anormale din corp.

Link-ul catre dataset: https://github.com/tampapath/lung_colon_image_set

## Prescurtari/abrevieri
    
* Lung adenocarcinoma -> lung_aca (Adenocarcinoma of the lung is the most common type of lung cancer, and like other forms of lung cancer, it is characterized by distinct cellular and molecular features)

* Lung squamous cell carcinoma -> lung_scc (Lung squamous cell carcinoma (SCC) is a type of non-small cell (NSCLC) cancer that occurs when abnormal lung cells multiply out of control and form a tumor)

* Lung benign tissue -> lung_n (Lung benign tissue is an abnormal growth of tissue that serves no purpose and is found not to be cancerous.)

## TODO
* Improve accuracy by: 
      i. increasing the dataset size(✔️)
      ii. optimize the learning rate (https://www.jeremyjordan.me/nn-learning-rate/)
      iii. randomizing the training data order(✔️)
      iv. improve the network design(✔️)
      v. Change bianry_crossentropy to loss_mean_squared_logarithmic_error(✔️)
      vi. used GANs implementation, slight improvement
      vii. improving the model using VGG19 and a custom LRA implementation(to be done in the future)

## Problems
1. binary_crossentropy da loss error foarte mare(>=2), iar val_accuracy mic(<=40%). Am incercat loss_mean_squared_logarithmic_error. 
