## Hair Product Recommendation System for Common Hair Diseases (HPRS)
An application of Modern Convolutional Neural Networks to Recommend Hair Products.

## Data
Alopecia, psoriasis, and seborrheic-dermatitis photos taken from DermNet (Ani's source and [Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)) and DermNetNZ ([Github](https://github.com/Mrinmoy-Roy/Scalp-Hair-Diseases-Detection/tree/main)).

Healthy hair photos taken from [Patch 1k](http://projects.i-ctm.eu/it/progetto/figaro-1k).

## To-do
- [x] ResNeXt implementation: 77% validation accuracy
- [x] GoogLeNet implementation: 100% validation accuracy
- [x] AdamNet implementation: 78% validation accuracy
- [x] Find and add more data?
- [x] Decision tree for hair product recommendation
- [x] Finish paper
  - [x] Redo abstract, include models besides ResNeXt
  - [x] Finish model explanation (4)
  - [x] Finish (5)
  - [x] Discuss conditions in intro
  - [x] Discuss hyperparameters in methodologies
  - https://www.overleaf.com/read/tqvqgkvzsyzn#d77e2c
  - Linking github: https://academia.stackexchange.com/questions/20358/how-should-i-reference-my-github-repository-with-materials-for-my-paper
- [x] Create and finish poster
  - [x] https://docs.google.com/presentation/d/1hA5rWyWDMbacixMjWD9grUZqYVMUG_hAqZQ-85CnVOQ/edit?usp=sharing
  - [x] Submit for printing (By Wed 5.00 PM!!!)
  - [x] Poster status: https://cscit.cs.gsu.edu/support/tickets.php
    - jyoo30@student.gsu.edu
    - #00092
  - [x] Pickup poster from 25 PP, 7th floor
  - [x] Print finished poster segments
- [x] Implement focal loss to fix over-represented healthy hair
- [ ] Clean ResNeXt code
  - [ ] Remove data loading code
- [ ] Transition ResNext model to inherit from `nn.Module`
  - "Your models should also subclass this class."
  - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- [ ] Submit to arxiv?
  -  https://info.arxiv.org/help/submit/index.html
