## Hair Product Recommendation System for Common Hair Diseases (HPRS)
An application of Modern Convolutional Neural Networks to Recommend Hair Products.

## Data
Alopecia, psoriasis, and seborrheic-dermatitis photos taken from DermNet (Ani's source and [Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)).

Healthy hair photos taken from [Patch1k](http://projects.i-ctm.eu/it/progetto/).

## To-do
- [x] ResNeXt implementation: 77% validation accuracy
- [x] GoogLeNet implementation: 100% validation accuracy
- [x] AdamNet implementation: 78% validation accuracy
- [ ] Decision tree for hair product recommendation
- [ ] Finish paper
  - https://www.overleaf.com/read/tqvqgkvzsyzn#d77e2c
- [ ] Create and finish poster
  - [ ] Create and link canva or figma
  - [ ] Submit for printing (By Wed 5.00 PM!!!)
    - https://cscit.cs.gsu.edu/sp/printers/poster
- [ ] Implement focal loss to fix over-represented healthy hair
- [ ] Clean ResNeXt code
  - [ ] Remove data loading code
- [ ] Transition ResNext model to inherit from `nn.Module`
  - "Your models should also subclass this class."
  - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- [ ] Find and add more data?
- [ ] Make logo
- [ ] Submit to arxiv?
  -  https://info.arxiv.org/help/submit/index.html
