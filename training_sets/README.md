
## About Training Data

This holds big datasets intended for AI model training. The dataset consists of two files: regs and spects. Regs are random synthesizer configurations, and spects are the frequency spectra obtained using each configuration. 

The data is generated using 'src/make_training_set.py', which takes several hours to generate a suitably huge training set. This is then processed by the pytorch model trainer in 'src/pytorch_train_NN.py', which creates a model at 'models/torch_model.pth'.  The model, if properly trained, should output a suitable synthesizer configuration when given an arbitrary frequency spectrum as input.


