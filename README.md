# Machine-Learning
                              
Using a different program, I ran 50,000 inflation simulations with various randomized inputs.
I stored the inputs with the maximum drag force and stored it all in a csv

This program is running that csv data through my deep learning neural network

Neural Network:
  - Architecture: 6 x 12 x 12 x 12 x 12 x 1
  - Activation Functions: ReLu (rectified linear unit).... might try leakyReLu and compare results

Progress:
  Completed:
    - data preparation: data has been scaled and split into batches
    - network creation: all weights/biases created relative to input array size
    - forward pass: data is run through network creating prediction value for all samples in batch
   Problems:
    - gradients: stuck on dL_dM. Doing a forward pass with a batch and then backpropogating one sample at a time is causing the matrices to not
                 line up like they need to
____________________________________________________________________________________________________
