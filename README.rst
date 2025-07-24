=======
A3EM-AI
=======

This package provides tools for training and converting AI models to formats suitable for deployment on microcontrollers. It includes compatibility layers for PyTorch and TensorFlow Lite, as well as utilities for model quantization.

---------------
Getting Started
---------------

To get started with A3EM-AI, first clone the repository to your machine, ``cd`` into the root directory of the repo, then install the package locally using pip:

.. code-block:: bash

   pip install -e .

Any changes you make to the code will be immediately reflected in your local installation; however, if you add new files, you will need to run the installation command again.

Top-level tests can be run from the root directory of the repo using:

.. code-block:: bash

   python -m tests.[TestModule]

where ``[TestModule]`` is the name of the test module you want to run (e.g., ``python -m tests.TestYamnet``).

---------
TODO List
---------

* keras.model.quantize('int8') seems to work, can we train on it? QAT beforehand?
* Investigate ``ai-edge-quantizer`` for quantization
* Use each channel in 4-channel audio as a separate training example
* For DirectoryDataSet, use metadata labels if they exist instead of directory name
