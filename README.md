# torchspleeter
Torchspleeter is a simplified interface for spleeter running completely on Torch, enabling cross platform functionality. It comes with a converted version of the 2stems model built in, and largely builds off the work of [Tuan Nguyen](https://github.com/tuan3w/spleeter-pytorch). 

---

## Installation

`pip install torchspleeter`

or download from this repository and 

`pip install .`

---

## Usage

Torchspleeter comes with both a Python API and a standard CLI for simplistic use. Once installed, you can use 
`torchspleeter -h` for instructions on how to use the CLI. 

For the Python API, an example is pretty simple:

```
from torchspleeter.command_interface import *

outputfiles=split_to_parts(input_audio_file,output_directory)

```

This will return two files, one the vocals and the second everything but the vocals, using the default 2stems model included with torchspleeter. The number of files scales to the number of models specified.


See [the testing example](./torchspleeter/tests/test_estimator.py) for in depth useage. 



---

## Reference
* [Original Spleeter](https://github.com/deezer/spleeter)
* [Tuan Nguyen](https://github.com/tuan3w/spleeter-pytorch)



---

## Contributing

If you'd like to contribute, please do! Please check the [CONTRIBUTING.md](./CONTRIBUTING.md) for details on the best way to get started. 

---

## License

**MIT**.
