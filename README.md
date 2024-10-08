# Penn Medicine BioBank Dataset (PMBB Data) 

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-witschey%40pennmedicine.upenn.edu-blue)](mailto:witschey@pennmedicine.upenn.edu)

`pmbb_data` is a library for loading and interacting with the PMBB dataset for building vision-language foundation models.

## Installation

Installation can be done using `pip`:

```
git clone https://github.com/PennMedicineVision/pmbb-data.git 
cd pmbb-data
python -m pip install .
```

Prior to usage, please run the [`setup.sh`](setup.sh) script provided in the repository:

```
bash setup.sh
```

Example usage can be found in in the [`example_usage.py`](example_usage.py) sample script.

## Changelog

### Version 0.0.3

- Significantly updated the API and moved study/series/report discovery logic to `setup.sh`.

### Version 0.0.2

- Added the ability to filter the dataset to only include certain imaging modalities and/or body parts examined. See example usage in the sample script.

### Version 0.0.1

- Initial commit.

## Contact

Questions and comments are welcome. Suggestions can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

[Walter Witschey](mailto:witschey@pennmedicine.upenn.edu)

## License

This repository is MIT licensed (see [LICENSE](LICENSE)).
