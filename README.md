# rainfall_downscalling
This repository contains code related to rainfall downscaling.

# Initial Update
SpateGan1.0 is the original model Luca sent me before, including some input and target samples.
GCM2AORC is the model we have modified based on Luca's model. Input GCM daily data, and the target will be the AORC hour data. The data folder includes the training data for the month of 01/2005. Currenrtly, we are trying to do no downscaling by using this model.
AORC2AORC is trying to downscale from coarsen of AORC's 4-hours data to AORC's 1-hour data.
AORC2MRMS is trying to downscale from dataset of AORC's hourly data to MRMS's 10mins data.



## Acknowledgements

This repository builds upon the work presented in the following study:
**Global spatio-temporal downscaling of ERA5 precipitation through generative AI**  
[arXiv:2411.16098](https://arxiv.org/abs/2411.16098)
*Luca Glawion, Julius Polz, Harald Kunstmann, Benjamin Fersch, Christian Chwala*  
[Karlsruhe Institute of Technology, University of Augsburg]

We would like to thank the authors, in particular **Luca Glawion**, for providing the original model and codebase that served as a foundation for this work.  
If you use this repository, please consider citing their work.

```bibtex
@misc{glawion2024globalspatiotemporaldownscalingera5,
  title={Global spatio-temporal downscaling of ERA5 precipitation through generative AI}, 
  author={Luca Glawion and Julius Polz and Harald Kunstmann and Benjamin Fersch and Christian Chwala},
  year={2024},
  eprint={2411.16098},
  archivePrefix={arXiv},
  primaryClass={physics.ao-ph},
  url={https://arxiv.org/abs/2411.16098}
}