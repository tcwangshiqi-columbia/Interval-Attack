# Interval Attack (adversarial ML)

Currently, most first-order adversarial attacks like PGD, CW or DAA maximize customized loss function to find adversarial examples. To maximize their chances, all of them repeat their attack algorithm from different random starting points. We propose a new first-order attack that leverages symbolic interval analysis to locate interesting regions that are more likely to contain adversarial examples and then use them as starting points to maximize the loss function. We found our interval attacks can provide significant improvement in attack success rate against popular defense models. More details about our interval attacks can be found at https://arxiv.org/abs/1811.02625.  

Note that symbolic interval analysis is a sound network output approximation method for given input ranges. The details of symbolic interval analysis can be found in our [Neurify](https://arxiv.org/abs/1809.08098) (NeurIPS 2018) and [ReluVal](https://www.cs.columbia.edu/~tcwangshiqi/docs/reluval.pdf) (Usenix Security 2018) papers.



## Downloading

```
git clone https://github.com/tcwangshiqi-columbia/Interval-Attack
```

## Running 

```
python attack2.py
```



## Citing MixTrain
```
@article{wang2018mixtrain,
  title={MixTrain: Scalable Training of Formally Robust Neural Networks},
  author={Wang, Shiqi and Chen, Yizheng and Abdou, Ahmed and Jana, Suman},
  journal={arXiv preprint arXiv:1811.02625},
  year={2018}
}
```


## Contributors

* [Shiqi Wang](https://sites.google.com/view/tcwangshiqi) - tcwangshiqi@cs.columbia.edu
* [Yizheng Chen](https://surrealyz.github.io/) - surrealyz@gmail.com
* Ahmed Abdou
* [Suman Jana](http://www.cs.columbia.edu/~suman/) - suman@cs.columbia.edu


## License
Copyright (C) 2018-2019 by its authors and contributors and their institutional affiliations under the terms of modified BSD license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
