# Simulation codes for the article: Optimization algorithms inspired by the geometry of dissipative systems
Alessandro Bravetti, Maria L. Daza-Torres, Hugo Flores-Arguedas and Michael Betancourt. https://arxiv.org/abs/1912.02928

The available files are:

- integrators.py
    Implements all the algorithms tested: classical momentum (CM),
    Nesterov’s  accelerated  gradient (NAG), relativistic gradient descent (RGD)
    and contact relativistic gradient descent (CRGD).
    (see `requirements.txt` for a compatible  set of dependencies).

- examples.py
    Contains all the examples presented in the paper.

The following files, contain the code to generate the figures used in the paper,
- plot_ex1.py
- plot_ex2.py
- plot_ex3.py
- plot_ex4.py
Here, we use the tuned parameters found in the params folder.


The following files contain the tuning process for the params used in the paper,

- Tuning_ex1.py
- Tuning_ex2.py
- Tuning_ex3.py
- Tuning_ex4.py

- params
Contain the tuned parameters used to generate the figures of the article. These files were obtained
for running:

- Tuning_ex1.py
- Tuning_ex2.py
- Tuning_ex3.py
- Tuning_ex4.py
