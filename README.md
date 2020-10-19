# mlp
Source code related to the article "[Numerical simulations for full history recursive multilevel Picard approximations for systems of high-dimensional partial differential equations](https://arxiv.org/abs/2005.10206)" by Sebastian Becker, Ramon Braunwarth, Martin Hutzenthaler, Arnulf Jentzen, and Philippe von Wurstemberger

### Notes:
The provided source code uses the Eigen C++ Library (version 3.3.7) and the POSIX Threads API to allow for parallelism on modern multicore CPUs. The examples in the article were compiled with the C++ compiler of the GNU Compiler Collection (version 7.5.0) with optimization level 3 (`-O3`). The different examples can be selected at compile time by providing a preprocessor symbol using the -D option to activate the corresponding preprocessor macro. 
Possible choices for the preprocessor symbol are: `ALLEN_CAHN, SINE_GORDON, PDE_SYSTEM, SEMILINEAR_BS, BS_SYSTEM`

For example, the source code for the Allen-Cahn example was compiled using the command:
```
g++ -DALLEN_CAHN -O3 -o mlp mlp.cpp -lpthread
```

Note that if the Eigen headers are not available system-wide, the path has to be provided using the `-I` option in the command above.
