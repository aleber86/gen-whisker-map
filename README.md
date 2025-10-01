# Generalized Whisker Map  
This code is part of my thesis work by the name of _Estudio del Mapa de la Separatriz Generalizado_ (Study of a Generalized Whisker Map), to get a 5 years Ba. degree in Astronomy. Developed in Python with OpenCL 1.1 and implemented for GPGPUs.  
  
## About The Code  
*What this code does?* It calculates the evolution of an ensamble of initial conditions within the chaoctic layer (or stochastic layer) to get the half-width, metric entropy (KS-entropy), mLCE (maximal Lyapunov Caracteristic Exponent), and porosity of the layer.  
  
*How this attributes are calculated?* In a general manner, the equations that describes the dynamical system are iterated and in every step this attributes are calculated for every element in an ensamble of initial conditions. Let's see it in more detail.  
  
First, we need a free parameter, $\eta$, of the _whisker map_ (refer to *Mathematical Backgroud* section) and the half-width of the layer as a previous step (we get this values by iterating the map without calculating any attribute beside half-width and $\eta$). With these values the *device* (GPGPU side) program searches for the minimum and maximum value of the $y$ coordinate for every element in the ensamble, and stores in a global memory space each one (independent for every element in the ensamble). On the other hand, we use a *raster* (grid) to convert vector space information $(x,t,y)$ in counting information $(a[i][j] = n, i=1,\dots,N \wedge j = 1,\dots,M,\  N,M,n\in \mathbb{N})$, by the means of divide the surface area of the layer using the raster scaled cells with the previous half-width calculation. For every point that falls into a cell, the $a_{i,j}$ variable will add 1 into its value.  This process repeates for the $n_{it}$ iterations of the maps. So, the OpenCL Kernel will have the flowchart:
 
```mermaid  
flowchart TD  
A([Initial Condition]) --> B[Map Iteration]  
 
B --> C[/Save Attribs: max width, min width, rasterisation/] 
C --i=i+1--> D{Iteration Step i<n}
D --> E[Yes]
E --> B
D --> F[No]
F --> G[/Output Global Buffer/]  
```  
When reaches the total iteration step it returns the control to *host* program. 
At host side, we use Python (NumPy) to get the maximum and minimum value of array type variables and other statistics, input and output to files. And the host side program has the flowchart:

```mermaid
flowchart TD
A(For loop *Chunk*)
A --Chunk iteration-->B
B[OpenCL Kernels]-->C[/I/I_r Calc., Save values to arrays/]
C --> D{Every Chunk Itered?}
D --> E[Yes]--> G[Statistic, KS-Entropy, half-width]
D --> F[No] --> A
G --> H[/Save to file/]
```
_In which devices can it run?_ In every OpenCL compliant device, but there's a caveat, it has intensive computation with a bulk of data, so it is expected to be run in a GPGPU or in the best of cases in a HPC device.  
  
  
## Mathematical Background
To study Arnold's diffusion (Arnold, 1964), Chirikov (1979) propose the map that describes de dynamical system,

$$
\begin{align*}
y' = y + \frac{1}{\lambda_{1}}\sin{t} - \frac{\upsilon}{\lambda_{1}}\cos{x},\\
t = t - \lambda_{1}\log{|y'|} + \tilde\eta,\quad  t\mod{2\pi},\\
x = x -\lambda_{2}\log{|y'| + \omega_{2}\tilde\eta,\quad  x\mod{2\pi},}
\end{align*}
$$

when $\omega_{2}>1$ and $\upsilon \ll 1$, if we multiply the first equation of the system by $\lambda_{1}$, we get $\xi =\lambda_{1}\cdot y$ and re-arrange the other equations to get,

$$
\begin{align*}
\xi' = \xi + \sin{t} - \upsilon\cos{x},\\\\
t = t - \lambda_{1}\log{|\xi'|} + \tilde\eta,\quad  t\mod{2\pi},\\\\
x = x -\lambda_{2}\log{|\xi'| + \omega_{2}\tilde\eta,\quad  x\mod{2\pi},}
\end{align*}
$$

where $\tilde\eta$ absorbs the constant values that emerges from the sustitution. By looking the first equation, if $\upsilon \to 0$, we will find the whisker map for the perturbed separatrix of the pendulum model.


## Requisits

In order to get this program working, you will need to install OpenCL ICD for your device (make shure you have an OpenCL device compliant). If that so, install NumPy and PyOpenCL packages in your Python 3 enviroment:

```
pip install numpy pyopencl
```

## How do I use this code?
For $\eta$ non-fixed, you need to execute the script 
```
python wm_eta_finder.py
```
then you will get an auxiliary file in *data* folder. Then, you need to copy this file to the main folder, change the name of this file to match 'aux_pre_cached.dat' and then execute,

```
python eta_wm_lambda_1.py
```
you will get another auxiliary file on *data* folder. With it, again, copy back to main folder and change the name to 'aux.dat'. Lastly, execute

```
python porosity.py
```

Inside every Python script there is some guide, and if you want to chage the name files, for now, you have to do it inside the scripts.

## TODO List
Nowadays we are working on improving the readability of the code so we are in the stage of **housekeeping**. Much of the enviroment variables are hardwire into the code.

- [x] Replace cos(t) and sin(t) for sincos(t, *ptr) function.
- [ ] Housekeeping.
- [ ] Explicit use of FMA functions.
- [ ] Fixed raster scale. 
- [ ] Improve memory access.
- [ ] Output files from plain text to binary.

## Acknowledgments
I would like to thank the director of this work, PhD Pablo M. Cincotta, for having guided me throughout the development process of this work, in addition to providing me with his FORTRAN code for calculating the chaos estimator, MEGNO, which I then parallelized using OpenCL. 

## References
Chirikov, B. V. (1979). A universal instability of many-dimensional oscillator systems. Physics Reports, 52(5):263–379.

Arnold, V. I. (1964). On the instability of dynamic systems with many degrees of freedom. Dokl. Akad, Nauk SSSR.

Cincotta, P. y Giordano, C. (2016). Theory and applications of the Mean Exponential Growth factor of Nearby Orbits (MEGNO) method. Lecture Notes in Physics, 915:93–128.

> Written with [StackEdit](https://stackedit.io/).
