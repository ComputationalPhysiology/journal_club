# Journal Club September 12 2019 - gotran

## gotran - General Ode TRANslator

* Developed by Johan Hake
* Maintained by Henrik Finsberg and Kristan Hustad


Source code: https://github.com/ComputationalPhysiology/gotran
Documentation: http://computationalphysiology.github.io/gotran/

### About

Gotran:

* provides a Python interface to declare arbitrary ODEs.

* provides an interface for generating CUDA/C/C++/Python/Matlab/Julia/OpenCL code for a number of functions including the right hand side and symbolic generation of a Jacobian.

* is intentionally lightweight, and could be interfaced by other Python libraries needing functionalities to abstract a general ODE.
* depends on NumPy, and on SymPy. See further instructions in can load models from external ODE desciption files such as CellML

### Install
```
pip install gotran
```
or install the development version
```
pip install git+https://github.com/ComputationalPhysiology/gotran.git
```

For some features you also need to install [swig](http://www.swig.org).


## Supported programming languages

* Python
    ```
    gotran2py tentusscher_2004_mcell.ode
    ```
* Matlab
    ```
    gotran2matlab tentusscher_2004_mcell.ode
    ```
* C
    ```
    gotran2c tentusscher_2004_mcell.ode
    ```
* C++
    ```
    gotran2cpp tentusscher_2004_mcell.ode
    ```
* CUDA
    ```
    gotran2cuda tentusscher_2004_mcell.ode
    ```
* Julia
    ```
    gotran2julia tentusscher_2004_mcell.ode
    ```
* FEniCS
    ```
    gotran2dolfin tentusscher_2004_mcell.ode
    ```
* CBCBeat
    ```
    gotran2beat tentusscher_2004_mcell.ode
    ```
* OpenCL
    ```
    gotran2opencl tentusscher_2004_mcell.ode
    ```


## Latex
Write ODE in a nice Latex document

```
gotran2latex tentusscher_2004_mcell.ode
```

### Demo - writing your first ODE file

Say we want to solve the following ODE
![ode](ode.png)

Then one way is to use the python interface
```python
from gotran import ODE
u0 = 0.5
v0 = -1.0

ode = ODE("MyFirstModel")

eps_value = 1e-5

u = ode.add_state("u", u0)
v = ode.add_state("v", v0)

eps = ode.add_parameter("eps", eps_value)

ode.add_derivative(u, ode.t, - (1.0/eps) * v**3)
ode.add_derivative(v, ode.t, (1.0/eps) * u**3)

ode.finalize()
ode.save("my_first_ode")
```

which would output the following `.ode` file

```
# Saved Gotran model

states(u=ScalarParam(0.5),
       v=ScalarParam(-1.0))

parameters(eps=ScalarParam(1e-05))

expressions("my_first_ode")
du_dt = -1.0*(v*v*v)/eps
dv_dt = 1.0*(u*u*u)/eps
```

or you can simple write the `.ode`-file directly.

## Demo - single cell model

### Get cellmodel

You can either for example download the cell models from https://www.cellml.org


Get cell model
```shell
wget https://models.physiomeproject.org/workspace/tentusscher_noble_noble_panfilov_2004/@@rawfile/941ec8e54e46e6fe82765c17f1d47582169baac2/tentusscher_noble_noble_panfilov_2004_a.cellml
```

### Convert .cellml files to an .ode files

```shell
cellml2gotran tentusscher_noble_noble_panfilov_2004_a.cellml
```


### Convert .ode files to python files


```shell
gotran2py  tentusscher_2004_mcell.ode --output cell_model_python
```

### Solving the models

We can for example use the ode-solver in Scipy

* [Python script](single_cell.py)
* [Notebook](single_cell.ipynb)

Alternatively you  can use `gotranrun`.


## Integration with CBCBeat

### Convert .ode files to cbcbeat files

```shell
cellml2gotran tentusscher_noble_noble_panfilov_2004_a.cellml
gotran2py  tentusscher_2004_mcell.ode --output cell_model_python
gotran2beat  tentusscher_2004_mcell.ode --output cell_model_cbcbeat
```


## Potential enhancements

* `gotran2cellml`
    - Convert `.ode`-file to `.cellml` so that any file written in `.ode`-format can be converted to `.cellml`
* Incorporate optimizations from Kristian's master thesis
* Make [`goss`](https://bitbucket.org/johanhake/goss) work again.
* Write better documentation - more example ++
* Convert tests to `pytest`
* Fix bugs