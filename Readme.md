# PINN , Physics-informed Neural Networks
Here is the folder to record the notes and codes that I learned about PINN topics. 

## Basic Concepts 

Basic sturcutre of PINN 
* Neural network 
* Initial and boundary conditions 
* Loss function
* Others 

Also there are many different variations of PINN to improve and adapt to increase and accomandate the corresponding application scenarios. 

---
## Project
I am doing the research on Laser metal deposition. I also read there are many research that used the PINN to simulate the deposition process. Here are something I want to try:
1. Moving heat source simulation 
A moving heat source which means the moving laser source. 

### Current challenges 
- Training data 
How many data for the PINN model training

- Verification and Validation 
To validate my model, there are some studies show using the classic problems like Possion's Equations, Berg's Equations which's analytical solutions are solved to compare the results with PINN's. 

2. Deposition Path Planning 



---

## Ref Github Repository 
### PINN
- [physics-informed-surrogate-modeling](https://github.com/joon-stack/physics-informed-surrogate-modeling)
- [Transfer learning PINN](https://github.com/shi-tong/Transfer-learning-based-PINN/tree/main)
- [Heat-Transfer-in-Advanced-Manufacturing-using-PINN](https://github.com/doomsday4/Heat-Transfer-in-Advanced-Manufacturing-using-PINN)
- [cladplus](https://github.com/openlmd/cladplus)

### Path planning
- [Manufacturing: 3D Slicing and 2D Path Planning](https://www.intechopen.com/chapters/50453)
 講述綜觀3d列印中，由不同路徑產生策略，去分割物件幾何的方法
- [Additive Manufacturing Path planning](https://fab.cba.mit.edu/classes/865.21/topics/path_planning/additive.html)