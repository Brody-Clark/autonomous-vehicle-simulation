# autonomous-vehicle-simulation

This project is a simulation written in Python utilitzing the `RK45` numerical integration algorithm to show potential control of an autonomous vehicle. This simulation implements a PID controller to achieve a desired velocity and y-axis offset from given initial conditions. Results are displayed in 3D using panda3d with 2D plotting also available.

![demo screenshot](./resources/vehicle-simulation-demo.png)

## Running

With Python installed, you can run `pip install numpy panda3d scipy matplotlib` to install the necessary libraries. Then you can run `python` in a command terminal and enter the path to the `Simulations.py` file to run the default simulation.

## Modeling and Simulation

This simulation uses a simplified model of a 2D vehicle. As such, side slip, suspensions, and other variables are not considered. The simplified equations of motion used in the state-space representation are as follows:

$$\dot{x} = v*cos(\theta)$$

$$\dot{y} = v*sin(\theta)$$

$$\dot{\theta} = v/L*tan(\delta)$$

$$\dot{v} = \alpha$$

$$\dot{\gamma} = 0$$

With input vector:

$$ u = [\alpha,  \delta] $$

Where v is the speed in the car's forward direction, $\theta$ is the heading angle with respect to the x-axis, $\delta$ is a steering angle, and $\alpha$ is acceleration.
