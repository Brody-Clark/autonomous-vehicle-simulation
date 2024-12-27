from scipy.integrate import solve_ivp
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
import numpy as np
from math import floor
import matplotlib.pyplot as plt

def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class Car(ShowBase):
    def __init__(self, x_0 = 0, y_0 = 0, v_0 = 0, heading_0 = 0, 
                 yaw_rate_0 = 0, side_slip_0 = 0, a_0 = 0, steering_0 = 0, 
                 wheel_base = 2):
        ShowBase.__init__(self)
        
        # Set up the window properties
        self.setBackgroundColor(0, 0, 0) # RGB values for black
        props = WindowProperties()
        props.setTitle("Car Simulation")
        self.win.requestProperties(props)
      
        self.wheel_base = wheel_base
        self.initial_conditions = np.array([x_0, y_0, heading_0, v_0, yaw_rate_0, side_slip_0])
        
        # Initialize the PID controllers
        self.pid_vel = PIDController(kp=2.0, ki=0.01, kd=0.01)
        self.pid_y_position = PIDController(kp=1, ki=.001, kd=17)
        
        # Disable the default camera control
        self.disableMouse()

        # Set the camera position
        self.camera.setPos(10, -20, 10)
        self.camera.lookAt(0, 0, 0)
        
        self.parent = NodePath("parent_node")
        self.parent.reparentTo(self.render)
        
        # Create the car body
        self.car_body = self.loader.loadModel("models/misc/rgbCube")
        self.car_body.setScale(2, 1, 0.5)
        self.car_body.setPos(0, 0, 0.35)
        self.car_body.reparentTo(self.parent)

        # Create the car wheels
        self.wheels = []
        for x, y in [(-0.8, 0.5), (-0.8, -0.5), (0.8, 0.5), (0.8, -0.5)]:
            wheel = self.loader.loadModel("models/misc/rgbCube")
            wheel.setScale(0.2, 0.4, 0.4)
            wheel.setPos(x, y, 0.1)
            wheel.setHpr(90, 0, 0)
            wheel.reparentTo(self.parent)
            self.wheels.append(wheel)

        # Create the path line 
        self.path_line = LineSegs() 
        self.path_line.setColor(1, 0, 0, 1) # Red color 
        self.path_line.setThickness(2.0) 
        self.path_line.moveTo(-1, 0, 0) 
        self.path_line.drawTo(100, 0, 0) # Draw a straight line along the x-axis 
        self.path_node = self.path_line.create() 
        self.path_np = self.render.attachNewNode(self.path_node)
        
        self.draw_axes()
        
        # Create OnscreenText to display speed and variable names 
        self.pos_text_x = OnscreenText(text=f"x: {x_0}", pos=(1.3, 0.9), scale=0.055, fg=(1,1,0,1), align=TextNode.ARight) 
        self.pos_text_y = OnscreenText(text=f"y: {y_0}", pos=(1.3, 0.8), scale=0.055, fg=(1,1,0,1), align=TextNode.ARight) 
        self.heading_angle_text = OnscreenText(text=f"heading angle (deg): {heading_0}", pos=(1.3, 0.7), scale=0.055, fg=(1,1,0,1), align=TextNode.ARight)
        self.steering_angle_text = OnscreenText(text=f"steering angle (deg): {steering_0}", pos=(1.3, 0.6), scale=0.055, fg=(1,1,0,1), align=TextNode.ARight)
        self.velocity_text = OnscreenText(text=f"velocity: {v_0}", pos=(1.3, 0.5), scale=0.055, fg=(1,1,0,1), align=TextNode.ARight)
        self.time_text = OnscreenText(text=f"time: {0}", pos=(1.3, 0.4), scale=0.055, fg=(1,1,0,1), align=TextNode.ARight)
        
    
    # Solve the ODE and visualize results
    def run(self, timespan = 10, dt = 0.1, desired_velocity=10.0, desired_y_pos=0.0, show_plot = False):
        self.dt = dt
        self.desired_velocity = desired_velocity  
        self.desired_y_position = desired_y_pos
        
        t_span = (0, timespan)
        
        solution = solve_ivp(self.ode_wrapper, t_span, self.initial_conditions, method='RK45',
                             t_eval=np.linspace(0, timespan, floor(timespan/dt)))
        self.t = solution.t
        self.states = solution.y
        self.cur_step = 0
                
        self.taskMgr.add(self.simulate_task, "SimulateTask")
        
        if show_plot:
            self.plot_results()
            
        ShowBase.run(self)

    # Plot the state variables
    def plot_results(self):
        plt.figure()
        plt.plot(self.t, self.states[0], label='x (position)')
        plt.plot(self.t, self.states[1], label='y (position)')
        plt.plot(self.t, self.states[2], label='theta (orientation)')
        plt.plot(self.t, self.states[3], label='v (velocity)')
        plt.xlabel('Time (s)')
        plt.ylabel('State Variables')
        plt.legend()
        plt.title('State Variables Over Time')
        plt.show()
        
    # Equations of motion for the car
    def state_space_equations(self, t, x, u):
        v = x[3]  # Velocity
        theta = x[2]  # Orientation angle
        delta = u[1]  # Steering angle
        
        # State space equations
        dxdt = np.zeros_like(x)
        dxdt[0] = v * np.cos(theta) # Update x_velocity 
        dxdt[1] = v * np.sin(theta)  # Update y_velocity 
        dxdt[2] = (v / self.wheel_base) * np.tan(delta)  # Update orientation angle rate of change
        dxdt[3] = u[0]  # Update velocity rate of change (acceleration input)
        dxdt[4] = 0  # Yaw acceleration rate
        return dxdt

    # Determines what inputs go into the next iteration
    def input_function(self, t, x):
        current_velocity = x[3]
        current_y_position = x[1] 
        
        # Velocity control
        velocity_error = self.desired_velocity - current_velocity
        acceleration = self.pid_vel.control(velocity_error, dt=0.1)
        
        # Y-axis position control
        y_position_error = self.desired_y_position - current_y_position
        steering_angle = self.pid_y_position.control(y_position_error, dt=0.1)
        
        # Clamp the steering angle 
        delta_max = np.radians(30) # Maximum steering angle in radians
        steering_angle = clamp(steering_angle,-delta_max, delta_max)
                
        return np.array([acceleration, steering_angle])

    # Define the ODE wrapper function
    def ode_wrapper(self, t, x):
        u = self.input_function(t, x) 

        return self.state_space_equations(t, x, u)

    def draw_axes(self): 
        axes = LineSegs() 
        axes.setThickness(2.0) 

        # X-axis (red) 
        axes.setColor(1, 0, 0, 1) 
        axes.moveTo(0, 0, 0) 
        axes.drawTo(1, 0, 0) 

        # Z-axis (blue) 
        axes.setColor(0, 0, 1, 1) 
        axes.moveTo(0, 0, 0) 
        axes.drawTo(0, 0, 1) 

        axes_node = axes.create() 
        axis_path = NodePath(axes_node) # Attach the coordinate axis to aspect2d 
        axis_path.reparentTo(aspect2d) # Position the coordinate axis in the upper left corner 
        axis_path.setPos(-1.2, 0, 0.8) 
        axis_path.setScale(0.1)
        
    # Create a small orange sphere at the position specified
    def createSphere(self, position):
        sphere = self.loader.loadModel("models/misc/sphere") 
        sphere.setScale(0.1, 0.1, 0.1) 
        sphere.setColor(1, 0.5, 0, 1) # Orange color 
        sphere.setPos(position) 
        sphere.reparentTo(self.render)
        
    def update_text(self, x, y, heading, velocity, steering, time):
        self.pos_text_x.setText(f"x: {x:.3f}") 
        self.pos_text_y.setText(f"y: {y:.3f}") 
        self.velocity_text.setText(f"velocity: {velocity:.3f}")
        self.heading_angle_text.setText(f"heading angle (deg): {heading:.3f}")
        self.steering_angle_text.setText(f"steering angle (deg): {steering:.3f}")
        self.time_text.setText(f"time: {time:.2f}")
        
    # Rotates wheels continuously around axel and updates heading angle
    #TODO: Allow for rotation speed to be adjusted
    def rotate_wheels(self, task, heading = 0):
        angle = task.time * 360 % 360
        for index,wheel in enumerate(self.wheels):
            if index in [2,3]:
                yaw = heading
            else:
                yaw = 0
            wheel.setHpr(90 + yaw, angle, 0)
            
    def update_camera(self):
        target_pos = self.parent.getPos()
        self.camera.setPos(target_pos + Point3(10, -20, 10))
        self.camera.lookAt(self.parent)
        
    def simulate_task(self, task):
        # Find the index of the closest time in the IVP solution set
        current_time = task.time
        index = np.searchsorted(self.t, current_time) 
        
        # If the index is within the range of the IVP solution, update the car's position and orientation 
        if index < len(self.t): 
            pos = Point3(self.states[0][index],self.states[1][index],0 )
            
            theta = self.states[2][index]
            vel = self.states[3][index]
            
            self.parent.setPos(pos) 
            self.parent.setHpr(Vec3(np.degrees(theta), 0, 0))
            steering_angle = np.arctan((theta/self.dt)*self.wheel_base/vel)
            self.rotate_wheels(task, np.degrees(steering_angle))
                
            self.update_text(x=pos.getX(), y=pos.getY(), heading=np.degrees(theta), velocity=vel, 
                             steering=np.degrees(steering_angle), time=task.time)
            
            self.createSphere(pos)
        
        self.update_camera()
        
        if task.time > self.t[-1]:
            return Task.done
        
        return Task.cont
        

# Define simulation parameters and run the simulation
app = Car(x_0=0, y_0=5, v_0=0, heading_0=(np.pi/4), a_0=0, steering_0=0, 
          yaw_rate_0=0, side_slip_0=0, wheel_base=2.5)
app.run(timespan=10, dt=0.1, desired_velocity=10.0, desired_y_pos=0.0, show_plot=False)

