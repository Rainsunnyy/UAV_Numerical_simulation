import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
from tf.transformations import quaternion_from_euler,euler_from_quaternion

def loaddata(path):
    data = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=2) 
    q = np.column_stack((data[:,0],data[:,1],data[:,2],data[:,3],data[:,8],data[:,9],data[:,10],data[:,5],data[:,6],data[:,7],data[:,4],data[:,20],data[:,21],data[:,22],data[:,23],data[:,11],data[:,12],data[:,13]))
    return q

def time_search(time, t_list):
    pointer = 0
    while(pointer < len(t_list)-1):
        if time < t_list[pointer]:
            return pointer
        else:
            pointer += 1

    return pointer

def rotation2euler(R) :
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]  
    
    x = np.arctan(r21 / r11)
    y = np.arcsin(-r31)
    z = np.arctan(r32 / r33)

    return np.array([x, y, z])

class DroneControlSim:
    def __init__(self):
        self.sim_time = 3
        self.sim_step = 0.002
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.pointer = 0 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 2.32e-3
        self.m = 1
        self.g = 9.8
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])

        self.k = 0

        self.path = "/home/rainsuny/racing_ws/src/time_optimal_trajectory/example/result.csv"
        self.plan_cmd = loaddata(self.path)

        self.forward_thrust_cmd = 0
        self.forward_bodyrate_cmd = np.array((3,))
        self.forward_orien_cmd = np.array((4,))
        self.forward_attitude_cmd = np.array((3,))

        self.position_des = [[0,0,0]]
        self.velocity_des = [[0,0,0]]
        self.attitude_des = [[0,0,0]]
        self.bodyrate_des = [[0,0,0]]


    def run(self):
        for self.pointer in range(self.drone_states.shape[0]-1):
            self.time[self.pointer] = self.pointer * self.sim_step
            # thrust_cmd = -9.8 * 0.5

            # rate_cmd = np.array([0,1,0])
            # self.velocity_cmd[self.pointer] = np.array([1,0,-1])
            # self.position_cmd[self.pointer] = np.array([1,0,0])
            # self.velocity_cmd[self.pointer] = self.position_controller(self.position_cmd[self.pointer])
            # thrust_cmd,phi_cmd,theta_cmd = self.velocity_controller(self.velocity_cmd[self.pointer])
            # attitude_cmd = np.array([phi_cmd, theta_cmd, 0])
            # self.attitude_cmd[self.pointer] = attitude_cmd
            # self.rate_cmd[self.pointer] = self.attitude_controller(self.attitude_cmd[self.pointer])
            # M_cmd = self.rate_controller(self.rate_cmd[self.pointer])
            # dx = self.drone_dynamics(thrust_cmd, M_cmd)
            # self.drone_states[self.pointer + 1] = self.drone_states[self.pointer] + self.sim_step * dx

            self.set_forwardcontrol(self.time[self.pointer])
            # self.bodyrate_cmd.append[self.forward_bodyrate_cmd] 
            # print(self.bodyrate_cmd.shape)
            thrust_cmd,M_cmd = self.feedback_control()
            # print(thrust_cmd)
            dx = self.drone_dynamics(thrust_cmd, M_cmd)
            # print(dx)
            self.drone_states[self.pointer + 1] = self.drone_states[self.pointer] + self.sim_step * dx



            
        self.time[-1] = self.sim_time
        

    def drone_dynamics(self,T,M):
        # Input:
        # T: float Thrust
        # M: np.array (3,)  Moments in three axes
        # Output: np.array (12,) the derivative (dx) of the drone 
        
        x = self.drone_states[self.pointer,0]
        y = self.drone_states[self.pointer,1]
        z = self.drone_states[self.pointer,2]
        vx = self.drone_states[self.pointer,3]
        vy = self.drone_states[self.pointer,4]
        vz = self.drone_states[self.pointer,5]
        phi = self.drone_states[self.pointer,6]
        theta = self.drone_states[self.pointer,7]
        psi = self.drone_states[self.pointer,8]
        p = self.drone_states[self.pointer,9]
        q = self.drone_states[self.pointer,10]
        r = self.drone_states[self.pointer,11]

        R_d_angle = np.array([[1,tan(theta)*sin(phi),tan(theta)*cos(phi)],\
                             [0,cos(phi),-sin(phi)],\
                             [0,sin(phi)/cos(theta),cos(phi)/cos(theta)]])


        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        d_position = np.array([vx,vy,vz])
        d_velocity = np.array([.0,.0,-self.g]) + R_E_B.transpose()@np.array([.0,.0,T])/self.m
        d_angle = R_d_angle@np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position,d_velocity,d_angle,d_q))

        return dx 

    def set_forwardcontrol(self,t):
        self.k = time_search(t,self.plan_cmd[:,0])
        # print(self.k)
        if self.k < self.plan_cmd.shape[0] - 1:
            self.forward_control(self.plan_cmd[self.k,:])
            # self.planning = self.plan_att[self.k,1:6]
        else:
            self.forward_control(self.plan_cmd[self.plan_cmd.shape[0] - 2,:])
        self.position_des.append(self.forward_position_cmd)
        self.velocity_des.append(self.forward_velocity_cmd)
        self.attitude_des.append(self.forward_attitude_cmd)
        self.bodyrate_des.append(self.forward_bodyrate_cmd)
        
        # print(len(self.bodyrate_cmd))


    def forward_control(self,data):
        self.forward_position_cmd = np.array([data[1],data[2],data[3]])
        self.forward_velocity_cmd = np.array([data[4],data[5],data[6]])
        self.forward_thrust_cmd = data[11] + data[12] + data[13] + data[14]
        # print(self.forward_thrust_cmd)
        self.forward_orien_cmd = np.array([data[7],data[8],data[9],data[10]])
        # print(self.forward_orien_cmd)
        self.forward_bodyrate_cmd = np.array([data[15],data[16],data[17]])
        self.forward_attitude_cmd = np.array(euler_from_quaternion([self.forward_orien_cmd[0],self.forward_orien_cmd[1],self.forward_orien_cmd[2],self.forward_orien_cmd[3]]))
        # print(self.forward_bodyrate_cmd)

    def feedback_control(self):
        k_p = 2
        k_v = 3
        K_pos = np.array([[k_p,0,0],[0,k_p,0],[0,0,k_p]])
        K_vel = np.array([[k_v,0,0],[0,k_v,0],[0,0,k_v]])
        acc_g = np.array([0, 0, self.g])
        acc_T = np.array([0, 0, self.forward_thrust_cmd])
        # print(acc_T)

        phi = self.drone_states[self.pointer,6]
        theta = self.drone_states[self.pointer,7]
        psi = self.drone_states[self.pointer,8]

        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])

        acc_fb = K_pos @ (self.forward_position_cmd - self.drone_states[self.pointer, 0:3]).T + K_vel @ (self.forward_velocity_cmd - self.drone_states[self.pointer, 3:6]).T
        # print(acc_fb)
        acc_ff = R_E_B.transpose()@acc_T
        # print(acc_ff)
        acc_des = np.array(acc_ff + acc_fb - acc_g)
        # acc_des = acc_des.reshape(-1,1)
        # print(acc_des)

        z_b_des = np.array(acc_des.T / np.linalg.norm(acc_des.T))
        # print(z_b_des)
        y_c = np.array([-sin(psi),cos(psi),0])
        x_b_des = np.cross(y_c.T,z_b_des) / np.linalg.norm(np.cross(y_c.T,z_b_des))
        y_b_des = np.cross(z_b_des,x_b_des)
        # print(y_b_des.shape)
        T_des = np.dot(acc_des, z_b_des)
        # print(T_des)

        R = (x_b_des, y_b_des, z_b_des)
        euler = rotation2euler(R)
        print(R)
        psi_cmd = euler[0]
        theta_cmd = euler[1]
        phi_cmd = euler[2]
        

        att_cmd = np.array([phi_cmd,theta_cmd,psi_cmd])
        # print(att_cmd)
        k_att = np.array([0.5, 0.5, 0.5])
        rate_cmd = k_att * (att_cmd - self.drone_states[self.pointer, 6:9])
        # rate_cmd = self.forward_bodyrate_cmd
        # T_des = self.forward_thrust_cmd

        k_rate = np.array([0.5, 0.5, 0.5])
        M_cmd = (k_rate * (rate_cmd - self.drone_states[self.pointer, 9:12]))
        # print(M_cmd)
        return T_des,M_cmd





    def rate_controller(self,cmd):
        kp = np.array([0.1, 0.1, 0.1])
        M_cmd = kp * (cmd - self.drone_states[self.pointer, 9:12])
        # self.drone_states[self.pointer, 9:12] = M
        # M = self.drone_states[self.pointer, 9:12] + kp * self.drone_states[self.pointer, 9:12]
        # p = self.drone_states[self.pointer, 9]
        # q = self.drone_states[self.pointer,10]
        # r = self.drone_states[self.pointer,11]
        # M[0] = p + kp * (rate_set[0] - p)
        # M[1] = q + kp * (rate_set[1] - q)
        # M[2] = r + kp * (rate_set[2] - r)
        # self.drone_states[self.pointer, 9] = M[0]
        # self.drone_states[self.pointer,10] = M[1]
        # self.drone_states[self.pointer,11] = M[2]
        # Input: cmd np.array (3,) rate commands
        # Output: M np.array (3,) moments
        return M_cmd

    def attitude_controller(self,cmd):
        kp = [0.5, 0.5, 0.5]
        rate_cmd = kp * (cmd - self.drone_states[self.pointer, 6:9])
        # self.drone_states[self.pointer, 6:9] = M 
        # Input: cmd np.array (3,) attitude commands
        # Output: M np.array (3,) rate commands
        return rate_cmd

    def velocity_controller(self,cmd):
        kp = [0.8, 0.8, 0.8]
        phi_cmd = kp[0] * (cmd[0] - self.drone_states[self.pointer, 3])
        theta_cmd = kp[1] * (cmd[1] - self.drone_states[self.pointer, 4])
        thrust_cmd = -self.m * self.g + kp[2] * (cmd[2] - self.drone_states[self.pointer, 5])
        
        # kp = [2.25, 2.25, 2.25]
        # attitude_cmd = kp * (cmd - self.drone_states[self.pointer, 6:9])
        # Input: cmd np.array (3,) velocity commands
        # Output: M np.array (2,) phi and theta commands and thrust cmd
        return thrust_cmd,phi_cmd,theta_cmd

    def position_controller(self,cmd):
        kp = [0.3, 0.3, -0.3]
        position_cmd = kp * (cmd - self.drone_states[self.pointer, 0:3])

        # Input: cmd np.array (3,) position commands
        # Output: M np.array (3,) velocity commands
        return position_cmd


    def plot_states(self):
        fig1, ax1 = plt.subplots(4,3)
        self.position_cmd[-1] = self.position_cmd[-2]
        ax1[0,0].plot(self.time,self.drone_states[:,0],label='real')
        ax1[0,0].plot(self.time,np.array(self.position_des)[:,0],label='cmd')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.drone_states[:,1])
        ax1[0,1].plot(self.time,np.array(self.position_des)[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.drone_states[:,2])
        ax1[0,2].plot(self.time,np.array(self.position_des)[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3])
        ax1[1,0].plot(self.time,np.array(self.velocity_des)[:,0])
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.drone_states[:,4])
        ax1[1,1].plot(self.time,np.array(self.velocity_des)[:,1])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.drone_states[:,5])
        ax1[1,2].plot(self.time,np.array(self.velocity_des)[:,2])
        ax1[1,2].set_ylabel('vz[m/s]')

        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.drone_states[:,6])
        ax1[2,0].plot(self.time,np.array(self.attitude_des)[:,0])
        ax1[2,0].set_ylabel('phi[rad]')
        ax1[2,1].plot(self.time,self.drone_states[:,7])
        ax1[2,1].plot(self.time,np.array(self.attitude_des)[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        ax1[2,2].plot(self.time,self.drone_states[:,8])
        ax1[2,2].plot(self.time,np.array(self.attitude_des)[:,2])
        ax1[2,2].set_ylabel('psi[rad]')

        # self.rate_cmd[-1] = self.rate_cmd[-2]
        # ax1[3,0].plot(self.time,self.drone_states[:,9])
        # ax1[3,0].plot(self.time,self.rate_cmd[:,0])
        # ax1[3,0].set_ylabel('p[rad/s]')
        # ax1[3,1].plot(self.time,self.drone_states[:,10])
        # ax1[3,1].plot(self.time,self.rate_cmd[:,1])
        # ax1[3,1].set_ylabel('q[rad/s]')
        # ax1[3,2].plot(self.time,self.drone_states[:,11])
        # ax1[3,2].plot(self.time,self.rate_cmd[:,2])
        # ax1[3,2].set_ylabel('r[rad/s]')

        # self.forward_bodyrate_cmd[-1] = self.forward_bodyrate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9])
        ax1[3,0].plot(self.time,np.array(self.bodyrate_des)[:,0])
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.drone_states[:,10])
        ax1[3,1].plot(self.time,np.array(self.bodyrate_des)[:,1])
        ax1[3,1].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.drone_states[:,11])
        ax1[3,2].plot(self.time,np.array(self.bodyrate_des)[:,2])
        ax1[3,2].set_ylabel('r[rad/s]')

if __name__ == "__main__":
    drone = DroneControlSim()
    drone.run()
    drone.plot_states()
    plt.show()
    