import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan,atan,asin,acos
from scipy.spatial.transform import Rotation as R
from dronecontrol_ff import DroneControl_feed_foward


class DroneControlSim:
    def __init__(self):
        self.sim_time = 6 
        self.sim_step = 0.002
        self.drone_states = np.zeros((int(self.sim_time/self.sim_step), 12))
        self.time= np.zeros((int(self.sim_time/self.sim_step),))
        self.rate_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.attitude_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.velocity_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        # self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3)) 
        self.position_cmd = np.zeros((int(self.sim_time/self.sim_step), 3))
        self.pointer = 0 
        self.drone_states[0,0:3]= np.array([0,0,-0.6]) 



        self.acc_ref_int_debug = np.zeros((int(self.sim_time/self.sim_step), 6)) 

        self.trajectory_ref = np.zeros((int(self.sim_time/self.sim_step), 6)) 

        self.I_xx = 2.32e-3
        self.I_yy = 2.32e-3
        self.I_zz = 2.32e-3
        self.m = 1
        self.g = 9.81
        self.I = np.array([[self.I_xx, .0,.0],[.0,self.I_yy,.0],[.0,.0,self.I_zz]])

        # self.drone_states[0,0:3] = [0,0,-5]


        self.position_des = [[0,0,0]]
        self.velocity_des = [[0,0,0]]
        self.attitude_ff = [[0,0,0]]
        self.attitude_fb = [[0,0,0]]
        self.bodyrate_ff = [[0,0,0]]
        self.bodyrate_fb = [[0,0,0]]
        self.acc_fb = [[0,0,0]]
        


    def run(self):
        for self.pointer in range(self.drone_states.shape[0]-1):
            self.time[self.pointer] = self.pointer * self.sim_step

            t = self.time[self.pointer]
            pos_ff,vel_ff,thrust_ff,att_ff,rate_ff,is_done = dronecontol_ff.set_forwardcontrol(t)

            att_cmd,thrust_cmd = self.feedback_control(pos_ff,vel_ff,att_ff,thrust_ff)

            rate_fb = self.attitude_controller(att_cmd)

            # rate_fb = np.zeros((3,))
            # thrust_cmd = thrust_ff


            # rate_ff = np.zeros((3,))
            rate_cmd = rate_ff + rate_fb

            M_cmd = self.rate_controller(rate_cmd)

            # M_cmd = self.rate_controller(rate_ff)

            dx = self.drone_dynamics(thrust_cmd, M_cmd)
            self.drone_states[self.pointer + 1] = self.drone_states[self.pointer] + self.sim_step * dx

            self.position_des.append(pos_ff)
            self.velocity_des.append(vel_ff)
            self.attitude_ff.append(att_ff)
            self.attitude_fb.append(att_cmd)
            self.bodyrate_ff.append(rate_ff)
            self.bodyrate_fb.append(rate_fb)



            
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
        d_velocity = np.array([.0,.0,self.g]) + R_E_B.transpose()@np.array([.0,.0,T])
        d_angle = R_d_angle@np.array([p,q,r])
        d_q = np.linalg.inv(self.I)@(M-np.cross(np.array([p,q,r]),self.I@np.array([p,q,r])))

        dx = np.concatenate((d_position,d_velocity,d_angle,d_q))

        return dx 

    def pos_vel_acc_model(self,acc):
        d_position = self.acc_ref_int_debug[self.pointer,3:6]
        d_velocity = acc
        dx = np.concatenate((d_position,d_velocity))
        return dx


    def feedback_control(self,pos_ff,vel_ff,att_ff,thrust_ff):
        k_p = 5 
        k_v = 1 
        K_pos = np.array([[k_p,0,0],[0,k_p,0],[0,0,k_p]])
        K_vel = np.array([[k_v,0,0],[0,k_v,0],[0,0,k_v]])
        acc_g = np.array([0, 0, self.g])

        current_pos = self.drone_states[self.pointer,0:3]
        current_vel = self.drone_states[self.pointer,3:6]

        # psi = self.drone_states[self.pointer,8]
        psi = att_ff[2]

        acc_fb = K_pos @ (pos_ff - current_pos) + K_vel @ (vel_ff - current_vel)

        self.acc_fb.append(acc_fb)

        # R_E_B = R.from_euler('zyx',[att_ff[2],att_ff[1],att_ff[0]])
        # R_B_E = R_E_B.inv() 

        phi = att_ff[0]
        theta = att_ff[1]
        psi = att_ff[2]

        R_E_B = np.array([[cos(theta)*cos(psi),cos(theta)*sin(psi),-sin(theta)],\
                          [sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi),sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi),sin(phi)*cos(theta)],\
                          [cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi),cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi),cos(phi)*cos(theta)]])
        acc_ref = acc_g + R_E_B.transpose() @ np.array([0.0,0.0,thrust_ff])


        dx = self.pos_vel_acc_model(acc_ref)
        self.acc_ref_int_debug[self.pointer+1] = self.acc_ref_int_debug[self.pointer]+self.sim_step*dx

        acc_des = acc_fb + acc_ref - acc_g

        z_b_des = np.array(-acc_des / np.linalg.norm(acc_des))
        y_c = np.array([-sin(psi),cos(psi),0])
        x_b_des = np.cross(y_c,z_b_des) / np.linalg.norm(np.cross(y_c,z_b_des))
        y_b_des = np.cross(z_b_des,x_b_des)
        T_des = np.dot(acc_des, z_b_des)

        # R_E_B = R.from_matrix(np.transpose(np.array([x_b_des,y_b_des,z_b_des])))
        # psi_cmd,theta_cmd,phi_cmd = R_E_B.as_euler('zyx')
        R_E_B = np.transpose(np.array([x_b_des,y_b_des,z_b_des]))
        psi_cmd = atan(R_E_B[1,0]/R_E_B[0,0])
        theta_cmd = asin(-R_E_B[2,0])
        phi_cmd = atan(R_E_B[2,1]/R_E_B[2,2])

        if self.pointer < 20:
            print(phi_cmd)

        phi_cmd = self.cmd_bound(phi_cmd,60.0/180*3.14,-60.0/180*3.14)
        theta_cmd = self.cmd_bound(theta_cmd,60.0/180*3.14,-60.0/180*3.14)

        att_cmd = np.array([phi_cmd,theta_cmd,psi_cmd])
        
        return att_cmd, T_des


    def cmd_bound(self,cmd,ub,lb):
        if cmd > ub:
            cmd = ub
        elif cmd < lb:
            cmd = lb
        return cmd

    def rate_controller(self,cmd):
        kp = np.array([0.1, 0.1, 0.1])
        M_cmd = kp * (cmd - self.drone_states[self.pointer, 9:12])

        return M_cmd

    def attitude_controller(self,cmd):
        kp = [5, 5, 5]
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
        # ax1[0,0].plot(self.time,np.array(self.acc_ref_int_debug)[:,0],label='int')
        ax1[0,0].set_ylabel('x[m]')
        ax1[0,1].plot(self.time,self.drone_states[:,1])
        ax1[0,1].plot(self.time,np.array(self.position_des)[:,1])
        #ax1[0,1].plot(self.time,np.array(self.acc_ref_int_debug)[:,1])
        ax1[0,1].set_ylabel('y[m]')
        ax1[0,2].plot(self.time,self.drone_states[:,2])
        ax1[0,2].plot(self.time,np.array(self.position_des)[:,2])
        # ax1[0,2].plot(self.time,np.array(self.acc_ref_int_debug)[:,2])
        ax1[0,2].set_ylabel('z[m]')
        ax1[0,0].legend()

        self.velocity_cmd[-1] = self.velocity_cmd[-2]
        ax1[1,0].plot(self.time,self.drone_states[:,3])
        ax1[1,0].plot(self.time,np.array(self.velocity_des)[:,0])
        # ax1[1,0].plot(self.time,np.array(self.acc_ref_int_debug)[:,3],label='int')
        ax1[1,0].set_ylabel('vx[m/s]')
        ax1[1,1].plot(self.time,self.drone_states[:,4])
        ax1[1,1].plot(self.time,np.array(self.velocity_des)[:,1])
        # ax1[1,1].plot(self.time,np.array(self.acc_ref_int_debug)[:,4])
        ax1[1,1].set_ylabel('vy[m/s]')
        ax1[1,2].plot(self.time,self.drone_states[:,5])
        ax1[1,2].plot(self.time,np.array(self.velocity_des)[:,2])
        # ax1[1,2].plot(self.time,np.array(self.acc_ref_int_debug)[:,5])
        ax1[1,2].set_ylabel('vz[m/s]')

        #ax1[2,0].plot(self.time,np.array(self.acc_fb[:,0]))
        #ax1[2,0].set_ylabel('acc[m/s^2]')
        #ax1[2,1].plot(self.time,self.acc_fb[:,0])
        #ax1[2,2].plot(self.time,self.acc_fb[:,0])



        self.attitude_cmd[-1] = self.attitude_cmd[-2]
        ax1[2,0].plot(self.time,self.drone_states[:,6],label='real')
        ax1[2,0].plot(self.time,np.array(self.attitude_ff)[:,0],label = 'feedforward')
        ax1[2,0].plot(self.time,np.array(self.attitude_fb)[:,0],label = 'feedback')
        ax1[2,0].set_ylabel('phi[rad]')
        ax1[2,1].plot(self.time,self.drone_states[:,7])
        ax1[2,1].plot(self.time,np.array(self.attitude_ff)[:,1])
        ax1[2,1].plot(self.time,np.array(self.attitude_fb)[:,1])
        ax1[2,1].set_ylabel('theta[rad]')
        ax1[2,2].plot(self.time,self.drone_states[:,8])
        ax1[2,2].plot(self.time,np.array(self.attitude_ff)[:,2])
        ax1[2,2].plot(self.time,np.array(self.attitude_fb)[:,2])
        ax1[2,2].set_ylabel('psi[rad]')
        ax1[2,0].legend()


        # self.forward_bodyrate_cmd[-1] = self.forward_bodyrate_cmd[-2]
        ax1[3,0].plot(self.time,self.drone_states[:,9])
        ax1[3,0].plot(self.time,np.array(self.bodyrate_ff)[:,0],label='feedforward')
        ax1[3,0].plot(self.time,np.array(self.bodyrate_fb)[:,0],label = 'feedback')
        ax1[3,0].set_ylabel('p[rad/s]')
        ax1[3,1].plot(self.time,self.drone_states[:,10])
        ax1[3,1].plot(self.time,np.array(self.bodyrate_ff)[:,1])
        ax1[3,1].plot(self.time,np.array(self.bodyrate_fb)[:,1])
        ax1[3,1].set_ylabel('q[rad/s]')
        ax1[3,2].plot(self.time,self.drone_states[:,11])
        ax1[3,2].plot(self.time,np.array(self.bodyrate_ff)[:,2])
        ax1[3,2].plot(self.time,np.array(self.bodyrate_fb)[:,2])
        ax1[3,2].set_ylabel('r[rad/s]')
        ax1[3,0].legend()

if __name__ == "__main__":
    drone = DroneControlSim()
    dronecontol_ff = DroneControl_feed_foward()
    drone.run()
    drone.plot_states()
    plt.show()
    
