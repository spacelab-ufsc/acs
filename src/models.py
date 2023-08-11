"""
 * models.py
 * 
 * Copyright (C) 2021, SpaceLab.
 * 
 * This file is part of an ACS simulation for FloripaSat-2
 * 
 * ACS simulation for FloripaSat-2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * ACS simulation for FloripaSat-2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with ACS simulation for FloripaSat-2. If not, see <http://www.gnu.org/licenses/>.
 * 
"""

from operator import imod
import igrf_utils as iut
from scipy import interpolate
import numpy as np
from scipy.spatial.transform import Rotation 
import math
import pandas as pd
import matplotlib.pyplot as plt
from math import sin
from pickle import LONG
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline

class GeomagneticField:
    def __init__(self):
        return

    def evaluate_field(self,date_years, altitude, latitude, longitude, coordinate_systen):
        """
            date in decimmal years
            altitude in kilometers
            latitude in degrees from the equator in the north direction
            longitude in degrees from the greenwich meridian in the east direction
            coordinate_system: "ECI" for Earth Centered Inertial and "TSC" for tangential spherical coordinates
        """

        if altitude < 0:
            print("Altitude must be greater than zero")
            return

        if latitude < -90 or latitude > 90:
            print("Latitude must be between -90 and 90 degrees")
            return

        if longitude < -180 or longitude > 190:
            print("Latitude must be between -180 and 180 degrees")
            return

        # Load in the file of coefficients
        IGRF_FILE = r'./IGRF13.shc'
        igrf = iut.load_shcfile(IGRF_FILE, None)

        date=date_years
        alt=6371 + altitude
        lat=latitude
        colat=90-lat 
        lon=longitude
            
        # Interpolate the geomagnetic coefficients to the desired date(s)
        f = interpolate.interp1d(igrf.time, igrf.coeffs, fill_value='extrapolate')
        coeffs = f(date)    

        # Compute the main field B_r, B_theta and B_phi value for the location(s) 
        Br, Bt, Bp = iut.synth_values(coeffs.T, alt, colat, lon,
                                    igrf.parameters['nmax'])

        # Rearrange to X, Y, Z components 
        X = Bt; Y = Bp; Z = Br

        if coordinate_systen == 'TSC':
            return Br, Bt, Bp 

        L0 = 99.9677494687
        L1 = 360.98564736628603
        L2 = 2.907879*10**(-13)
        L3 = -5.302*10**(-22)

        sidereal_time = L0 + L1*(date*365) + L2*(date*365)**2 + L3*(date*365)**3 + longitude

        sidereal_time = sidereal_time % 360

        X_ecf= Br*np.cos(sidereal_time*np.pi/180) - Bt*np.sin(sidereal_time*np.pi/180)
        Y_ecf= Br*np.sin(sidereal_time*np.pi/180) + Bt*np.cos(sidereal_time*np.pi/180)
        Z_ecf= Bp

        X_eci = X_ecf*np.cos(latitude*np.pi/180) - Z_ecf*np.sin(latitude*np.pi/180)
        Y_eci = Y_ecf
        Z_eci = Z_ecf*np.cos(latitude*np.pi/180) + X_ecf*np.sin(latitude*np.pi/180)

        return X_eci, Y_eci, Z_eci

class UniformMagneticField:
    def __init__(self, field_intensity, field_direction):
        self.field_intensity = field_intensity
        self.field_direction = field_direction

    def evaluate_field(self):
        return self.field_intensity*self.field_direction

class Orbit:
    def __init__(self, altitude, alpha, beta, gamma):
        gravitational_constant = 6.674*10**(-11)
        earth_mass = 5.972*10**(24)
        self.altitude = altitude #m
        earth_radius = 6731*10**3 #m
        self.radius = earth_radius + altitude
        self.velocity = np.sqrt(gravitational_constant*earth_mass/(self.radius)**3)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.z_rotation_matrix = np.zeros((3,3))
        self.y_rotation_matrix = np.zeros((3,3))
        self.x_rotation_matrix = np.zeros((3,3))

        self.flux_density = np.zeros(3)
        self.previous_magnetic_field = np.zeros(3)
        self.magnetic_field = np.zeros(3)
        self.magnetic_field_derivative = np.zeros(3)
        
        self.assemble_orbital_plane_rotation_matrices()

        self.orbital_plane_rotation = Rotation.from_matrix(self.y_rotation_matrix@self.z_rotation_matrix@self.x_rotation_matrix)

        self.initial_position = np.array([self.radius, 0,0])

        self.orbital_angular_position = 0
        self.base_orbital_position = self.initial_position
        self.orbital_position = self.orbital_plane_rotation.apply(self.base_orbital_position)

        
    def assemble_orbital_plane_rotation_matrices(self):

        self.z_rotation_matrix = np.array([ [np.cos(self.gamma), -np.sin(self.gamma),       0],
                                       [np.sin(self.gamma),  np.cos(self.gamma),       0],
                                       [     0            ,        0           ,       1] ])

        self.y_rotation_matrix = np.array([ [ np.cos(self.beta),        0           , np.sin(self.beta)],
                                       [     0            ,        1           ,        0],
                                       [-np.sin(self.beta),        0           , np.cos(self.beta)] ])

        self.x_rotation_matrix = np.array([ [     1            ,        0,                   0],
                                       [     0            ,  np.cos(self.alpha),-np.sin(self.alpha)],
                                       [     0            ,  np.sin(self.alpha), np.cos(self.alpha)] ])

    def propagate_orbit(self,dt):
        self.orbital_angular_position = self.orbital_angular_position + dt*self.velocity
        self.orbit_motion_rotation = Rotation.from_matrix( np.array([ [np.cos(self.orbital_angular_position), -np.sin(self.orbital_angular_position),       0],
                                                        [np.sin(self.orbital_angular_position),  np.cos(self.orbital_angular_position),       0],
                                       [     0            ,        0           ,       1]] ))

        self.base_orbital_position = self.orbit_motion_rotation.apply(self.initial_position)
        self.orbital_position = self.orbital_plane_rotation.apply(self.base_orbital_position)

    def plot(self):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        n = 100

        ax.scatter(self.base_orbital_position[0], self.base_orbital_position[1], self.base_orbital_position[2])
        ax.scatter(self.orbital_position[0], self.orbital_position[1], self.orbital_position[2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        l = np.linspace(0,2*np.pi,n)

        R = self.radius

        zline = np.zeros(n)
        xline = np.zeros(n)
        yline = np.zeros(n)

        zrline = np.zeros(n)
        xrline = np.zeros(n)
        yrline = np.zeros(n)

        for i in range(n):
            zline[i] = 0
            xline[i] = R*np.cos(l[i])
            yline[i] = R*np.sin(l[i])
            
            xrline[i], yrline[i], zrline[i]  = self.orbital_plane_rotation.apply(np.array([xline[i],yline[i],zline[i]]))

        plt.show()

class AttitudeControlSystem:
    def __init__(self):
        self.hysteresis_rods = []
        self.permanent_magnets = []
        self.torque = np.zeros(3)
        return
    
    class HysteresisRod:
        def __init__(self, coercivity, remanence, saturation_field, volume, direction):
            self.Hc = coercivity
            self.Br = remanence
            self.Bs = saturation_field
            self.B = 0
            self.state = np.tan(self.B*np.pi/(2*self.Bs))
            self.k = (1/self.Hc)*np.tan(0.5*np.pi*self.Br/self.Bs)
            self.volume = volume
            self.direction = direction
            self.velocity = np.zeros(direction.size)
            self.initial_direction = direction
            self.torque = 0

        def set_initial_attitude(self, attitude):
            r = Rotation.from_quat(attitude)
            self.direction = r.apply(self.initial_direction)

        def set_initial_condition(self,magnetic_field, magnetic_field_time_derivative):
            magnetic_field_projection = np.dot(magnetic_field, self.direction)
            magnetic_field_time_derivative_projection = np.dot(magnetic_field_time_derivative, self.direction)
            
            if magnetic_field_time_derivative_projection >= 0:
                k = (1/self.Hc)*np.tan(0.5*np.pi*self.Br/self.Bs)
                self.B = self.Bs*(2/np.pi)*np.arctan(k*(magnetic_field_projection-self.Hc))
                self.state = np.tan(0.5*np.pi*self.B/self.Bs)
            else: 
                k = (1/self.Hc)*np.tan(0.5*np.pi*self.Br/self.Bs)
                self.B = self.Bs*(2/np.pi)*np.arctan(k*(magnetic_field_projection+self.Hc))
                self.state = np.tan(0.5*np.pi*self.B/self.Bs)


        def propagate_field(self,dt, magnetic_field, magnetic_field_derivative):
            magnetic_field_projection = np.dot(magnetic_field, self.direction)
            magnetic_field_derivative_projection = np.dot(magnetic_field_derivative, self.direction) 

            if magnetic_field_derivative_projection >= 0:
                self.state = self.state + dt*self.k*(magnetic_field_derivative_projection)*((magnetic_field_projection - self.state/self.k + self.Hc)/(2*self.Hc))**2
            else:
                self.state = self.state + dt*self.k*(magnetic_field_derivative_projection)*((magnetic_field_projection - self.state/self.k - self.Hc)/(2*self.Hc))**2
            
            self.B = 2*self.Bs*np.arctan(self.state)/np.pi
            self.B = self.B

        def compute_torque(self, magnetic_field):
            self.torque = np.cross(self.volume*self.B*self.direction, magnetic_field)

    class PermanentMagnet:
        def __init__(self, remanence, volume, direction):
            self.Br = remanence
            self.volume = volume
            self.direction = direction
            self.initial_direction = direction
            self.B = self.Br*self.direction
            

        def compute_torque(self, magnetic_field):
            self.torque = np.cross(self.volume*self.B, magnetic_field)

    def add_hysteresis_rod(self, coercivity, remanence, saturation_field, volume, direction):
        self.hysteresis_rods.append(self.HysteresisRod(coercivity, remanence, saturation_field, volume, direction))
    
    def add_permanent_magnet(self, remanence, volume, direction):
        self.permanent_magnets.append(self.PermanentMagnet(remanence, volume, direction))
    
    def compute_torque(self,dt, magnetic_field, magnetic_field_derivative):
        torque = np.zeros(3)
        
        for rod in self.hysteresis_rods:
            rod.propagate_field(dt, magnetic_field, magnetic_field_derivative)
            rod.compute_torque(magnetic_field)
            torque = torque + rod.torque
        
        for magnet in self.permanent_magnets:
            magnet.compute_torque(magnetic_field)
            torque = torque + magnet.torque

        self.torque = torque

    def rotate(self, quaternion,dt):
        r = Rotation.from_quat(quaternion)
        for rod in self.hysteresis_rods:
            last_direction = rod.direction
            rod.direction = r.apply(rod.initial_direction)
            rod.velocity = (rod.direction - last_direction)/dt
        
        for magnet in self.permanent_magnets:
            magnet.direction = r.apply(magnet.initial_direction)     

class Spacecraft:
    def __init__(self, inertia_matrix, viscous_damping, attitude_control_system):
        self.inertia_matrix = inertia_matrix
        initial_attitude = np.array([0.5,0,0,0])
        initial_velocity = np.array([25*np.pi/180,25*np.pi/180,25*np.pi/180])
        self.viscous_damping = viscous_damping
        self.inertia_matrix_inverse = np.linalg.inv(self.inertia_matrix)

        self.velocity = initial_velocity
        self.attitude = initial_attitude

        self.attitude_control_system = attitude_control_system

    def propagate_state(self, dt, magnetic_field, magnetic_field_derivative):
        normalized_velocity = self.velocity/(np.linalg.norm(self.velocity))
        finite_rotation = np.array([normalized_velocity[0]*np.sin(np.linalg.norm(self.velocity)*dt/2),normalized_velocity[1]*np.sin(np.linalg.norm(self.velocity)*dt/2),normalized_velocity[2]*np.sin(np.linalg.norm(self.velocity)*dt/2),np.cos(np.linalg.norm(self.velocity)*dt/2)])

        self.attitude_control_system.rotate(self.attitude ,dt)
        self.attitude_control_system.compute_torque(dt, magnetic_field, magnetic_field_derivative)

        self.attitude = self.quaternion_multiply(self.attitude, finite_rotation)
        self.velocity = self.velocity + dt*self.inertia_matrix_inverse@(self.attitude_control_system.torque-np.cross(self.velocity, self.inertia_matrix@self.velocity))
    
    def quaternion_multiply(self,quaternion0, quaternion1):
        x0, y0, z0, w0 = quaternion0
        x1, y1, z1, w1 = quaternion1
        return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)

    def attitude_euler_angles(self):
        r = Rotation.from_quat(self.attitude)
        return r.as_euler("xyz", degrees=True)

    def local_geomagnetic_field_time_derivative(self,magnetic_field, magnetic_field_directional_derivative, time):
        return np.dot(magnetic_field,np.cross(self.velocity, self.attitude_control_system.hysteresis_rods[0].direction))*self.attitude_control_system.hysteresis_rods[0].direction + np.dot(magnetic_field,self.attitude_control_system.hysteresis_rods[0].direction)*np.cross(self.velocity, self.attitude_control_system.hysteresis_rods[0].direction)

    def plot_cube(self):

        points = np.array( [ [0,0,0],[0,0,1],[0,1,1],[0,1,0],[0,0,0], [1,0,0],[1,0,1],[1,1,1],[1,1,0],[1,0,0], [0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,0], [0,0,1],[1,0,1],[1,1,1],[0,1,1],[0,0,1] ]  )

        r = Rotation.from_quat(self.attitude)
        points = r.apply(points)

        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111, projection='3d')

        xline = []
        yline = []
        zline = []

        for i in range(1,21):
            xline.append(points[i-1][0])
            yline.append(points[i-1][1])
            zline.append(points[i-1][2])

            if i % 5 == 0:
                ax.plot3D(xline, yline, zline, 'gray')
                xline = []
                yline = []
                zline = []

        plt.show()

   


hysteresis_bar = AttitudeControlSystem()
hysteresis_bar.add_hysteresis_rod(coercivity=0.8, remanence=0.06, saturation_field=0.08, volume=17.6*10**(-6), direction=np.array([0,1,0]))
rigid_body = Spacecraft(inertia_matrix=np.diag([0.01, 0.01, 0.01]), viscous_damping=0, attitude_control_system=hysteresis_bar)
magnetic_field = UniformMagneticField(field_intensity=20, field_direction=np.array([0,1,0]))

rigid_body.velocity = np.array([25*np.pi/180,0,0])
rigid_body.attitude = np.array([np.sin(np.pi/4),0,0,np.cos(np.pi/4)])
magnetic_field_derivative = rigid_body.local_geomagnetic_field_time_derivative(magnetic_field.evaluate_field(),0,0)
hysteresis_bar.hysteresis_rods[0].set_initial_attitude(rigid_body.attitude)
hysteresis_bar.hysteresis_rods[0].set_initial_condition(magnetic_field.evaluate_field(), magnetic_field_derivative)

total_time = 1*60 #s
dt = 0.1

computation_steps = int(total_time/dt)
velocity = np.zeros((computation_steps,3))

for i in range(computation_steps):
    velocity[i] = rigid_body.velocity
    rigid_body.propagate_state(dt, magnetic_field.evaluate_field(), rigid_body.local_geomagnetic_field_time_derivative(magnetic_field.evaluate_field(), 0,0))
