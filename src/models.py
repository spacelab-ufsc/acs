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

import igrf_utils as iut
from scipy import interpolate
import numpy as np
from scipy.spatial.transform import Rotation 
import math

class GeomagneticField:
    def __init__(self, year):
        self.date = year    
        return
    
    def compute_covariant_componets(self, colatitude, longitude, radius):
        # Load in the file of coefficients
        IGRF_FILE = './IGRF13.shc'
        igrf = iut.load_shcfile(IGRF_FILE, None)

        date = self.date

        f = interpolate.interp1d(igrf.time, igrf.coeffs, fill_value='extrapolate')
        coeffs = f([date])    

        Br, Bt, Bp = iut.synth_values(coeffs.T, radius, colatitude, longitude,
                                igrf.parameters['nmax'])

        return np.array([Br, Bt, Bp])

    def compute_basis_vectors(self, colatitude, longitude, radius):
        r = np.array([np.sin(colatitude)*np.cos(longitude), np.sin(colatitude)*np.sin(longitude), np.cos(colatitude)])
        theta = np.array([np.cos(colatitude)*np.cos(longitude), np.cos(colatitude)*np.sin(longitude), -np.sin(colatitude)])
        phi = np.array([-np.sin(longitude), np.cos(longitude), 0])

        return r,theta,phi

    def evaluate_field(self, colatitude, longitude, radius):

        if colatitude < 0 or colatitude > 180:
            print('Colutatide must be between 0 and 180 degrees') 
            return

        if longitude < -180 or longitude > 180: 
            print('Longitude must be between -180 and 180 degrees')
            return

        if radius < 6371:
            print('Radius must be greater than 6371')
            return
            

        covariant_components = self.compute_covariant_componets(colatitude, longitude, radius)
        basis_vectors = self.compute_basis_vectors(colatitude, longitude, radius)
        
        B = covariant_components[0]*basis_vectors[0] + covariant_components[1]*basis_vectors[1] + covariant_components[2]*basis_vectors[2]

        return B

class Orbit:
    def __init__(self, altitude, alpha, beta, gamma, geomagnetic_field):
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

        self.geomagnetic_field = geomagnetic_field

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

        self.flux_density = self.evaluate_geomagnetic_field()
        vaccum_magnetic_permeability = 4*np.pi*10**(-7)
        self.magnetic_field = self.flux_density/vaccum_magnetic_permeability

    def evaluate_geomagnetic_field(self):
        r = np.sqrt(self.orbital_position[0]**2 + self.orbital_position[1]**2 + self.orbital_position[2]**2)
        theta = np.arccos(self.orbital_position[2]/(r))
        phi = np.arccos(self.orbital_position[0]/(r*np.sin(theta)))

        return self.geomagnetic_field.evaluate_field(theta,phi,r/1000)*10**(-9)

    def propagate_geomagnetic_field(self, dt):
        r = np.sqrt(self.orbital_position[0]**2 + self.orbital_position[1]**2 + self.orbital_position[2]**2)
        theta = np.arccos(self.orbital_position[2]/(r))
        if np.abs(self.orbital_position[0]/(r*np.sin(theta))) - 1 < 1e-6:
            if self.orbital_position[0]/(r*np.sin(theta)) > 0:
                phi = np.arccos(1)
            elif self.orbital_position[0]/(r*np.sin(theta)) < 0:
                phi = np.arccos(-1)
        else:
            phi = np.arccos(self.orbital_position[0]/(r*np.sin(theta)))

        vaccum_magnetic_permeability = 4*np.pi*10**(-7)

        self.flux_density = self.geomagnetic_field.evaluate_field(theta,phi,r/1000)*10**(-9)
        self.previous_magnetic_field = self.magnetic_field
        self.magnetic_field = self.flux_density/vaccum_magnetic_permeability
        self.magnetic_field_derivative = (self.magnetic_field - self.previous_magnetic_field)/dt
        
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

        self.propagate_geomagnetic_field(dt)

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
        
        r = np.sqrt(self.orbital_position[0]**2 + self.orbital_position[1]**2 + self.orbital_position[2]**2)
        theta = np.arccos(self.orbital_position[2]/(r))
        phi = np.arccos(self.orbital_position[0]/(r*np.sin(theta)))
        
        b = self.geomagnetic_field.evaluate_field(theta,phi,r/1000)

        a = Arrow3D([self.orbital_position[0], self.orbital_position[0] + b[0]*100], [self.orbital_position[1], self.orbital_position[1] + b[1]*100], 
                    [self.orbital_position[2], self.orbital_position[2] + b[2]*100], mutation_scale=15, 
                    lw=1, arrowstyle="-|>", color="r")
        
        ax.add_artist(a)
            
        ax.plot3D(xline, yline, zline, 'gray')
        ax.plot3D(xrline, yrline, zrline, 'green')

        plt.show()

class Spacecraft:
    def __init__(self, inertia_matrix, viscous_damping, attitude_control_system):
        self.inertia_matrix = inertia_matrix
        initial_attitude = np.array([0.5,0,0,0])
        initial_velocity = np.array([6.28/8,6.28/4,6.28/2])
        self.viscous_damping = viscous_damping
        self.inertia_matrix_inverse = np.linalg.inv(self.inertia_matrix)

        self.velocity = initial_velocity
        self.attitude = initial_attitude

        self.attitude_control_system = attitude_control_system

    def propagate_state(self, dt, magnetic_field, magnetic_field_derivative):
        normalized_velocity = self.velocity/(np.linalg.norm(self.velocity))
        finite_rotation = np.array([np.cos(np.linalg.norm(self.velocity)*dt/2), normalized_velocity[0]*np.sin(np.linalg.norm(self.velocity)*dt/2),normalized_velocity[1]*np.sin(np.linalg.norm(self.velocity)*dt/2),normalized_velocity[2]*np.sin(np.linalg.norm(self.velocity)*dt/2)])

        self.attitude_control_system.rotate(self.attitude ,dt)
        self.attitude_control_system.compute_torque(dt, magnetic_field, magnetic_field_derivative)

        self.attitude = self.quaternion_multiply(self.attitude, finite_rotation)
        self.velocity = self.velocity + dt*self.inertia_matrix_inverse@(self.attitude_control_system.torque-np.cross(self.velocity, self.inertia_matrix@self.velocity))
    
    def quaternion_multiply(self,quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def attitude_euler_angles(self):
        r = Rotation.from_quat(self.attitude)
        return r.as_euler("xyz", degrees=True)

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
            self.B = -self.Bs
            self.state = np.tan(self.B*np.pi/(2*self.Bs))
            self.k = (1/self.Hc)*np.tan(0.5*np.pi*self.Br/self.Bs)
            self.volume = volume
            self.direction = direction
            self.velocity = np.zeros(direction.size)
            self.initial_direction = direction
            self.torque = 0

        def set_initial_condition(self,magnetic_field):
            magnetic_field_projection = np.dot(magnetic_field, self.direction)
            alpha = 0.5*(1 + magnetic_field_projection/self.Hc)
            beta = np.exp(0)
            self.state = self.k*(magnetic_field_projection + self.Hc*(1-2*((1+alpha+beta*(alpha-1))/(1 + alpha - beta*(alpha-1)))))                
            self.B = 2*self.Bs*np.arctan(self.state)/np.pi

        def propagate_field(self,dt, magnetic_field, magnetic_field_derivative):
            magnetic_field_projection = np.dot(magnetic_field, self.direction)
            magnetic_field_derivative_projection = np.dot(magnetic_field_derivative, self.direction) 
            magnetic_field_rotational_derivative = np.dot(magnetic_field, self.velocity)
            magnetic_field_total_derivative = magnetic_field_derivative_projection + magnetic_field_rotational_derivative

            if magnetic_field_total_derivative >= 0:
                self.state = self.state + dt*self.k*(magnetic_field_derivative_projection + magnetic_field_rotational_derivative)*((magnetic_field_projection - self.state/self.k + self.Hc)/(2*self.Hc))**2
            else:
                self.state = self.state + dt*self.k*(magnetic_field_derivative_projection + magnetic_field_rotational_derivative)*((magnetic_field_projection - self.state/self.k - self.Hc)/(2*self.Hc))**2
            
            self.B = 2*self.Bs*np.arctan(self.state)/np.pi
            self.B = self.B*self.direction

        def compute_torque(self, magnetic_field):
            self.torque = np.cross(self.volume*self.B, magnetic_field)

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