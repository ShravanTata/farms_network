 # distutils: language = c++

"""
-----------------------------------------------------------------------
Copyright 2018-2020 Jonathan Arreguit, Shravan Tata Ramalingasetty
Copyright 2018 BioRobotics Laboratory, École polytechnique fédérale de Lausanne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------------------------------------
"""


from libc.math cimport sqrt as csqrt
from libc.stdlib cimport free, malloc

from .ornstein_uhlenbeck_cy cimport mt19937, mt19937_64, normal_distribution

from typing import List

import numpy as np

from ..core.options import OrnsteinUhlenbeckOptions


cdef void evaluate_a(
    double time,
    double[:] states,
    double[:] drift,
    ornstein_uhlenbeck_params_t params
) noexcept:
    cdef unsigned int j
    for j in range(params.n_dim):
        drift[j] = (params.mu-states[j])/params.tau


cdef void evaluate_b(
    double time,
    double[:] states,
    double[:] diffusion,
    ornstein_uhlenbeck_params_t params
) noexcept:
    cdef unsigned int j
    for j in range(params.n_dim):
        diffusion[j] = params.sigma*csqrt(2.0/params.tau)*params.distribution(params.random_generator)


cdef class OrnsteinUhlenbeckCy(SDESystem):
    """ Ornstein Uhlenheck parameters """

    def __cinit__(
            self,
            n_dim: int,
            noise_options: List[OrnsteinUhlenbeckOptions],
            random_seed: int = None
    ):
        """ C initialization for manual memory allocation """

        self.n_dim = n_dim

        self.params = <ornstein_uhlenbeck_params_t**>malloc(
            self.n_dim * sizeof(ornstein_uhlenbeck_params_t*)
        )

        if self.params is NULL:
            raise MemoryError(
                "Failed to allocate memory for OrnsteinUhlenbeck Parameters"
            )

    def __dealloc__(self):
        """ Deallocate any manual memory as part of clean up """

        if self.params is not NULL:
            free(self.params)

    def __init__(
            self,
            n_dim: int,
            noise_options: List[OrnsteinUhlenbeckOptions],
            random_seed: int = None
    ):
        super().__init__()

        self.initialize_parameters_from_options(noise_options)

    cdef void evaluate_a(self, double time, double[:] states, double[:] drift) noexcept:
        cdef unsigned int j
        cdef ornstein_uhlenbeck_params_t* param

        for j in range(self.n_dim):
            param = self.params[j]
            drift[j] = (param.mu - states[j])/param.tau

    cdef void evaluate_b(self, double time, double[:] states, double[:] diffusion) noexcept:
        cdef unsigned int j
        cdef ornstein_uhlenbeck_params_t* param

        for j in range(self.n_dim):
            param = self.params[j]
            diffusion[j] = param.sigma*(
                csqrt(2.0/param.tau)*(self.distribution(self.random_generator))
            )

    def py_evaluate_a(self, time, states, drift):
        self.evaluate_a(time, states, drift)
        return drift

    def py_evaluate_b(self, time, states, diffusion):
        self.evaluate_b(time, states, diffusion)
        return diffusion

    def initialize_parameters_from_options(self, noise_options, random_seed=123124):
        """ Initialize the parameters from noise options

        # TODO: Remove default random seed in code
        """
        for index in range(self.n_dim):
            noise_option = noise_options[index]
            self.params[index].mu = noise_option.mu
            self.params[index].sigma = noise_option.sigma
            self.params[index].tau = noise_option.tau

        self.random_generator = mt19937_64(random_seed)
        # The distribution should always be mean=0.0 and std=1.0
        self.distribution = normal_distribution[double](0.0, 1.0)
