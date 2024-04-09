from typing import Self
from ufl import (
    TestFunctions,
    TrialFunction,
    Identity,
    as_tensor,
    as_vector,
    eq,
    grad,
    det,
    div,
    dev,
    inv,
    tr,
    sqrt,
    conditional,
    gt,
    inner,
    derivative,
    dot,
    ln,
    split,
    acos,
    cos,
    sin,
    lt,
    as_tensor,
    as_vector,
    outer,
    SpatialCoordinate,
)
import ufl
from dolfinx.fem import Constant
from petsc4py import PETSc
from dolfinx.fem import (
    Constant,
    dirichletbc,
    Function,
    FunctionSpace,
    Expression,
    locate_dofs_topological,
)

import numpy as np


# Class that decribes the thermoelastic axial strain problem
class DE_Axial_Symmetric:
    def __init__(self, domain, **kwargs):
        # Create Contants
        self.Gshear_0 = Constant(domain, 1000.0)  # Shear modulus, kPa
        self.lambdaL = Constant(domain, 100.0)  # Locking stretch
        self.Kbulk = Constant(
            domain, 1000.0 * self.Gshear_0.__float__()
        )  # Bulk modulus, kPa
        self.Omega = Constant(domain, 1.00e5)  # Molar volume of fluid
        self.D = Constant(domain, 5.00e-3)  # Diffusivity
        self.chi = Constant(domain, 0.1)  # Flory-Huggins mixing parameter
        self.theta0 = Constant(domain, 298.0)  # Reference temperature
        self.R_gas = Constant(domain, 8.3145e6)  # Gas constant
        self.RT = 8.3145e6 * self.theta0
        #
        self.phi0 = Constant(domain, 0.999)  # Initial polymer volume fraction
        self.mu0 = (
            ln(1.0 - self.phi0) + self.phi0 + self.chi * self.phi0 * self.phi0
        )  # Initialize chemical potential

        if "alpha" in kwargs.keys():
            self.alpha = kwargs["alpha"]

        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement(
            "Lagrange", domain.ufl_cell(), 2
        )  # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # For  pressure
        self.TH = ufl.MixedElement(
            [self.U2, self.P1, self.P1, self.P1]
        )  # Taylor-Hood style mixed element
        self.ME = FunctionSpace(domain, self.TH)  # Total space for all DOFs

        self.w = Function(self.ME)
        self.u, self.p, self.mu, self.c = split(self.w)
        self.w_old = Function(self.ME)
        self.u_old, self.p_old, self.mu_old, self.c_old = split(self.w_old)

        self.u_test, self.p_test, self.mu_test, self.c_test = TestFunctions(self.ME)
        self.dw = TrialFunction(self.ME)

        self.x = SpatialCoordinate(domain)
        self.domain = domain

        # Initialize problem
        self.w.sub(2).interpolate(lambda x: np.full((x.shape[1],), float(self.mu0)))
        self.w_old.sub(2).interpolate(lambda x: np.full((x.shape[1],), float(self.mu0)))

        c0 = 1 / self.phi0 - 1
        self.w.sub(3).interpolate(lambda x: np.full((x.shape[1],), float(c0)))
        self.w_old.sub(3).interpolate(lambda x: np.full((x.shape[1],), float(c0)))

    def WeakForms(self, dt):
        dk = Constant(self.domain, float(dt))
        dx = ufl.dx(metadata={"quadrature_degree": 4})

        # The weak form for the equilibrium equation
        Res_0 = inner(self.Tmat, self.ax_grad_vector(self.u_test)) * self.x[0] * dx

        # The weak form for the auxiliary pressure variable definition
        Res_1 = (
            dot((self.p * self.Je / self.Kbulk + ln(self.Je)), self.p_test)
            * self.x[0]
            * dx
        )

        # The weak form for the mass balance of solvent
        Res_2 = (
            dot((self.c - self.c_old) / dk, self.mu_test) * self.x[0] * dx
            - self.Omega
            * dot(self.Jmat, self.ax_grad_scalar(self.mu_test))
            * self.x[0]
            * dx
        )

        # The weak form for the concentration
        fac = 1 / (1 + self.c)
        fac1 = self.mu - (ln(1.0 - fac) + fac + self.chi * fac * fac)
        fac2 = -(self.Omega * self.Je / self.RT) * self.p
        fac3 = fac1 + fac2
        #
        Res_3 = dot(fac3, self.c_test) * self.x[0] * dx

        # Total weak form
        self.Res = Res_0 + Res_1 + Res_2 + Res_3
        self.a = derivative(self.Res, self.w, self.dw)

    def Kinematics(self):
        # Kinematics
        self.F = self.F_ax_calc(self.u)
        self.J = det(self.F)  # Total volumetric jacobian

        # Elastic volumetric Jacobian
        self.Je = self.Je_calc(self.u, self.c)
        self.Je_old = self.Je_calc(self.u_old, self.c_old)

        #  Normalized Piola stress
        self.Tmat = self.Piola_calc(self.u, self.p)

        #  Normalized species  flux
        self.Jmat = self.Flux_calc(self.u, self.mu, self.c)

    def ax_grad_vector(self, u):
        grad_u = grad(u)
        return as_tensor(
            [
                [grad_u[0, 0], grad_u[0, 1], 0],
                [grad_u[1, 0], grad_u[1, 1], 0],
                [0, 0, u[0] / self.x[0]],
            ]
        )

    # Gradient of scalar field y
    # (just need an extra zero for dimensions to work out)
    def ax_grad_scalar(self, y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.0])

    # Axisymmetric deformation gradient
    def F_ax_calc(self, u):
        dim = len(u)
        Id = Identity(dim)  # Identity tensor
        F = Id + grad(u)  # 2D Deformation gradient
        F33 = 1 + (u[0]) / self.x[0]  # axisymmetric F33, R/R0
        F33 = conditional(eq(self.x[0], 0), 1, F33)
        return as_tensor(
            [[F[0, 0], F[0, 1], 0.0], [F[1, 0], F[1, 1], 0.0], [0.0, 0.0, F33]]
        )  # Full axisymmetric F

    def lambdaBar_calc(self, u):
        F = self.F_ax_calc(u)
        C = F.T * F
        I1 = tr(C)
        lambdaBar = sqrt(I1 / 3.0)
        return lambdaBar

    def zeta_calc(self, u):
        lambdaBar = self.lambdaBar_calc(u)
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        z = lambdaBar / self.lambdaL
        z = conditional(gt(z, 0.95), 0.95, z)  # Prevent the function from blowing up
        beta = z * (3.0 - z**2.0) / (1.0 - z**2.0)
        zeta = (self.lambdaL / (3 * lambdaBar)) * beta
        return zeta

    def zeta0_calc(self):
        # Use Pade approximation of Langevin inverse (A. Cohen, 1991)
        # This is sixth-order accurate.
        z = 1 / self.lambdaL
        z = conditional(gt(z, 0.95), 0.95, z)  # Keep from blowing up
        beta0 = z * (3.0 - z**2.0) / (1.0 - z**2.0)
        zeta0 = (self.lambdaL / 3) * beta0
        return zeta0

    #  Elastic Je
    def Je_calc(self, u, c):
        F = self.F_ax_calc(u)
        detF = det(F)
        #
        detFs = 1.0 + c  # = Js
        Je = detF / detFs  # = Je
        return Je

    # Normalized Piola stress for Arruda_Boyce material
    def Piola_calc(self, u, p):
        F = self.F_ax_calc(u)
        zeta = self.zeta_calc(u)
        zeta0 = self.zeta0_calc()
        Tmat = (zeta * F - zeta0 * inv(F.T)) - self.J * p * inv(F.T) / self.Gshear_0
        return Tmat

    # Normalized species flux
    def Flux_calc(self, u, mu, c):
        F = self.F_ax_calc(u)
        #
        Cinv = inv(F.T * F)
        #
        Mob = (self.D * c) / (self.Omega * self.RT) * Cinv
        #
        Jmat = -self.RT * Mob * self.ax_grad_scalar(mu)
        return Jmat


class DE_2D_PE:
    def __init__(self, domain, **kwargs):
        # Create Contants
        self.Geq_0 = Constant(domain, 15.0)  # Shear modulus, kPa

        self.I_m = Constant(domain, 175.0)  # Locking stretch
        self.Kbulk = 1000.0 * self.Geq_0

        self.vareps_0 = Constant(domain, 8.85e-3)  #  permittivity of free space pF/mm
        self.vareps_r = Constant(domain, 5.5)  #  relative permittivity, dimensionless

        self.vareps = self.vareps_r * self.vareps_0  #  permittivity of the material

        length = 10

        if "length" in kwargs.keys():
            length = kwargs["length"]

        if "phiTot" in kwargs.keys():
            self.phiTot = kwargs["phiTot"]
        else:
            self.phiTot = 1.25 * float(
                length * np.sqrt(float(self.Geq_0) / float(self.vareps))
            )
        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement(
            "Lagrange", domain.ufl_cell(), 2
        )  # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # For  pressure

        self.TH = ufl.MixedElement(
            [self.U2, self.P1, self.P1]
        )  # Taylor-Hood style mixed element

        self.ME = FunctionSpace(domain, self.TH)  # Total space for all DOFs

        self.w = Function(self.ME)

        self.u, self.p, self.phi = split(
            self.w
        )  # displacement u, pressure p, potential  phi
        self.w_old = Function(self.ME)

        self.u_old, self.p_old, self.phi_old = split(
            self.w_old
        )  # displacement u, pressure p, potential  phi

        self.u_test, self.p_test, self.phi_test = TestFunctions(self.ME)
        self.dw = TrialFunction(self.ME)

        self.x = SpatialCoordinate(domain)
        self.domain = domain

    def WeakForms(self, dt):
        dk = Constant(self.domain, float(dt))
        dx = ufl.dx(metadata={"quadrature_degree": 4})

        Res_0 = inner(self.Tmat, self.pe_grad_vector(self.u_test)) * dx

        # The weak form for the pressure
        Res_1 = dot((self.p / self.Kbulk + ln(self.J) / self.J), self.p_test) * dx

        #  The weak form for Gauss's equation
        Res_2 = inner(self.Dmat, self.pe_grad_scalar(self.phi_test)) * dx

        # Total weak form
        self.Res = Res_0 + Res_1 + Res_2

        # Automatic differentiation tangent:
        self.a = derivative(self.Res, self.w, self.dw)

        self.Res_0 = Res_0
        self.Res_1 = Res_1
        self.Res_2 = Res_2

    def Kinematics(self):
        # Kinematics
        F = self.F_pe_calc(self.u)
        self.J = det(F)
        C = F.T * F

        self.F = F
        self.Fdis = self.J ** (-1 / 3) * F
        self.Cdis = self.J ** (-2 / 3) * C

        # Mechanical Cauchy stress
        self.T_mech = self.T_mech_calc(self.u, self.p)

        # Electrostatic Cauchy stress
        self.T_maxw = self.T_maxw_calc(self.u, self.phi)

        # Piola stress
        self.Tmat = self.T_mat_calc(self.u, self.p, self.phi)

        # Referential electric displacement
        self.Dmat = self.Dmat_calc(self.u, self.phi)

    def pe_grad_vector(self, u):
        grad_u = grad(u)
        return as_tensor(
            [
                [grad_u[0, 0], grad_u[0, 1], 0],
                [grad_u[1, 0], grad_u[1, 1], 0],
                [0, 0, 0],
            ]
        )

    def pe_grad_scalar(self, y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.0])

    # Plane strain deformation gradient
    def F_pe_calc(self, u):
        dim = len(u)
        Id = Identity(dim)  # Identity tensor
        F = Id + grad(u)  # 2D Deformation gradient
        return as_tensor(
            [[F[0, 0], F[0, 1], 0], [F[1, 0], F[1, 1], 0], [0, 0, 1]]
        )  # Full pe F

    # Generalized shear modulus for Gent model
    def Geq_Gent_calc(self, u):
        F = self.F_pe_calc(u)
        J = det(F)
        C = F.T * F
        Cdis = J ** (-2 / 3) * C
        I1 = tr(Cdis)
        z = I1 - 3
        z = conditional(gt(z, self.I_m), 0.95 * self.I_m, z)  # Keep from blowing up
        Geq_Gent = self.Geq_0 * (self.I_m / (self.I_m - z))
        return Geq_Gent

    # Mechanical Cauchy stress for Gent material
    def T_mech_calc(self, u, p):
        Id = Identity(3)
        F = self.F_pe_calc(u)
        J = det(F)
        B = F * F.T
        Bdis = J ** (-2 / 3) * B
        Geq = self.Geq_Gent_calc(u)
        T_mech = (1 / J) * Geq * dev(Bdis) - p * Id
        return T_mech

    # Maxwell contribution to the Cauchy stress
    def T_maxw_calc(self, u, phi):
        F = self.F_pe_calc(u)
        e_R = -self.pe_grad_scalar(phi)  # referential electric field
        e_sp = inv(F.T) * e_R  # spatial electric field
        # Spatial Maxwel stress
        T_maxw = self.vareps * (
            outer(e_sp, e_sp) - 1 / 2 * (inner(e_sp, e_sp)) * Identity(3)
        )
        return T_maxw

    # Piola  stress
    def T_mat_calc(self, u, p, phi):
        Id = Identity(3)
        F = self.F_pe_calc(u)
        J = det(F)
        #
        T_mech = self.T_mech_calc(u, p)
        #
        T_maxw = self.T_maxw_calc(u, phi)
        #
        T = T_mech + T_maxw
        #
        Tmat = J * T * inv(F.T)
        return Tmat

    # Referential electric displacement
    def Dmat_calc(self, u, phi):
        F = self.F_pe_calc(u)
        J = det(F)
        C = F.T * F
        e_R = -self.pe_grad_scalar(phi)  # referential electric field
        Dmat = self.vareps * J * inv(C) * e_R
        return Dmat


class DE_3D:
    def __init__(self, domain, **kwargs):
        # Create Contants
        self.Geq_0 = Constant(domain, 15.0)  # Shear modulus, kPa

        self.I_m = Constant(domain, 175.0)  # Locking stretch
        self.Kbulk = 1000.0 * self.Geq_0

        self.vareps_0 = Constant(domain, 8.85e-3)  #  permittivity of free space pF/mm
        self.vareps_r = Constant(domain, 5.5)  #  relative permittivity, dimensionless

        self.vareps = self.vareps_r * self.vareps_0  #  permittivity of the material

        length = 10

        if "length" in kwargs.keys():
            length = kwargs["length"]

        if "phiTot" in kwargs.keys():
            self.phiTot = kwargs["phiTot"]
        else:
            self.phiTot = float(
                length * np.sqrt(float(self.Geq_0) / float(self.vareps))
            )
        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement(
            "Lagrange", domain.ufl_cell(), 2
        )  # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # For  pressure

        self.TH = ufl.MixedElement(
            [self.U2, self.P1, self.P1]
        )  # Taylor-Hood style mixed element

        self.ME = FunctionSpace(domain, self.TH)  # Total space for all DOFs

        self.w = Function(self.ME)

        self.u, self.p, self.phi = split(
            self.w
        )  # displacement u, pressure p, potential  phi
        self.w_old = Function(self.ME)

        self.u_old, self.p_old, self.phi_old = split(
            self.w_old
        )  # displacement u, pressure p, potential  phi

        self.u_test, self.p_test, self.phi_test = TestFunctions(self.ME)
        self.dw = TrialFunction(self.ME)

        self.x = SpatialCoordinate(domain)
        self.domain = domain

    def WeakForms(self, dt):
        dk = Constant(self.domain, float(dt))
        dx = ufl.dx(metadata={"quadrature_degree": 4})

        Res_0 = inner(self.Tmat, grad(self.u_test)) * dx

        # The weak form for the pressure
        Res_1 = dot((self.p / self.Kbulk + ln(self.J) / self.J), self.p_test) * dx

        #  The weak form for Gauss's equation
        Res_2 = inner(self.Dmat, grad(self.phi_test)) * dx

        # Total weak form
        self.Res = Res_0 + Res_1 + Res_2

        # Automatic differentiation tangent:
        self.a = derivative(self.Res, self.w, self.dw)

        self.Res_0 = Res_0
        self.Res_1 = Res_1
        self.Res_2 = Res_2

    def Kinematics(self):
        # Kinematics
        F = self.F_calc(self.u)
        self.J = det(F)
        C = F.T * F

        self.F = F
        self.Fdis = self.J ** (-1 / 3) * F
        self.Cdis = self.J ** (-2 / 3) * C

        # Mechanical Cauchy stress
        self.T_mech = self.T_mech_calc(self.u, self.p)

        # Electrostatic Cauchy stress
        self.T_maxw = self.T_maxw_calc(self.u, self.phi)

        # Piola stress
        self.Tmat = self.T_mat_calc(self.u, self.p, self.phi)

        # Referential electric displacement
        self.Dmat = self.Dmat_calc(self.u, self.phi)

    # Deformation gradient
    def F_calc(self, u):
        Id = Identity(3)
        F = Id + grad(u)
        return F

    # Generalized shear modulus for Gent model
    def Geq_Gent_calc(self, u):
        F = self.F_calc(u)
        J = det(F)
        C = F.T * F
        Cdis = J ** (-2 / 3) * C
        I1 = tr(Cdis)
        z = I1 - 3
        z = conditional(gt(z, self.I_m), 0.95 * self.I_m, z)  # Keep from blowing up
        Geq_Gent = self.Geq_0 * (self.I_m / (self.I_m - z))
        return Geq_Gent

    # Mechanical Cauchy stress for Gent material
    def T_mech_calc(self, u, p):
        Id = Identity(3)
        F = self.F_calc(u)
        J = det(F)
        B = F * F.T
        Bdis = J ** (-2 / 3) * B
        Geq = self.Geq_Gent_calc(u)
        T_mech = (1 / J) * Geq * dev(Bdis) - p * Id
        return T_mech

    # Maxwell contribution to the Cauchy stress
    def T_maxw_calc(self, u, phi):
        F = self.F_calc(u)
        e_R = -grad(phi)  # referential electric field
        e_sp = inv(F.T) * e_R  # spatial electric field
        # Spatial Maxwel stress
        T_maxw = self.vareps * (
            outer(e_sp, e_sp) - 1 / 2 * (inner(e_sp, e_sp)) * Identity(3)
        )
        return T_maxw

    # Piola  stress
    def T_mat_calc(self, u, p, phi):
        Id = Identity(3)
        F = self.F_calc(u)
        J = det(F)
        #
        T_mech = self.T_mech_calc(u, p)
        #
        T_maxw = self.T_maxw_calc(u, phi)
        #
        T = T_mech + T_maxw
        #
        Tmat = J * T * inv(F.T)
        return Tmat

    # Referential electric displacement
    def Dmat_calc(self, u, phi):
        F = self.F_calc(u)
        J = det(F)
        C = F.T * F
        e_R = -grad(phi)  # referential electric field
        Dmat = self.vareps * J * inv(C) * e_R
        return Dmat
