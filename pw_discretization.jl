using PyPlot
using DFTK
using LinearAlgebra
using DoubleFloats
using GenericLinearAlgebra

# model
γ = 5e2
V(x) = 1 / (1/γ + sin(x)^2)
B0 = asinh(1/√γ)
@show B0
terms = [Kinetic(; scaling_factor=2),
         ExternalFromReal(r -> V(r[1]))]
a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]
model = Model(lattice; n_electrons=1, terms,
              spin_polarization=:spinless, symmetries=false)
model = convert(Model{Double64}, model)

# Set up DFTK framework for linear eigenvalue problem
kgrid = (1, 1, 1)
tol = 1e-13
# cut function to avoid numerical noise
seuil(x) = abs(x) > tol ? x : 0.0

# reference solution and ground state wave function / eigenvalue
N_ref = 1e3
Ecut_ref = N_ref^2/2
basis_ref = PlaneWaveBasis(model; Ecut=Ecut_ref, kgrid)
Gs = DFTK.G_vectors(basis_ref, basis_ref.kpoints[1])
Gs = [G[1] for G in Gs]
ham_ref = Hamiltonian(basis_ref)
res_ref = lobpcg_hyper(ham_ref[1],
                       DFTK.random_orbitals(basis_ref, basis_ref.kpoints[1], 2);
                       prec = PreconditionerTPA(ham_ref[1]),
                       maxiter=1000, tol)
λ_ref = res_ref.λ[1]
X_ref = res_ref.X[:,1]
@show res_ref.n_iter
@show λ_ref

# solve problem for various Ecuts
N_list = [n*1e1 for n = 1:30]
λ_list = []
H1_list = []

for N in N_list
    # PW discretization basis
    println("============================")
    Ecut = N^2 / 2
    @show Ecut
    @show N
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    ham = Hamiltonian(basis)
    res = lobpcg_hyper(ham[1],
                       DFTK.random_orbitals(basis, basis.kpoints[1], 2);
                       prec = PreconditionerTPA(ham[1]),
                       maxiter=1000, tol)
    @show res.n_iter

    # ground state
    λ = res.λ[1]
    X = res.X[:,1]
    @show λ

    # compute error
    append!(λ_list, λ - λ_ref)
    Xr = DFTK.transfer_blochwave_kpt(X, basis, basis.kpoints[1],
                                     basis_ref, basis_ref.kpoints[1])[:,1]
    errX = X_ref - (Xr'X_ref)*Xr
    append!(H1_list, norm(abs2.(Gs) .* errX))
end

#  convergence of the eigenvalues and wave functions
figure(1, figsize=(30,15))
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
m = "P"

subplot(121)
semilogy(N_list, λ_list, m, label="\$\\lambda_N - \\lambda\$", markersize=10)
semilogy(N_list, 1e-1 ./ (N_list.^2 .* exp.(2B0 .* N_list)), "--", label="\$ \\exp(-2B_0N)/N^2 \$")
semilogy(N_list, H1_list, m, label="\$\\| u_{N} - u \\|_{H^1}\$",markersize=10)
semilogy(N_list, 1 ./ (exp.(B0 .* N_list)), "--", label="\$ \\exp(-B_0N) \$")
legend()

xlabel("\$ N \$")

subplot(122)
plot(N_list[2:end], 0.1 .* log.( λ_list[2:end] ./ λ_list[1:end-1]), m, label = "\$\\lambda_N -  \\lambda\$", markersize=10)
plot(N_list[2:end], -0.1 .* log.((N_list[2:end].^2 .* exp.(2B0 .* N_list[2:end])) ./
                                 (N_list[1:end-1].^2 .* exp.(2B0 .* N_list[1:end-1]))),
    "--", label="\$ \\exp(-2B_0N)/N^2 \$")
plot(N_list[2:end], 0.1 .* log.( H1_list[2:end] ./ H1_list[1:end-1]), m, label = "\$\\|u_N -  u\\|_{H^1}\$", markersize=10)
plot(N_list[2:end], [-B0 for n in N_list[2:end]], "--", label = "\$-B_0 \$", )
plot(N_list[2:end], [-2B0 for n in N_list[2:end]], "--", label = "\$-2B_0 \$", )
ylim(-0.5,0.0)
legend()

xlabel("\$N\$")

savefig("test_pw_discretization.png")
