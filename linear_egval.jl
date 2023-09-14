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
Ecut = 1e6
kgrid = (1, 1, 1)
tol = 1e-20
basis = PlaneWaveBasis(model; Ecut, kgrid)

# cut function to avoid numerical noise
seuil(x) = abs(x) > tol ? x : 0.0

# solution with scf (actually converges in one iteration)
scfres = self_consistent_field(basis; tol, damping=0.5)
@show scfres.eigenvalues[1][1]
# ground state density and wave function
ρ = real(scfres.ρ)[:, 1, 1, 1]
ψ = scfres.ψ[1][:, 1]
ψr = ifft(basis, basis.kpoints[1], ψ)[:, 1, 1]

# plot Fourier coefficients and decrease rate
figure(1, figsize=(30,15))
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
m = "P"

# Fourier modes
Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
# solution is even because V is even
nG = div(length(Gs),2)
GGs = [Gs[2k+1] for k=0:nG]

# Fourier coefficients to plots
ψG = [ψ[2k+1] for k=0:nG]
ψGn = [ψ[2(k+1)+1] for k=0:(nG-1)]

subplot(121)
semilogy(GGs, (seuil.(abs.(ψG))), m, label = "\$|\\widehat{u}_k|\$",
         markersize=10, markevery=10)
semilogy(GGs, 1e-1 ./ sqrt.(abs.(GGs).^4 .* cosh.(2B0 .* abs.(GGs))), "--",
         label="\$ \\frac{1}{(|k|^{4} \\sqrt{\\cosh(2B_0k)})} \$")
#  semilogy(GGs, 1e-5 ./ sqrt.(cosh.(2B0 .* abs.(GGs))), "--",
#           label="\$ \\frac{1}{\\sqrt{\\cosh(2B_0k)}} \$")
subplot(122)
plot(GGs[2:end], 1/2 * log.(abs.( seuil.(ψGn) ./ seuil.(ψG[1:end-1] ))), m,
     markersize=10, markevery=10)
plot(Gs[2:end], log.(sqrt.(abs.(Gs[2:end]).^4 .* cosh.(2B0 .* abs.(Gs[2:end]))) ./ sqrt.(abs.(Gs[1:end-1]).^4 .* cosh.(2B0 .* abs.(Gs[1:end-1])))), "--",
     label="\$ \\frac{1}{(|k|^{4} \\sqrt{\\cosh(2B_0k)})} \$")
plot(GGs[2:end], [-B0 for k in GGs[2:end]], "--", label="\$ -B_0 \$")

# end up with legend and x labels
subplot(121)
xlabel("\$ |k| \$")
xlim(-50, 800)
ylim(tol/100, 1)
legend()
subplot(122)
xlabel("\$ |k| \$")
legend()
xlim(-50, 800)
ylim(-0.5, 0)
savefig("test_decay_linear_egval.png")
