using PyPlot
using DFTK
using LinearAlgebra

include("custom_direct_minimization.jl")

# extend cbrt to complex numbers
function cbrt_cplx(z)
    z = Complex(z)
    real(z) >= 0 ? z^(1//3) : -(-z)^(1//3)
end

# real solution of u + p*u^3 = b, using Cardan formulas
# https://en.wikipedia.org/wiki/Cubic_equation#Cardano's_formula
function cardan(b)
    # we are in the case where p = 1
    p = 1.0
    q = -b
    # the discriminant is R = -(4p^3 + 27q^2) <= 0 when p = 1
    R = -(4p^3 + 27q^2)
    v1 = cbrt_cplx((-q+sqrt(-R/27))/2)
    v2 = cbrt_cplx((-q-sqrt(-R/27))/2)
    v1 + v2
end

# u0 is the real solution of u + u^3 = μ*sin(x) on [0,2π]
μ = 10
B = imag(asin(√(4/27)/μ * 1im))
function u0(x)
    cardan(μ*sin(x))
end
# plot u0
figure(1, figsize=(20,10))
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
subplot(121)
is = range(-0.1, 0.1, length=1000)
plot(is, imag.(u0.(is .* im)), label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
plot(is, [1/√3 for i in is], "k--", label="\$ \\pm 1/\\sqrt{3} \$")
plot(is, [-1/√3 for i in is], "k--")
plot([B,B], [-0.8, 0.8], "r--", label="\$ B_0 \$")
plot([-B,-B], [-0.8, 0.8], "r--")
xlabel("\$ y \$")
legend(loc="upper left", framealpha=1.)

subplot(122)
rs = range(-0.1, 0.1, length=400)
is = range(-0.1, 0.1, length=400)
res = [u0(x + im*y) for x in rs, y in is]
pcolormesh(rs, is, angle.(res)', cmap="hsv")
plot(0, B, "ro")
plot(0, -B, "ro")
colorbar()
savefig("u0.png")

# Set up DFTK framework for Gross-Pitaevskii
Ecut = 1000000
kgrid = (1, 1, 1)
tol = 1e-15

# constant potential V≡1 and PowerNonlinearity parameters
V(r) = 1
C = 1/2
α = 2

# periodic lattice
a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

f(r) = μ * sin(r)
source_term = ExternalFromReal(r -> f(r[1]))

# list of ε's for different numerical simulations
ε_list = [0., 1e-5, 5e-5, 0.0001, 0.001]

# cut function to avoid numerical noise
seuil(x) = abs(x) < 1e-12 ? zero(x) : x

figure(2, figsize=(20,10))
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
marker_list = ["P", "X", "^", "o", "D"]

for (i,ε) in enumerate(ε_list)
    println("---------------------------------")
    println("ε = $(ε)")
    terms = [Kinetic(;scaling_factor=2*ε), # 2 is here to cancel 1/2 term in Kinetic definition
             ExternalFromReal(r -> V(r[1])),
             LocalNonlinearity(ρ -> C * ρ^α),
            ]
    model = Model(lattice; n_electrons=1, terms, spin_polarization=:spinless,
                  symmetries=false)

    basis = PlaneWaveBasis(model; Ecut, kgrid)
    # u0 solution for ε = 0
    u0r = ExternalFromReal(r->u0(r[1]))
    u0G = r_to_G(basis, basis.kpoints[1],
                 ComplexF64.(u0r(basis).potential_values))
    ψ0 = [reshape(u0G, length(u0G), 1)]
    if ε != 0.0
        scfres = custom_direct_minimization(basis, source_term, ψ0; tol)
        println(scfres.energies)
        # check that u is indeed a solution
        ψ = scfres.ψ[1][:, 1]
        Hψ = scfres.ham.blocks[1] * ψ
        Hψr = G_to_r(basis, basis.kpoints[1], Hψ)[:,1,1]
        println("|Hψ-f| = ", abs(real(sum(Hψr - source_term(basis).potential_values[:,1,1])*basis.dvol)))
    else
        ψ = ψ0[1][:,1]
    end

    # plot
    Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
    GGs = Gs[2:div(length(Gs)+1,2)]
    nG = length(GGs)
    m = marker_list[i]
    # select every two component (the others are zero)
    ψG = [ψ[2*k] for k =1:div(nG,2)]
    GGGs = [GGs[2*k] for k =1:div(nG,2)]
    ψGn = ψG[2:end]
    subplot(121)
    semilogy(GGGs, (seuil.(abs.(ψG))), m, label="\$ \\varepsilon = $(ε) \$",
             markersize=10, markevery=2)
    subplot(122)
    if ε != 0
        plot(GGGs[2:end], log.(abs.( seuil.(ψGn) ./ seuil.(ψG[1:end-1] ))), m, label="\$ \\varepsilon = $(ε) \$",
             markersize=10, markevery=4)
    else
        plot(GGGs[2:end], log.(abs.( ψGn ./ ψG[1:end-1] )), m, label="\$ \\varepsilon = $(ε) \$",
             markersize=10, markevery=4)
    end
end

# end up with legend and x labels
subplot(121)
xlabel("\$ |k| \$")
xlim(-20, 500)
legend()
subplot(122)
xlabel("\$ |k| \$")
xlim(-20,500)
ylim(-1, 0)
legend()
savefig("test_decay.png")
