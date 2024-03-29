using PyPlot
using DFTK
using LinearAlgebra

# custom V from computations to have poles
# parameters for LocalNonlinearity(ρ -> C * ρ^α) [nonlinear term in GP]
C = 1/2
α = 2

λ0 = α * C / (2π)
β = 1 + 1e-5
γ = λ0 / β
V(x) = γ * cos(x)

# complex square root
function sqrt_cplx(z)
    z = Complex(z)
    Θ = angle(z)
    r = abs(z)
    Θ = mod(Θ + π, 2π) - π
    √r * exp(Θ/2*1im)
end

# u0 is the real solution of Vu + αC * u^3 = λ0u on [0,2π]
function u0(x)
    sqrt_cplx(λ0 - V(x)) / √(α*C)
end

# branching point
B = log(β + √(β^2-1))

# plot u0
figure(1, figsize=(30,15))
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
Is = range(-1.2*B, 1.2*B, 10000)
is = range(-2*B, 2*B, 200)
rs = range(-0.01, 0.01, 200)
fr(z) = real(u0(z))
fi(z) = imag(u0(z))
f(z)  = u0(z)
subplot(121)
plot(Is, fr.((1im).*Is), label="\$ {\\rm Re}(u_0({\\rm i} y)) \$")
plot(Is, fi.((1im).*Is), label="\$ {\\rm Im}(u_0({\\rm i} y)) \$")
xlabel("\$ y \$")
plot(B, 0, "ro", label="\$ \\pm B_0 \$", markersize=10)
plot(-B, 0, "ro", markersize=10)
legend()

subplot(122)
res = [u0(x + im*y) for x in rs, y in is]
pcolormesh(rs, is, angle.(res)')
plot(0, B, "ro", markersize=10)
plot(0, -B, "ro", markersize=10)
xlabel("\$ x \$")
ylabel("\$ y \$")
colorbar()
savefig("u0_gp.png")

# Set up DFTK framework for Gross-Pitaevskii
Ecut = 1000000
kgrid = (1, 1, 1)
tol = 1e-13

# periodic lattice
a = 2π
lattice = a * [[1 0 0.]; [0 0 0]; [0 0 0]]

# list of ε's for different numerical simulations
ε_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# cut function to avoid numerical noise
seuil(x) = abs(x) > 1e-11 ? x : 0.0

figure(2, figsize=(30,15))
ftsize = 30
rc("font", size=ftsize, serif="Computer Modern")
rc("text", usetex=true)
marker_list = ["P", "X", "^", "o", "D", "s"]

for (i,ε) in enumerate(ε_list)
    println("---------------------------------")
    println("ε = $(ε)")
    terms = [Kinetic(; scaling_factor=2*ε),
             ExternalFromReal(r -> V(r[1])),
             LocalNonlinearity(ρ -> C*(ρ^α)),
            ]
    model = Model(lattice; n_electrons=1, terms, spin_polarization=:spinless,
                  symmetries=false)

    basis = PlaneWaveBasis(model; Ecut, kgrid)
    u0r = ExternalFromReal(r->u0(r[1]))
    u0G = fft(basis, basis.kpoints[1], ComplexF64.(u0r(basis).potential_values))[:,1]
    ψ0 = [reshape(u0G, length(u0G), 1)]
    if ε != 0.0
        scfres = direct_minimization(basis, ψ0; tol, maxiter=10000, show_trace=false)
        ρ = real(scfres.ρ)[:, 1, 1, 1]
        ψ = scfres.ψ[1][:, 1]
        ψr = ifft(basis, basis.kpoints[1], ψ)[:, 1, 1]
        println(scfres.energies)
        println(scfres.optim_res)
    end
    figure(3)
    rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis
    x = a * [r[1] for r in rvecs]                   # only keep the x coordinate
    (ε != 0.0) ? plot(x, ψr) : plot(x, u0.(x))

    Gs = [abs(G[1]) for G in G_vectors(basis, basis.kpoints[1])][:]
    nG = length(Gs)
    m = marker_list[i]
    i += 1
    if ε == 0.0
        # plot fourier
        figure(2)
        subplot(121)
        semilogy(Gs, (seuil.(abs.(u0G))), m, label="\$ \\varepsilon = 0 \$",
                 markersize=10, markevery=20)
        semilogy(Gs, 1e-1 ./ sqrt.(abs.(Gs).^3 .* cosh.(2B .* abs.(Gs))), "--",
                 label="\$ \\frac{1}{(|k|^{3/2} \\sqrt{\\cosh(2B_0k)})} \$")
        subplot(122)
        u0Gn = [u0G[k+1] for k=1:(nG-1)]
        plot(Gs[2:end], log.(abs.( seuil.(u0Gn) ./ seuil.(u0G[1:end-1] ))), m, label="\$ \\varepsilon = 0 \$",
             markersize=10, markevery=20)
        plot(Gs[2:end], log.(sqrt.(abs.(Gs[2:end]).^3 .* cosh.(2B .* abs.(Gs[2:end]))) ./ sqrt.(abs.(Gs[1:end-1]).^3 .* cosh.(2B .* abs.(Gs[1:end-1])))), "--",
             label="\$ \\frac{1}{(|k|^{3/2} \\sqrt{\\cosh(2B_0k)})} \$")
        plot(Gs[2:end], [-B for k in Gs[2:end]], "--", label="\$ -B_0 \$")
    else
        εpow = Int(log10(ε))
        ψG = [ψ[k] for k=1:nG]
        ψGn = [ψ[k+1] for k=1:(nG-1)]

        figure(2)
        subplot(121)
        semilogy(Gs, (seuil.(abs.(ψG))), m, label="\$ \\varepsilon = 10^{$(εpow)} \$",
                 markersize=10, markevery=20)
        subplot(122)
        plot(Gs[2:end], log.(abs.( seuil.(ψGn) ./ seuil.(ψG[1:end-1] ))), m, label="\$ \\varepsilon = 10^{$(εpow)} \$",
             markersize=10, markevery=20)

        println("λ0 = ", λ0)
        println("λε = ", scfres.eigenvalues[1][1])
        println("|λ0-λε| = ", abs(λ0 - scfres.eigenvalues[1][1]))
    end
end

# end up with legend and x labels
subplot(121)
xlabel("\$ |k| \$")
xlim(-50, 1400)
ylim(tol*10, 1)
legend()
subplot(122)
xlabel("\$ |k| \$")
legend()
xlim(-50, 1400)
ylim(-0.1, 0)
savefig("test_decay_gp.png")
