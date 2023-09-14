# custom direct minimization for DFTK

using Optim
using LineSearches

# solving H(ψ)ψ = f
function custom_direct_minimization(basis::PlaneWaveBasis{T}, source_term, ψ0;
                                    prec_type=PreconditionerTPA,
                                    optim_solver=Optim.LBFGS, tol=1e-6, kwargs...) where {T}
    if mpi_nprocs() > 1
        # need synchronization in Optim
        error("Direct minimization with MPI is not supported yet")
    end
    model = basis.model
    @assert iszero(model.temperature)  # temperature is not yet supported
    @assert isnothing(model.εF)        # neither are computations with fixed Fermi level
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)

    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    # we need to copy the reinterpret array here to not raise errors in Optim.jl
    # TODO raise this issue in Optim.jl
    pack(ψ) = copy(DFTK.reinterpret_real(DFTK.pack_ψ(ψ)))
    unpack(x) = DFTK.unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ0))
    unsafe_unpack(x) = DFTK.unsafe_unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ0))

    # this will get updated along the iterations
    H = nothing
    energies = nothing
    ρ = nothing
    f = source_term(basis).potential_values

    # computes energies and gradients
    function fg!(E, G, ψ)
        ψ = unpack(ψ)
        ψr = ifft(basis, basis.kpoints[1], ψ[1][:,1])
        ρ = compute_density(basis, ψ, occupation)
        energies, H = energy_hamiltonian(basis, ψ, occupation; ρ)

        if G !== nothing
            G = unsafe_unpack(G)
            for ik = 1:Nk
                mul!(G[ik], H.blocks[ik], ψ[ik])
                G[ik] .*= 2*filled_occ
                fG = fft(basis, basis.kpoints[ik],
                         ComplexF64.(source_term(basis).potential_values))
                G[ik] .-= 2*filled_occ*fG
            end
        end
        # add source_term
        E = energies.total - 2 * real(sum(f .* ψr) * basis.dvol)
    end

    Pks = [prec_type(basis, kpt) for kpt in basis.kpoints]
    P = DFTK.DMPreconditioner(Nk, Pks, unsafe_unpack)

    kwdict = Dict(kwargs)
    optim_options = Optim.Options(; allow_f_increases=true, show_trace=true, iterations=10000,
                                  x_abstol=tol, f_abstol=-1, g_tol=-1,
                                  x_reltol=-1, f_reltol=-1, kwdict...)
    res = Optim.optimize(Optim.only_fg!(fg!), pack(ψ0),
                         optim_solver(; P, precondprep=DFTK.precondprep!,
                                      linesearch=LineSearches.HagerZhang()),
                         optim_options)
    @show res
    ψ = unpack(res.minimizer)

    # We rely on the fact that the last point where fg! was called is the minimizer to
    # avoid recomputing at ψ
    (; ham=H, basis, energies, converged=true, ρ, ψ, occupation, optim_res=res)
end
