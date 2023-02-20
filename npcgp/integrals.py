import torch
import math


@torch.jit.script
def full_I(
    ts,  # Nt x D or B x Nt x D
    alphas,  # D
    pgs,  # D
    wgs,  # B x D x Ng
    thetags,  # B x D x Ng
    betags,  # B x D x Ng
    zgs,  # Mg x D
    qgs,  # B x D x Mg
    pus,  # D
    wus,  # B x Q x Nu
    thetaus,  # B x Q x Nu x D
    betaus,  # B x Q x Nu
    zus,  # Mu x D
    qus,  # B x Q x Mu
    ampu,  # 1
):
    """
    Computes full form of pathwise sampling integral shown in Equation 8
    of the appendix. Uses JIT to speed up the large number of pointwise
    operations involved in computation.
    """
    if len(ts.shape) == 2:
        # t = Nt x 1
        tb = ts[None, None, :, None, :, None]
        kk_einstr = "bij, dtkij -> btki"
    else:
        tb = ts[:, None, :, None, :, None]
        kk_einstr = "bij, btkij -> btki"
    # B x Q x Nt x Nbu x D x Nbg
    Ipim = I_phi_imag(
        tb,
        alphas[None, None, None, None, :, None],
        thetags[:, None, None, None, :, :],
        betags[:, None, None, None, :, :],
        thetaus[:, :, None, :, :, None],
    )

    # B x Q x Nt x Nbu x D x Mg
    Ikim = I_k_imag(
        tb,
        alphas[None, None, None, None, :, None],
        pgs[None, None, None, None, :, None],
        zgs.T[None, None, None, None, :, :],
        thetaus[:, :, None, :, :, None],
    )
    # convert wgs to complex here as inputs to einsum must be same dtype
    It1a = torch.einsum(
        "bij, bqtkij -> bqtki",
        torch.complex(wgs, torch.tensor(0.0, device=wgs.device)),
        Ipim,
    )
    It2a = torch.einsum(
        "bij, bqtkij -> bqtki",
        torch.complex(qgs, torch.tensor(0.0, device=qgs.device)),
        Ikim,
    )
    Itap = It1a + It2a
    Ita = Itap.prod(4)

    ef = torch.exp(
        torch.complex(torch.tensor(0.0, device=betaus.device), betaus[:, :, None, :])
    )
    Ita = torch.real((ef * Ita + torch.conj(ef) * torch.conj(Ita))) / 2
    Ita = torch.einsum("bqi, bqti -> bqt", wus, Ita)

    # B x Nt x Mu x D x Nb
    Ipk = ampu * I_phi_k(
        tb[:, 0, :, :, :, :],
        alphas[None, None, None, :, None],
        thetags[:, None, None, :, :],
        betags[:, None, None, :, :],
        pus[None, None, None, :, None],
        zus[None, None, :, :, None],
    )

    # Nt x Mu x D x Mg
    Ikk = ampu * I_k_k(
        tb[:, 0, :, :, :, :],
        alphas[None, None, None, :, None],
        pgs[None, None, None, :, None],
        zgs.T[None, None, None, :, :],
        pus[None, None, None, :, None],
        zus[None, None, :, :, None],
    )

    Itb = torch.einsum("bij, btkij -> btki", wgs, Ipk) + torch.einsum(
        kk_einstr, qgs, Ikk
    )
    # Prod not supported in jit, same as
    Itb = Itb.prod(3)
    Itb = torch.einsum("bqi, bti -> bqt", qus, Itb)

    return Ita + Itb


def I_interdom_single(t, pg, thet):
    """1D integral for interdomain sampling."""
    const = torch.sqrt(math.pi / pg)
    return const * torch.exp(torch.complex(-(thet**2) / (4 * pg), t * thet))


def I_interdom(ts, pgs, thets, betas):  # Nt x D  # D  # B x Q x D x Nb  # B x Q x Nb
    """Extention of I_interdom_single to multidimensional input."""
    # B x Q x Nt x D x Nb
    It = I_interdom_single(
        ts[None, None, :, :, None],
        pgs[None, None, None, :, None],
        thets[:, :, None, :, :],
    ).prod(
        3
    )  # B x Q x Nt x Nb

    ef = torch.exp(
        torch.complex(torch.tensor(0.0, device=betas.device), betas[:, :, None, :])
    )
    It = torch.real((ef * It + torch.conj(ef) * torch.conj(It))) / 2
    return It


@torch.jit.script
def I_phi_phi(t, alpha, thet1, thet2, beta1, beta2):
    pi = math.pi
    coeff = 0.5 * torch.sqrt(pi / alpha)
    ea1 = torch.exp(-((thet1 + thet2) ** 2) / (4.0 * alpha))
    ea2 = torch.cos(beta1 - beta2 - t * thet2) + torch.exp(
        thet1 * thet2 / alpha
    ) * torch.cos(beta1 + beta2 + t * thet2)
    return coeff * ea1 * ea2


@torch.jit.script
def I_k_phi(t, alpha, p1, z1, thet2, beta2):
    pi = math.pi
    coeff = torch.sqrt(pi / (alpha + p1))
    ea1 = torch.exp(-(4 * alpha * p1 * z1**2 + thet2**2) / (4 * (alpha + p1)))
    ea2 = torch.cos(beta2 + thet2 * (t - (p1 * z1) / (alpha + p1)))
    return coeff * ea1 * ea2


@torch.jit.script
def I_phi_k(t, alpha, thet1, beta1, p2, z2):
    pi = math.pi
    coeff = torch.sqrt(pi / (alpha + p2))
    ea = -(4 * alpha * p2 * (t - z2) ** 2 + thet1**2) / (4 * (alpha + p2))
    ca = (alpha * beta1 + p2 * (beta1 + thet1 * (t - z2))) / (alpha + p2)
    return coeff * torch.exp(ea) * torch.cos(ca)


@torch.jit.script
def I_k_k(t, alpha, p1, z1, p2, z2):
    pi = math.pi
    coeff = torch.sqrt(pi / (alpha + p1 + p2))
    ea1 = alpha * (p1 * z1**2 + p2 * (t - z2) ** 2)
    ea2 = p1 * p2 * (z1 + z2 - t) ** 2
    return coeff * torch.exp(-(ea1 + ea2) / (alpha + p1 + p2))


@torch.jit.script
def I_phi_imag(t, alpha, thet1, beta1, thet2):
    """
    Computes integal of random part of G with e^{j thet2 t}.
    """
    pi = math.pi
    const = 0.5 * torch.sqrt(pi / alpha)
    t1 = torch.exp(
        torch.complex(-((thet1 + thet2) ** 2) / (4 * alpha), t * thet2 - beta1)
    )
    t2 = 1 + torch.exp(torch.complex(thet1 * thet2 / alpha, 2 * beta1))
    return const * t1 * t2


@torch.jit.script
def I_k_imag(t, alpha, p1, z1, thet2):
    """
    Computes integal of canonical basis part of G with e^{j thet2 t}.
    """
    pi = math.pi
    const = torch.sqrt(pi / (p1 + alpha))
    t1 = thet2 * torch.complex(thet2, -4 * p1 * (t - z1))
    t2 = 4 * alpha * torch.complex(p1 * z1**2, -t * thet2)
    return const * torch.exp(-(t1 + t2) / (4 * (alpha + p1)))
