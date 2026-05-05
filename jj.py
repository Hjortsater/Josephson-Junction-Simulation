import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline
from kc import KitaevChain

class JosephsonJunction:
    def __init__(self, chains, tCouple=0.1):
        self.chains = chains
        self.nc = len(chains)
        self.tCouple = tCouple

        for c in self.chains:
            if not hasattr(c, "phase"):
                c.phase = 0.0

    def uncoupledBdG(self):
        blocks = [c.buildBdGHamiltonian() for c in self.chains]
        dims = [b.shape[0] for b in blocks]
        total = sum(dims)

        h0 = np.zeros((total, total), dtype=complex)

        idx = 0
        for i, b in enumerate(blocks):
            n = dims[i]
            h0[idx:idx+n, idx:idx+n] = b
            idx += n

        return h0, dims

    def tunneling(self, dims):
        total = sum(dims)
        hj = np.zeros((total, total), dtype=complex)

        offsets = np.cumsum([0] + dims)


        for p in range(self.nc - 1):
            q = p + 1     

            phaseDiff = self.chains[q].phase - self.chains[p].phase
            t = self.tCouple * np.exp(1j * phaseDiff / 2)

            dim = dims[p]
            N = dim // 2

            p0 = offsets[p]

            q0 = offsets[q]

            i_e = p0 + (N - 1)
            j_e = q0 + 0      

            hj[i_e, j_e] += t
            hj[j_e, i_e] += np.conj(t)

            i_h = p0 + (2*N - 1)
            j_h = q0 + N         

            hj[i_h, j_h] += -np.conj(t)
            hj[j_h, i_h] += -t

        return hj

    def buildHamiltonian(self):
        h0, dims = self.uncoupledBdG()
        hj = self.tunneling(dims)
        return h0 + hj

    def groundStateEnergy(self, h):
        eigvals = np.linalg.eigvalsh(h)
        return np.sum(eigvals[eigvals < 0])

    def energy(self):
        h = self.buildHamiltonian()
        return self.groundStateEnergy(h)

    def current(self, idx, dphi=1e-5):
        original = self.chains[idx].phase

        self.chains[idx].phase = original + dphi
        ePlus = self.energy()

        self.chains[idx].phase = original - dphi
        eMinus = self.energy()

        self.chains[idx].phase = original

        deDphi = (ePlus - eMinus) / (2 * dphi)

        return (2 * 1.602e-19 / 1.055e-34) * deDphi

    def allCurrents(self):
        return np.array([self.current(i) for i in range(self.nc)])

    def andreev_spectrum(self, phiVals):
        spectra = []

        for phi in phiVals:
            self.chains[1].phase = phi
            self.chains[1].Delta = np.abs(self.chains[1].Delta) * np.exp(1j * phi)

            h = self.buildHamiltonian()
            eigvals = np.linalg.eigvalsh(h)

            spectra.append(eigvals)

        return np.array(spectra)


if __name__ == "__main__":
        
    def runAndPlot(axE, axI, titlePrefix, chains, tC, color):
        jj = JosephsonJunction(chains=chains, tCouple=tC)

        baseDelta = chains[0].Delta
        phiVals = np.linspace(0, 2 * np.pi, 200)

        spec = jj.andreev_spectrum(phiVals)
        energies = []

        for phi in phiVals:
            jj.chains[1].phase = phi
            jj.chains[1].Delta = baseDelta * np.exp(1j * phi)

            if len(chains) == 3:
                jj.chains[2].phase = 2 * phi
                jj.chains[2].Delta = baseDelta * np.exp(1j * 2 * phi)

            energies.append(jj.energy())

        energies = np.array(energies)
        energiesNorm = energies - np.min(energies)
        current = np.gradient(energies, phiVals)

        label = f"{titlePrefix}\n($t_c={tC}, N={chains[0].N}$)"


        Ecut = 0.6
        n_keep = 4

        masked_spec = []

        for j in range(len(phiVals)):
            energies_at_phi = spec[j, :]

            idx = np.argsort(np.abs(energies_at_phi))
            selected = energies_at_phi[idx[:n_keep]]
            masked_spec.append(np.sort(selected))
        masked_spec = np.array(masked_spec)

        for i in range(n_keep):
            axE.plot(
                phiVals,
                masked_spec[:, i],
                color='black',
                lw=1.6 if i < 2 else 1.0,
                alpha=0.9 if i < 2 else 0.6,
                zorder=2
            )

        axE.plot(phiVals, energiesNorm, label=label, lw=2.5, color=color, zorder=3)
        axE.axvline(x=np.pi, color='gray', linestyle=':', alpha=0.4)
        axE.set_ylabel(r"$E - E_{min}$")
        axE.grid(True, linestyle=':', alpha=0.5)

        axI.plot(phiVals, current, label=label, lw=2.5, color=color)
        axI.axvline(x=np.pi, color='gray', linestyle=':', alpha=0.4)
        axI.axhline(y=0, color='black', lw=1, alpha=0.7)
        axI.set_ylabel(r"Current $\propto \partial_\phi E$")
        axI.grid(True, linestyle=':', alpha=0.5)
    
    def formatPlotRow(axE, axI, superTitle):
        axE.set_title(superTitle + " (Energy)", fontsize=14)
        axI.set_title(superTitle + " (Current)", fontsize=14)
        
        for ax in [axE, axI]:
            ax.set_xticks([0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
            ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
            ax.set_xlabel(r"Phase Difference $\phi$")
        
        handles, labels = axE.get_legend_handles_labels()
        if labels: 
            axE.legend(handles, labels, loc='best', frameon=True)

    t = 1.0 
    delta = 0.5

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    ax2E = fig.add_subplot(gs[0, 0])
    ax2I = fig.add_subplot(gs[0, 1])
    ax3E = fig.add_subplot(gs[1, 0])
    ax3I = fig.add_subplot(gs[1, 1])
    
    print("Calculating 2-Chain Systems...")
    
    chainsTriv = [
        KitaevChain(N=60, t=t, mu=2.5, Delta=delta, phase=0.0),
        KitaevChain(N=60, t=t, mu=2.5, Delta=delta, phase=0.0)
    ]
    runAndPlot(ax2E, ax2I, r"Trivial ($|\mu| > 2t$)", chainsTriv, tC=0.2, color='#1f77b4')

    chainsTopoShort = [
        KitaevChain(N=12, t=t, mu=0.5, Delta=delta, phase=0.0),
        KitaevChain(N=12, t=t, mu=0.5, Delta=delta, phase=0.0)
    ]
    runAndPlot(ax2E, ax2I, r"Topological Short (MZMs)", chainsTopoShort, tC=0.2, color='#ff7f0e')

    chainsTopoStrong = [
        KitaevChain(N=60, t=t, mu=0.5, Delta=delta, phase=0.0),
        KitaevChain(N=60, t=t, mu=0.5, Delta=delta, phase=0.0)
    ]
    runAndPlot(ax2E, ax2I, r"Topological Long (MZMs)", chainsTopoStrong, tC=0.7, color='#2ca02c')
    
    formatPlotRow(ax2E, ax2I, "Two-Chain Junction Comparisons")

    print("Calculating 3-Chain Systems...")
    
    chains3Triv = [
        KitaevChain(N=40, t=t, mu=2.5, Delta=delta, phase=0.0),
        KitaevChain(N=40, t=t, mu=2.5, Delta=delta, phase=0.0),
        KitaevChain(N=40, t=t, mu=2.5, Delta=delta, phase=0.0)
    ]
    runAndPlot(ax3E, ax3I, r"3-Chain Trivial", chains3Triv, tC=0.3, color='#d62728')

    chains3TopoAsym = [
        KitaevChain(N=40, t=t, mu=0.5, Delta=delta, phase=0.0),
        KitaevChain(N=40, t=t, mu=0.8, Delta=0.2, phase=0.0), 
        KitaevChain(N=40, t=t, mu=0.2, Delta=0.6, phase=0.0)  
    ]
    runAndPlot(ax3E, ax3I, r"3-Chain Topological (Asym)", chains3TopoAsym, tC=0.3, color='#9467bd')

    formatPlotRow(ax3E, ax3I, "Three-Chain Stack Comparisons")

    fig.suptitle("Josephson Effects in Complex Kitaev Networks", fontsize=18, fontweight='bold', y=0.98)
    
    print("Complete.")
    plt.show()