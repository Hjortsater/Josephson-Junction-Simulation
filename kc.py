import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

"""Code based off of code for Kitaev Chains, converted in to instantiable object classes"""

class KitaevChain:
    def __init__(self, N=100, t=10.0, mu=5.0, Delta=0.0, phase=0.0) -> None:
        self.N = N
        self.t = t
        self.mu = mu
        self.phase = phase
        
        self.Delta = Delta * np.exp(1j * phase)

    def plotEnergyResults(self, k_values, energies) -> None:
        plt.figure(figsize=(8, 5))
        
        k_smooth = np.linspace(k_values.min(), k_values.max(), 300)
        
        styles = [
            {'ls': '-', 'marker': 'o', 'mfc': 'black', 'label': 'Band 1'},
            {'ls': '--', 'marker': 's', 'mfc': 'none', 'label': 'Band 2'}
        ]

        for i in range(energies.shape[1]):
            spline = make_interp_spline(k_values, energies[:, i], k=3)
            e_smooth = spline(k_smooth)
            
            plt.plot(k_smooth, e_smooth, color='black', linestyle=styles[i]['ls'], 
                    linewidth=1.2, zorder=1)
            
            plt.plot(k_values, energies[:, i], styles[i]['marker'], color='black', 
                    markerfacecolor=styles[i]['mfc'], markersize=4, 
                    label=styles[i]['label'], markeredgewidth=0.8, zorder=2)

        plt.axhline(0, color='black', linestyle=':', linewidth=0.8)
        plt.xlabel('Wavevector k')
        plt.ylabel('Energy E(k)')
        plt.title('Kitaev Chain BdG Dispersion')
        plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                ['0', 'π/2', 'π', '3π/2', '2π'])
        
        plt.grid(True, linestyle=':', alpha=0.4, color='gray')
        plt.legend(frameon=True, edgecolor='black', loc='best')
        plt.tight_layout()
        plt.show()

    def buildHNaught(self):
        H0 = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            H0[i, i] = -self.mu
            if i < self.N - 1:
                H0[i, i + 1] = -self.t
                H0[i + 1, i] = -self.t
        return H0

    def buildDelta(self):
        
        Delta_mat = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N - 1):
            Delta_mat[i, i + 1] = self.Delta
            Delta_mat[i + 1, i] = -self.Delta
        return Delta_mat

    def buildBdGHamiltonian(self):
            H0 = self.buildHNaught()
            Delta_mat = self.buildDelta()
            BdG = np.zeros((2 * self.N, 2 * self.N), dtype=complex)
            
            BdG[:self.N, :self.N] = H0
        
            BdG[:self.N, self.N:] = Delta_mat
        
            BdG[self.N:, :self.N] = Delta_mat.conj().T
        
            BdG[self.N:, self.N:] = -H0.conj()
            
            return BdG


    def buildKSpaceHamiltonian(self):
        k_values = np.linspace(0, 2 * np.pi, self.N)
        energies = []
        for k in k_values:
            Hk = np.array([
                [-self.mu - 2 * self.t * np.cos(k), 2j * self.Delta * np.sin(k)],
                [-2j * self.Delta * np.sin(k), self.mu + 2 * self.t * np.cos(k)]
            ])
            eigvals = np.linalg.eigvalsh(Hk)
            energies.append(eigvals)
        energies = np.array(energies)
        return k_values, energies

    def run(self):
        BdG = self.buildBdGHamiltonian()
        ks, energies = self.buildKSpaceHamiltonian()
        print(BdG)
        self.plotEnergyResults(ks, energies)


if __name__ == "__main__":
    
    kc = KitaevChain(Delta=1.0, phase=0)
    kc.run()