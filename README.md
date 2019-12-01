# MCMC_NHEK1

Here is the code which builds the near-extremal gravitational waveforms using the flux data a0.98,0.998 and a0.999 999 999 
Tuekolsky data for the flux. In the code Fluxes, one needs to change paths to the location of the directory containing the flux
data above. 

The Waveform_Code.py script is the script which actually builds the waveforms. In this code, I try to compute Fisher matrices
to estimate how well we can measure certain parameters.

The other codes are my attempts to perform MCMC to estimate the parameters.

To run this code, one must do the following.

Save the flux data (a5 etc) into some directory. Then open the folder called Matt_Changes. Then open

Flux_Combo.py
Fluxes.py
Waveform_MCMC.py
MCMC_full
MCMC_run

The code Waveform_MCMC.py now generates a signal and avoids unnecessary computation. It's more efficient. The code MCMC_full.py is a code full of functions which are used during the MCMC algorithm. The code MCMC_run_spin_LISA.py actually runs the algorithm.

At the moment, the other parameters (secondary mass, primary mass, distance D and initial phase) are commented out. As such, the only parameter that is being estimated is spin. 
