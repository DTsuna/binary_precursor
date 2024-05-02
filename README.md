# binary_precursor
This repository is a light curve model of supernova (SN) precursors, powered by a pre-SN outburst accompanying accretion onto a compact object companion. While it being only one of the possible models, it may be useful for interpretations of (bright) SN precursors highly exceeding the Eddington limit of massive stars, which are observed in a fraction of SNe with dense circumstellar matter (CSM) around the progenitor. 

Main parameters: (1) compact object mass, (2) progenitor mass, (3) progenitor radii, (4) opacity, (5) ionization temperature, (6) initial CSM velocity normalized by progenitor escape velocity (xi parameter), (7) CSM mass, and (8) binary separation. These can be specified by editing the two execution scripts below.

"prec_outburst_grid.py" calculates and outputs the characteristic quantities of the precursor (duration, luminosity and final CSM velocity), for a grid of CSM mass and binary separation. Each parameter set takes typically ~10 seconds to finish.

"prec_outburst_lc.py" outputs a time-dependent light curve for a given set of parameters.

If one aims to interpret a certain precursor, a way is to run the former "prec_outburst_grid.py" (or see our paper Tsuna+24, ApJ 966 30) to narrow down the parameters that reproduce the characteristic quantity, and then run "prec_outburst_lc.py" for a more detailed fit to the light curve.
