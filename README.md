# binary_precursor
This repository is a light curve model of supernova (SN) precursors, by SN progenitors with compact object companions. This model is useful for interpretations of (bright) SN precursors, which are observed in SNe with dense circumstellar matter (CSM) around the star. 

Main parameters: compact object mass, progenitor mass, progenitor radii, opacity, ionization temperature, initial CSM velocity (normalized by progenitor escape velocity), CSM mass, and binary separation

"prec_outburst_grid.py" calculates and outputs the characteristic properties of the precursor (duration, luminosity and final CSM velocity), for a grid of CSM mass and orbital separation (normalized by progenitor radii). Each parameter takes typically 10 seconds to finish.

"prec_outburst_lc.py" outputs a light curve for a given set of parameters.
