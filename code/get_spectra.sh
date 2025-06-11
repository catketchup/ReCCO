#!/bin/bash
python spectra.py -t1 vr -t2 vr -lmax 4000
python spectra.py -t1 vr -t2 taud -lmax 4000
python spectra.py -t1 vr -t2 g -lmax 4000
python spectra.py -t1 taud -t2 taud -lmax 4000
python spectra.py -t1 taud -t2 g -lmax 4000
python spectra.py -t1 g -t2 g -lmax 4000
python spectra.py -t1 pCMB -t2 pCMB -lmax 4000


# python spectra.py -t1 isw_lin -t2 g -lmax 4000
# python spectra.py -t1 isw_lin -t2 isw_lin -lmax 4000
# python spectra.py -t1 tSZ -t2 g -lmax 4000
# python spectra.py -t1 CIB -t2 g -lmax 4000
# python spectra.py -t1 tSZ -t2 tSZ -lmax 4000
# python spectra.py -t1 tSZ -t2 CIB -lmax 4000
# python spectra.py -t1 CIB -t2 CIB -lmax 4000
# python spectra.py -t1 vt -t2 vt -lmax 4000
# python spectra.py -t1 vt -t2 ml -lmax 4000
# python spectra.py -t1 vt -t2 g -lmax 4000
# python spectra.py -t1 ml -t2 ml -lmax 4000
# python spectra.py -t1 ml -t2 g -lmax 4000
# python spectra.py -t1 pCMB -t2 pCMB -lmax 4000
# python spectra.py -t1 lensing -t2 g -lmax 4000
# python spectra.py -t1 lensing -t2 lensing -lmax 4000
