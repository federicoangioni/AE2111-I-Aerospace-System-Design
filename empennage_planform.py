import numpy as np
from fuselage import fuselage_length
from CG_location import X_LEMAC
from SARoptimization import optimized_b, optimized_S
from planform import MAC
# inputs
Xlemac = X_LEMAC
SW = optimized_S  # surface wing area
MAC = MAC
b = optimized_b # wingspan

# from references
VH = 0.99
VV = 0.07

Xaft = Xlemac + 0.4 * MAC
Xh = 0.9 * fuselage_length
Xv = 0.9 * fuselage_length

LH = Xh - Xaft  # distance from tail's aerodynamic center to the aircraft aft cg
LV = Xv - Xaft  # distance from vertical tail's aerodynamic center to the aircraft aft cg

SH = VH * SW * MAC / LH
SV = VV * SW * b / LV

# Tail vertical

ARvert = 1.75
taperingvert = 0.33
QCsweepvert = 40  # degrees
QCsweepvert = QCsweepvert * np.pi / 180

bvert = np.sqrt(ARvert * SV)

bmacvert = bvert / 3 * ((1 + 2*taperingvert) / (1 + taperingvert))
# xmacvert = bmacvert * np.tan()

croot_vert = 2 * SV / (bvert * (1 + taperingvert))
ctip_vert = taperingvert * croot_vert

MACvert = 2/3 * croot_vert * (1 + taperingvert + taperingvert**2) / (1 + taperingvert)
LEsweepvert = np.arctan(np.tan(QCsweepvert) + (1/4) * ((2 * croot_vert) / bvert) * (1 - taperingvert))

ACtoRootC_vert = bmacvert * np.tan(LEsweepvert) + 0.35 * MACvert
LERoot_vert = Xv - ACtoRootC_vert
TERoot_vert = LERoot_vert + croot_vert

# Tail horizontal

ARhoriz = 4
taperinghoriz = 0.4
QCsweephoriz = 30  # degrees
QCsweephoriz = QCsweephoriz * np.pi / 180

bhoriz = np.sqrt(ARhoriz * SH)

bmachoriz = bvert / 6 * ((1 + 2*taperinghoriz) / (1 + taperinghoriz))

croot_horiz = 2 * SH / (bhoriz * (1 + taperinghoriz))
ctip_horiz = taperinghoriz * croot_horiz

MAChoriz = 2/3 * croot_horiz * (1 + taperinghoriz + taperinghoriz**2) / (1 + taperinghoriz)
LEsweephoriz = np.arctan(np.tan(QCsweephoriz) + (1/4) * ((2 * croot_horiz) / bhoriz) * (1 - taperinghoriz))

ACtoRootC_horiz = bmachoriz * np.tan(LEsweephoriz) + 0.35 * MAChoriz
LERoot_horiz = Xh - ACtoRootC_horiz
TERoot_horiz = LERoot_horiz + croot_horiz

if __name__ == "__main__":
    # print(f"""Tail Areas:
    #   Horizontal tail area: {SH}
    #   Vertical tail area:  {SV}
    #   """)

    # print(f"""Vertical tail:
    #   Area : {SV}
    #   Root chord: {croot_vert}
    #   Tip chord: {ctip_vert}
    #   Span: {bvert}
    #   LE sweep: {np.degrees(LEsweepvert)}
    #   Spanwise location MAC: {bmacvert}
    #   MAC: {MACvert}
    #   xMAC: {bmacvert * np.tan(LEsweepvert)}
    #   AC to root chord: {ACtoRootC_vert}
    #   Root LE position: {LERoot_vert}
    #   Root TE position: {TERoot_vert}
    #   """)

    # print(f"""Horizontal tail:
    #   Area : {SH}
    #   Root chord: {croot_horiz}
    #   Tip chord: {ctip_horiz}
    #   Span: {bhoriz}
    #   LE sweep: {np.degrees(LEsweephoriz)}
    #   Spanwise location MAC: {bmachoriz}
    #   MAC: {MAChoriz}
    #   xMAC: {bmachoriz * np.tan(LEsweephoriz)}
    #   AC to root chord: {ACtoRootC_horiz}
    #   Root LE position: {LERoot_horiz}
    #   Root TE position: {TERoot_horiz}
    #   """)

    # print(f"fuselage length: {fuselage_length}")