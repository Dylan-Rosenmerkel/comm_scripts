import pandas as pd

def ListIsotopes( material ):
    
    try:
        loi = [x for x in GetIsotopes[material] ]
        print(""" List of Isotopes for material {0}:\n""".format(material) )
        for iso in loi:
            print( iso )

    except KeyError:
        print(""" MaterialError: This material is not defined. See PrintMaterials """)


def PrintMaterials( ):

    print( """ List of Materials: """)
    for key in [k for k in GetIsotopes.keys()]:
        print( key )

