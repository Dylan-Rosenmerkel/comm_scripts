### METADATA FOR RADIOGENICS

isotope_half_life = {
    "56a27z": {"t_half_life": 77.236,   "sat": 230.},
    "57a27z": {"t_half_life": 271.81,   "sat": 1800.},
    "58a27z": {"t_half_life": 70.83,    "sat": 1650.},
    "60a27z": {"t_half_life": 1923.,    "sat": 2100.},
    "54a25z": {"t_half_life": 312.13,   "sat": 215.},
    "59a26z": {"t_half_life": 44.495,   "sat": 455.},
    "46a21z": {"t_half_life": 83.788,   "sat": 53.},
    "3a1z":   {"t_half_life": 12.32*365,"sat": 1447.}
}

### METADATA FOR DECAY CHAINS: parent chain for the considered isotopes
get_parent_chain={
        "234ma91z":"238U",
        "234a91z":"238U",
        "234a90z":"238U",
        "228a89z":"232Th",
        "228a88z":"232Th",
        "212a82z":"232Th",
        "212a83z":"232Th",
        "208a81z":"232Th",
        "214a82z":"226Ra",
        "214a83z":"226Ra",
        "210a82z":"210Pb",
        "210a83z":"210Pb",
        "40a19z":"40K",
        "32a14z":"32Si",
        "32a15z":"32Si",
        "60a27z":"activation",
        "56a27z":"activation",
        "57a27z":"activation",
        "58a27z":"activation",
        "54a25z":"activation",
        "59a26z":"activation",
        "46a21z":"activation",
        "22a11z":"22Na"
        }

### METADA FOR CONTAMINATION ACTIVITIES
###   - bulk in units of decays/kg/day (mass of the detector should be in kg!)
###   - surface is missing XXX
damic_act_in_CCD={
        "bulk":{
            "238U"  :(0.53,None),
            "226Ra" :(0.43,None),
            "210Pb" :(33,None),
            "232Th" :(0.4,None),
            "40K"   :(0.04,None)
            },
        "surface"   :{
            "238U"  :(0.53,None),
            "226Ra" :(0.43,None),
            "210Pb" :(33,None),
            "232Th" :(0.4,None),
            "40K"   :(0.04,None)
            }
        }
lbc_act_in_CCD=damic_act_in_CCD
damicm_act_in_CCD=damic_act_in_CCD

damic_act_in_KC={
        "bulk":{
            "238U"  :(5000,420),
            "226Ra" :(420,490),
            "210Pb" :(420,490),
            "232Th" :(280,40),
            "40K"   :(2480,170)
            },
        "surface"   :{
            "238U"  :(5013.8,423.4),
            "226Ra" :(420,490),
            "210Pb" :(420,490),
            "232Th" :(276.5,42.0),
            "40K"   :(2475.4,172.8)
            }
        }
lbc_act_in_KC=damic_act_in_KC
damicm_act_in_KC=damic_act_in_KC

damic_act_in_ColdCu={
        "bulk":{
            "238U"  :(10.7,None),
            "226Ra" :(11.2,None),
            "210Pb" :(2350,720),
            "232Th" :(3.5,None),
            "40K"   :(2.7,None)
            },
        "surface"   :{
            "238U"  :(10.7,None),
            "226Ra" :(11.2,None),
            "210Pb" :(2350,720),
            "232Th" :(3.5,None),
            "40K"   :(2.7,None)
            }
        }
lbc_act_in_ColdCu=damic_act_in_ColdCu
damicm_act_in_ColdCu=damic_act_in_ColdCu

### Canfranc -- LNGS - Bulk
damic_act_in_EFCu={
        "bulk":{
            "238U"  :(1.037,None),  # Canfranc
            "226Ra" :(1.20, None),  # OFHC/10. (from DAMIC-100)
            "210Pb" :(45.72,None),  # 0.53 mBq/kg --> 0.53e-3*60*60*24 (Radomir)
            "232Th" :(3.508,None),  # Canfranc
            "40K"   :(2.7,None)     # the same as OFHC (from DAMIC-100)
            },
        "surface":{
            "238U"  :(1.037,None),  # Canfranc
            "226Ra" :(1.20, None),  # OFHC/10. (from DAMIC-100)
            "210Pb" :(45.72,None),  # 0.53 mBq/kg --> 0.53e-3*60*60*24 (Radomir)
            "232Th" :(3.508,None),  # Canfranc
            "40K"   :(2.7,None)     # the same as OFHC (from DAMIC-100)
            }
        }
lbc_act_in_EFCu=damic_act_in_EFCu
damicm_act_in_EFCu=damic_act_in_EFCu

damic_act_in_screws={
        "bulk":{
            "238U"  :(1400,3800),
            "226Ra" :(138,None),
            "210Pb" :(2350,720),
            "232Th" :(200,140),
            "40K"   :(2400,1300)
            },
        "surface"   :{
            "238U"  :(1400,3800),
            "226Ra" :(138,None),
            "210Pb" :(2350,720),
            "232Th" :(200,140),
            "40K"   :(2400,1300)
            }
        }
lbc_act_in_screws=damic_act_in_screws
damicm_act_in_screws=damic_act_in_screws

damic_act_in_ancientLead={
        "bulk":{
            "238U"  :(2.0,None),
            "226Ra" :(22.5,None),
            "210Pb" :(2850,285),
            "232Th" :(0.2,None),
            "40K"   :(0.5,None)
            },
        "surface"   :{
            "238U"  :(10.7,None),
            "226Ra" :(25.9,None),
            "210Pb" :(2850,285),
            "232Th" :(2.8,None),
            "40K"   :(0.5,None)
            }
        }
lbc_act_in_ancientLead=damic_act_in_ancientLead
damicm_act_in_ancientLead=damic_act_in_ancientLead

damic_act_in_LBLead={
        "bulk":{
            "238U"  :(10.7,None),
            "226Ra" :(25.9,None),
            "210Pb" :(2850,285),
            "232Th" :(2.8,None),
            "40K"   :(0.5,None)
            },
        "surface"   :{
            "238U"  :(10.7,None),
            "226Ra" :(25.9,None),
            "210Pb" :(2850,285),
            "232Th" :(2.8,None),
            "40K"   :(0.5,None)
            }
        }
lbc_act_in_LBLead=damic_act_in_LBLead
damicm_act_in_LBLead=damic_act_in_LBLead

damic_act_in_outerLead={
        "bulk":{
            "238U"  :(1.1,None),
            "226Ra" :(13,None),
            "210Pb" :(1560000,430000),
            "232Th" :(0.4,None),
            "40K"   :(19,None)
            },
        "surface"   :{
            "238U"  :(1.1,None),
            "226Ra" :(13,None),
            "210Pb" :(1560000,430000),
            "232Th" :(0.4,None),
            "40K"   :(19,None)
            }
        }
lbc_act_in_outerLead=damic_act_in_outerLead
damicm_act_in_outerLead=damic_act_in_outerLead

#### 
LBC_activity={
    "CCD"           :lbc_act_in_CCD,
    "KaptonCable"   :lbc_act_in_KC,
    "ColdCopper"    :lbc_act_in_ColdCu,
    "EFCopper"      :lbc_act_in_EFCu,
    "Screws"        :lbc_act_in_screws,
    "AncientLead"   :lbc_act_in_ancientLead,
    "lowBkgLead"    :lbc_act_in_LBLead,
    "OuterLead"     :lbc_act_in_outerLead
    }

DAMICM_activity={
    "CCD"           :damicm_act_in_CCD,
    "KaptonCable"   :damicm_act_in_KC,
    "ColdCopper"    :damicm_act_in_ColdCu,
    "EFCopper"      :damicm_act_in_EFCu,
    "Screws"        :damicm_act_in_screws,
    "AncientLead"   :damicm_act_in_ancientLead,
    "lowBkgLead"    :damicm_act_in_LBLead,
    "OuterLead"     :damicm_act_in_outerLead
    }

DAMIC_activity={
    "CCD"           :damic_act_in_CCD,
    "KaptonCable"   :damic_act_in_KC,
    "ColdCopper"    :damic_act_in_ColdCu,
    "EFCopper"      :damic_act_in_EFCu,
    "Screws"        :damic_act_in_screws,
    "AncientLead"   :damic_act_in_ancientLead,
    "lowBkgLead"    :damic_act_in_LBLead,
    "OuterLead"     :damic_act_in_outerLead
    }


get_activity = {
        "DAMIC":DAMIC_activity,
        "DAMICM":DAMICM_activity,
        "LBC":LBC_activity
        }

def get_detector_region(pvname):
    if pvname in ["CCDSensor_PV"]:
        return "CCD"
    elif pvname in ["KConccd_PV"]:
        return "KaptonCable"
    elif pvname in ['ColdCopper_2_PV','ColdCopper_4_PV','ColdCopper_5_PV','ColdCopper_6_PV','ColdCopper_1_PV']:
        return "ColdCopper"
    elif pvname in ['Cryo_1_PV','Cryo_2_PV','Cryo_3_PV','Cryo_4_PV','Cryo_5_PV','Cryo_6_PV','Cryo_7_PV','Cryo_8_PV',
            'Cryo_9_PV','Cryo_10_PV','Cryo_11_PV','Cryo_12_PV']:
        return "ColdCopper"
    elif pvname in ['ColdCopper_3_PV']:#,'ColdCopper_1_PV']:
        return "EFCopper"
    elif pvname in ["ExtShieldingLead_PV","LBLead_1_PV","LBLead_2_PV","LBLead_3_PV","LBLead_4_PV"]:
        return "lowBkgLead"
    elif pvname in ['AncientLead_1_PV','AncientLead_2_PV','AncientLead_3_PV','AncientLead_4_PV']:
        return "AncientLead"
    else:
        if pvname.lower().count("ancientlead")>0:
            return "AncientLead"
        elif pvname.lower().count("lead")>0:
            return "lowBkgLead"
        elif pvname.lower().count("copper")>0:
            return "ColdCopper"

