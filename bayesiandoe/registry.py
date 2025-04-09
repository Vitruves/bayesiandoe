from typing import Dict, List, Any, Optional, Tuple
import json
import os

class ChemicalRegistry:
    def __init__(self):
        self.registry = {
            "solvents": {
                "Polar Protic": [
                    {"name": "Water", "properties": {"bp": "100°C", "ε": "80.1", "log P": "-1.38"}},
                    {"name": "Methanol", "properties": {"bp": "64.7°C", "ε": "32.7", "log P": "-0.77"}},
                    {"name": "Ethanol", "properties": {"bp": "78.5°C", "ε": "24.5", "log P": "-0.31"}}, 
                    {"name": "n-Propanol", "properties": {"bp": "97°C", "ε": "20.1", "log P": "0.25"}},
                    {"name": "i-Propanol", "properties": {"bp": "82.6°C", "ε": "19.9", "log P": "0.05"}},
                    {"name": "n-Butanol", "properties": {"bp": "117.7°C", "ε": "17.8", "log P": "0.88"}},
                    {"name": "t-Butanol", "properties": {"bp": "82.6°C", "ε": "12.5", "log P": "0.35"}},
                    {"name": "Acetic Acid", "properties": {"bp": "118°C", "ε": "6.2", "log P": "-0.17"}},
                    {"name": "Formic Acid", "properties": {"bp": "100.8°C", "ε": "58.5", "log P": "-0.54"}},
                    {"name": "TFE", "properties": {"bp": "73.6°C", "ε": "26.1", "log P": "0.41"}},
                    {"name": "HFIP", "properties": {"bp": "58.2°C", "ε": "16.7", "log P": "1.96"}},
                    {"name": "Glycerol", "properties": {"bp": "290°C", "ε": "42.5", "log P": "-1.76"}}
                ],
                "Polar Aprotic": [
                    {"name": "Acetone", "properties": {"bp": "56°C", "ε": "20.7", "log P": "-0.24"}},
                    {"name": "Acetonitrile", "properties": {"bp": "82°C", "ε": "37.5", "log P": "-0.34"}},
                    {"name": "DMF", "properties": {"bp": "153°C", "ε": "36.7", "log P": "-1.01"}},
                    {"name": "DMSO", "properties": {"bp": "189°C", "ε": "46.7", "log P": "-1.35"}},
                    {"name": "NMP", "properties": {"bp": "202°C", "ε": "32.2", "log P": "-0.46"}},
                    {"name": "Sulfolane", "properties": {"bp": "285°C", "ε": "43.3", "log P": "-0.77"}},
                    {"name": "HMPA", "properties": {"bp": "235°C", "ε": "30.0", "log P": "0.28"}},
                    {"name": "DMAc", "properties": {"bp": "165°C", "ε": "37.8", "log P": "-0.77"}},
                    {"name": "Propylene Carbonate", "properties": {"bp": "242°C", "ε": "64.9", "log P": "-0.41"}},
                    {"name": "Nitromethane", "properties": {"bp": "101°C", "ε": "35.9", "log P": "-0.33"}},
                    {"name": "TMU", "properties": {"bp": "177°C", "ε": "23.1", "log P": "-0.14"}},
                    {"name": "DMPU", "properties": {"bp": "246°C", "ε": "36.1", "log P": "-0.31"}}
                ],
                "Non-Polar": [
                    {"name": "Toluene", "properties": {"bp": "111°C", "ε": "2.4", "log P": "2.73"}},
                    {"name": "Benzene", "properties": {"bp": "80°C", "ε": "2.3", "log P": "2.13"}},
                    {"name": "Hexane", "properties": {"bp": "69°C", "ε": "1.9", "log P": "3.76"}},
                    {"name": "Heptane", "properties": {"bp": "98°C", "ε": "1.9", "log P": "4.27"}},
                    {"name": "Cyclohexane", "properties": {"bp": "81°C", "ε": "2.0", "log P": "3.44"}},
                    {"name": "Pentane", "properties": {"bp": "36°C", "ε": "1.8", "log P": "3.45"}},
                    {"name": "Diethyl Ether", "properties": {"bp": "34.6°C", "ε": "4.3", "log P": "0.89"}},
                    {"name": "Chloroform", "properties": {"bp": "61°C", "ε": "4.8", "log P": "1.97"}},
                    {"name": "DCM", "properties": {"bp": "40°C", "ε": "9.1", "log P": "1.25"}},
                    {"name": "THF", "properties": {"bp": "66°C", "ε": "7.6", "log P": "0.46"}},
                    {"name": "Dioxane", "properties": {"bp": "101°C", "ε": "2.3", "log P": "-0.27"}},
                    {"name": "Ethyl Acetate", "properties": {"bp": "77°C", "ε": "6.0", "log P": "0.73"}},
                    {"name": "MTBE", "properties": {"bp": "55°C", "ε": "2.6", "log P": "0.94"}},
                    {"name": "Xylene", "properties": {"bp": "138-144°C", "ε": "2.3", "log P": "3.15"}},
                    {"name": "Cyclopentyl methyl ether", "properties": {"bp": "106°C", "ε": "4.8", "log P": "1.59"}}
                ],
                "Green Solvents": [
                    {"name": "2-MeTHF", "properties": {"bp": "80°C", "ε": "6.2", "log P": "0.53", "color": "gray"}},
                    {"name": "Cyrene", "properties": {"bp": "203°C", "ε": "10.6", "log P": "-0.13", "color": "gray"}},
                    {"name": "PEG-400", "properties": {"bp": ">200°C", "ε": "13.6", "renewability": "medium", "color": "gray"}},
                    {"name": "Ethyl lactate", "properties": {"bp": "154°C", "ε": "15.4", "log P": "0.19", "color": "gray"}},
                    {"name": "Ethyl acetate", "properties": {"bp": "77°C", "ε": "6.0", "log P": "0.73", "color": "gray"}},
                    {"name": "Limonene", "properties": {"bp": "176°C", "ε": "2.3", "renewability": "high", "color": "gray"}},
                    {"name": "Gamma-valerolactone", "properties": {"bp": "207°C", "ε": "36.5", "log P": "0.08", "color": "gray"}},
                    {"name": "Glycerol carbonate", "properties": {"bp": "110-115°C", "ε": "109.7", "renewability": "high", "color": "gray"}},
                    {"name": "Dimethyl carbonate", "properties": {"bp": "90°C", "ε": "3.1", "log P": "0.15", "color": "gray"}},
                    {"name": "Anisole", "properties": {"bp": "154°C", "ε": "4.3", "renewability": "moderate", "color": "gray"}},
                    {"name": "Solketal", "properties": {"bp": "188°C", "ε": "12.2", "renewability": "high", "color": "gray"}},
                    {"name": "Tetrahydrofurfuryl alcohol", "properties": {"bp": "178°C", "ε": "19.6", "log P": "-0.89", "color": "gray"}}
                ]
            },
            "catalysts": {
                "Palladium": [
                    {"name": "Pd(OAc)2", "properties": {"loading": "1-5 mol%", "activation": "easy", "cost": "medium", "color": "gray"}},
                    {"name": "Pd(PPh3)4", "properties": {"loading": "0.5-5 mol%", "stability": "air-sensitive", "activation": "none", "color": "gray"}},
                    {"name": "Pd2(dba)3", "properties": {"loading": "1-5 mol%", "stability": "moderate", "activation": "needs ligand", "color": "gray"}},
                    {"name": "Pd/C", "properties": {"loading": "5-10 wt%", "type": "heterogeneous", "filtration": "easy", "color": "gray"}},
                    {"name": "PdCl2", "properties": {"loading": "1-10 mol%", "activation": "needs ligand", "cost": "low", "color": "gray"}},
                    {"name": "Pd(dppf)Cl2", "properties": {"loading": "1-5 mol%", "precatalyst": "yes", "stability": "high", "color": "gray"}},
                    {"name": "Pd(OAc)2/PPh3", "properties": {"loading": "1-5 mol%", "L:M": "2:1 to 4:1", "stability": "moderate", "color": "gray"}},
                    {"name": "Pd(TFA)2", "properties": {"loading": "1-5 mol%", "stability": "moderate", "activation": "fast", "color": "gray"}},
                    {"name": "PEPPSI-IPr", "properties": {"loading": "0.5-2 mol%", "precatalyst": "yes", "stability": "high", "color": "gray"}},
                    {"name": "PdCl2(MeCN)2", "properties": {"loading": "1-5 mol%", "activation": "moderate", "needs ligand": "yes", "color": "gray"}},
                    {"name": "Pd(PPh3)2Cl2", "properties": {"loading": "1-5 mol%", "stability": "high", "precatalyst": "yes", "color": "gray"}}
                ],
                "Nickel": [
                    {"name": "Ni(COD)2", "properties": {"loading": "5-10 mol%", "stability": "air-sensitive", "glove box": "yes", "color": "gray"}},
                    {"name": "NiCl2•DME", "properties": {"loading": "5-10 mol%", "activation": "needs reductant", "cost": "low", "color": "gray"}},
                    {"name": "NiCl2•6H2O", "properties": {"loading": "5-10 mol%", "activation": "needs reductant", "water stable": "yes", "color": "gray"}},
                    {"name": "Ni(acac)2", "properties": {"loading": "5-10 mol%", "activation": "moderate", "cost": "moderate", "color": "gray"}},
                    {"name": "NiBr2", "properties": {"loading": "5-10 mol%", "activation": "needs reductant", "hygroscopic": "yes", "color": "gray"}},
                    {"name": "Ni(dppp)Cl2", "properties": {"loading": "1-5 mol%", "precatalyst": "yes", "stability": "high", "color": "gray"}},
                    {"name": "NiCl2(PCy3)2", "properties": {"loading": "1-5 mol%", "precatalyst": "yes", "air stability": "moderate", "color": "gray"}}
                ],
                "Copper": [
                    {"name": "CuI", "properties": {"loading": "5-20 mol%", "type": "Cu(I)", "light sensitive": "yes", "color": "gray"}},
                    {"name": "CuCl", "properties": {"loading": "5-20 mol%", "type": "Cu(I)", "stability": "moderate", "color": "gray"}},
                    {"name": "CuBr", "properties": {"loading": "5-20 mol%", "type": "Cu(I)", "stability": "moderate", "color": "gray"}},
                    {"name": "Cu(OAc)2", "properties": {"loading": "5-20 mol%", "type": "Cu(II)", "stability": "high", "color": "gray"}},
                    {"name": "CuSO4", "properties": {"loading": "5-20 mol%", "type": "Cu(II)", "water soluble": "yes", "color": "gray"}},
                    {"name": "Cu(OTf)2", "properties": {"loading": "1-10 mol%", "type": "Cu(II)", "Lewis acid": "strong", "color": "gray"}},
                    {"name": "Cu2O", "properties": {"loading": "5-10 mol%", "type": "Cu(I)", "heterogeneous": "yes", "color": "gray"}},
                    {"name": "Cu powder", "properties": {"loading": "1-2 equiv", "type": "Cu(0)", "surface area": "important", "color": "gray"}},
                    {"name": "Cu(acac)2", "properties": {"loading": "5-10 mol%", "type": "Cu(II)", "solubility": "organic", "color": "gray"}}
                ],
                "Other Metals": [
                    {"name": "FeCl3", "properties": {"loading": "5-20 mol%", "type": "Fe(III)", "Lewis acid": "strong", "color": "gray"}},
                    {"name": "ZnCl2", "properties": {"loading": "10-20 mol%", "type": "Zn(II)", "Lewis acid": "moderate", "color": "gray"}},
                    {"name": "AuCl3", "properties": {"loading": "1-5 mol%", "type": "Au(III)", "pi-acid": "strong", "color": "gray"}},
                    {"name": "Ru(bpy)3Cl2", "properties": {"loading": "1-5 mol%", "photocatalyst": "yes", "redox": "versatile", "color": "gray"}},
                    {"name": "Ir(ppy)3", "properties": {"loading": "0.1-2 mol%", "photocatalyst": "yes", "excited state": "long", "color": "gray"}},
                    {"name": "AlCl3", "properties": {"loading": "1-2 equiv", "Lewis acid": "strong", "moisture sensitive": "high", "color": "gray"}},
                    {"name": "TiCl4", "properties": {"loading": "1-2 equiv", "Lewis acid": "strong", "handling": "difficult", "color": "gray"}},
                    {"name": "RhCl3", "properties": {"loading": "1-5 mol%", "stability": "high", "cost": "high", "color": "gray"}},
                    {"name": "Sc(OTf)3", "properties": {"loading": "5-20 mol%", "Lewis acid": "strong", "water stable": "yes", "color": "gray"}},
                    {"name": "In(OTf)3", "properties": {"loading": "5-20 mol%", "Lewis acid": "moderate", "water stable": "yes", "color": "gray"}},
                    {"name": "Fe(acac)3", "properties": {"loading": "5-10 mol%", "type": "Fe(III)", "cost": "low", "color": "gray"}}
                ],
                "Organocatalysts": [
                    {"name": "L-Proline", "properties": {"loading": "10-30 mol%", "type": "amino acid", "mode": "enamine/iminium", "color": "gray"}},
                    {"name": "MacMillan", "properties": {"loading": "10-20 mol%", "type": "imidazolidinone", "mode": "iminium", "color": "gray"}},
                    {"name": "BINOL", "properties": {"loading": "5-20 mol%", "type": "diol", "mode": "hydrogen-bonding", "color": "gray"}},
                    {"name": "Quinine", "properties": {"loading": "5-10 mol%", "type": "cinchona", "mode": "base/H-bond", "color": "gray"}},
                    {"name": "Thiourea", "properties": {"loading": "5-10 mol%", "mode": "hydrogen-bonding", "dual activation": "yes", "color": "gray"}},
                    {"name": "DABCO", "properties": {"loading": "5-20 mol%", "type": "diamine", "mode": "nucleophilic", "color": "gray"}},
                    {"name": "Proline-tetrazole", "properties": {"loading": "5-15 mol%", "type": "amino acid", "mode": "enamine", "color": "gray"}},
                    {"name": "Cinchonidine", "properties": {"loading": "5-10 mol%", "type": "cinchona", "mode": "base/H-bond", "color": "gray"}},
                    {"name": "Squaramide", "properties": {"loading": "2-10 mol%", "mode": "hydrogen-bonding", "dual activation": "yes", "color": "gray"}},
                    {"name": "DMAP", "properties": {"loading": "1-10 mol%", "type": "pyridine", "mode": "nucleophilic", "color": "gray"}},
                    {"name": "Jacobsen Thiourea", "properties": {"loading": "1-5 mol%", "type": "thiourea", "chiral": "yes", "color": "gray"}},
                    {"name": "Takemoto Catalyst", "properties": {"loading": "1-10 mol%", "type": "bifunctional", "mode": "H-bond/base", "color": "gray"}},
                    {"name": "Maruoka Catalyst", "properties": {"loading": "1-5 mol%", "type": "BINOL-derived", "mode": "phase-transfer", "color": "gray"}},
                    {"name": "D-Proline", "properties": {"loading": "10-30 mol%", "type": "amino acid", "mode": "enamine/iminium", "color": "gray"}},
                    {"name": "Hayashi-Jørgensen", "properties": {"loading": "5-20 mol%", "type": "prolinol", "mode": "enamine", "color": "gray"}},
                    {"name": "Cinchonine", "properties": {"loading": "5-10 mol%", "type": "cinchona", "mode": "base/H-bond", "color": "gray"}},
                    {"name": "Quinidine", "properties": {"loading": "5-10 mol%", "type": "cinchona", "mode": "base/H-bond", "color": "gray"}},
                    {"name": "Prolinol", "properties": {"loading": "5-20 mol%", "type": "amino alcohol", "mode": "enamine", "color": "gray"}},
                    {"name": "TBD", "properties": {"loading": "1-10 mol%", "type": "guanidine", "mode": "base", "color": "gray"}},
                    {"name": "DBU", "properties": {"loading": "1-20 mol%", "type": "amidine", "mode": "base", "color": "gray"}}
                ]
            },
            "ligands": {
                "Phosphines": [
                    {"name": "PPh3", "properties": {"bite angle": "n/a", "donor": "P", "cone angle": "145°", "color": "gray"}},
                    {"name": "BINAP", "properties": {"bite angle": "92°", "chiral": "yes", "rigidity": "high", "color": "gray"}},
                    {"name": "Xantphos", "properties": {"bite angle": "108°", "flexibility": "rigid", "electronics": "moderate", "color": "gray"}},
                    {"name": "dppf", "properties": {"bite angle": "99°", "flexibility": "moderate", "electronics": "moderate", "color": "gray"}},
                    {"name": "XPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "rich", "color": "gray"}},
                    {"name": "SPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "rich", "color": "gray"}},
                    {"name": "PCy3", "properties": {"bite angle": "n/a", "electron-rich": "yes", "cone angle": "170°", "color": "gray"}},
                    {"name": "dppe", "properties": {"bite angle": "85°", "flexibility": "rigid", "chelation": "strong", "color": "gray"}},
                    {"name": "dppp", "properties": {"bite angle": "91°", "flexibility": "moderate", "chelation": "strong", "color": "gray"}},
                    {"name": "JohnPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "rich", "color": "gray"}},
                    {"name": "DavePhos", "properties": {"bite angle": "n/a", "steric": "moderate", "electronics": "rich", "color": "gray"}},
                    {"name": "CyJohnPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "rich", "color": "gray"}},
                    {"name": "tBuXPhos", "properties": {"bite angle": "n/a", "steric": "very bulky", "electronics": "rich", "color": "gray"}},
                    {"name": "BrettPhos", "properties": {"bite angle": "n/a", "steric": "very bulky", "electronics": "rich", "color": "gray"}},
                    {"name": "RuPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "moderate", "color": "gray"}}
                ],
                "Nitrogen-based": [
                    {"name": "BiPy", "properties": {"bite angle": "78°", "rigidity": "rigid", "pi-acceptor": "good", "color": "gray"}},
                    {"name": "Phen", "properties": {"bite angle": "82°", "rigidity": "rigid", "pi-acceptor": "good", "color": "gray"}},
                    {"name": "TMEDA", "properties": {"bite angle": "85°", "flexibility": "high", "basicity": "high", "color": "gray"}},
                    {"name": "L-Proline", "properties": {"type": "organocatalyst", "chirality": "L", "mode": "enamine/iminium", "color": "gray"}},
                    {"name": "Pyridine", "properties": {"denticity": "monodentate", "pi-acceptor": "good", "basicity": "moderate", "color": "gray"}},
                    {"name": "DABCO", "properties": {"denticity": "bidentate", "basicity": "high", "nucleophilicity": "high", "color": "gray"}},
                    {"name": "DMAP", "properties": {"denticity": "monodentate", "nucleophilicity": "high", "basicity": "high", "color": "gray"}},
                    {"name": "DBU", "properties": {"basicity": "very high", "nucleophilicity": "low", "type": "amidine", "color": "gray"}},
                    {"name": "Quinuclidine", "properties": {"basicity": "high", "nucleophilicity": "high", "structure": "rigid", "color": "gray"}}
                ],
                "N-Heterocyclic Carbenes": [
                    {"name": "IMes", "properties": {"type": "imidazolium", "steric": "moderate", "electronics": "strongly donating", "color": "gray"}},
                    {"name": "IPr", "properties": {"type": "imidazolium", "steric": "bulky", "electronics": "strongly donating", "color": "gray"}},
                    {"name": "SIMes", "properties": {"type": "imidazolinium", "steric": "moderate", "electronics": "very donating", "color": "gray"}},
                    {"name": "SIPr", "properties": {"type": "imidazolinium", "steric": "bulky", "electronics": "very donating", "color": "gray"}},
                    {"name": "ICy", "properties": {"type": "imidazolium", "steric": "moderate", "electronics": "donating", "color": "gray"}},
                    {"name": "ItBu", "properties": {"type": "imidazolium", "steric": "very bulky", "electronics": "donating", "color": "gray"}},
                    {"name": "IAd", "properties": {"type": "imidazolium", "steric": "extremely bulky", "electronics": "donating", "color": "gray"}},
                    {"name": "IBioxMe4", "properties": {"type": "benzimidazolium", "steric": "moderate", "rigidity": "high", "color": "gray"}},
                    {"name": "IPrCl", "properties": {"type": "imidazolium", "steric": "bulky", "electronics": "less donating", "color": "gray"}},
                    {"name": "IPent", "properties": {"type": "imidazolium", "steric": "very bulky", "electronics": "strongly donating", "color": "gray"}},
                    {"name": "IiPr", "properties": {"type": "imidazolium", "steric": "moderate", "electronics": "donating", "color": "gray"}},
                    {"name": "CAAC", "properties": {"type": "cyclic alkylaminocarbene", "steric": "very bulky", "electronics": "extremely donating", "color": "gray"}}
                ],
                "P,N-Ligands": [
                    {"name": "PHOX", "properties": {"type": "P,N-ligand", "chiral": "yes", "hemilabile": "yes"}},
                    {"name": "PyPhos", "properties": {"type": "P,N-ligand", "hemilabile": "yes", "flexibility": "moderate"}},
                    {"name": "QUINAP", "properties": {"type": "P,N-ligand", "chiral": "yes", "atropisomeric": "yes"}}
                ]
            },
            "bases": {
                "Inorganic": [
                    {"name": "K2CO3", "properties": {"pKa": "10.3", "solubility(H2O)": "112 g/L", "solubility(organic)": "low", "color": "gray"}},
                    {"name": "Cs2CO3", "properties": {"pKa": "10.3", "solubility(H2O)": "261 g/L", "solubility(organic)": "moderate", "color": "gray"}},
                    {"name": "K3PO4", "properties": {"pKa": "12.3", "solubility(H2O)": "90 g/L", "solubility(organic)": "low", "color": "gray"}},
                    {"name": "NaH", "properties": {"pKa": "35", "reactivity": "high", "handling": "pyrophoric", "color": "gray"}},
                    {"name": "NaOH", "properties": {"pKa": "15.7", "solubility(H2O)": "1090 g/L", "hygroscopic": "yes", "color": "gray"}},
                    {"name": "KOH", "properties": {"pKa": "15.7", "solubility(H2O)": "1120 g/L", "hygroscopic": "yes", "color": "gray"}},
                    {"name": "NaHCO3", "properties": {"pKa": "6.4", "solubility(H2O)": "96 g/L", "mildness": "high", "color": "gray"}},
                    {"name": "Na2CO3", "properties": {"pKa": "10.3", "solubility(H2O)": "215 g/L", "solubility(organic)": "low", "color": "gray"}},
                    {"name": "KF", "properties": {"pKa": "10.8", "solubility(H2O)": "923 g/L", "nucleophilicity": "high", "color": "gray"}},
                    {"name": "CsF", "properties": {"pKa": "10.8", "solubility(H2O)": "367 g/L", "solubility(organic)": "moderate", "color": "gray"}},
                    {"name": "LiOH", "properties": {"pKa": "15.7", "solubility(H2O)": "128 g/L", "coordination": "strong", "color": "gray"}},
                    {"name": "Na2HPO4", "properties": {"pKa": "7.2", "solubility(H2O)": "77 g/L", "mildness": "high", "color": "gray"}},
                    {"name": "Li2CO3", "properties": {"pKa": "10.3", "solubility(H2O)": "13 g/L", "coordination": "strong", "color": "gray"}}
                ],
                "Organic": [
                    {"name": "Et3N", "properties": {"pKa": "10.8", "bp": "89°C", "nucleophilicity": "moderate", "color": "gray"}},
                    {"name": "DIPEA", "properties": {"pKa": "11.4", "bp": "127°C", "nucleophilicity": "low", "color": "gray"}},
                    {"name": "DBU", "properties": {"pKa": "13.5", "bp": "180°C", "nucleophilicity": "low", "color": "gray"}},
                    {"name": "LDA", "properties": {"pKa": "36", "reactivity": "very high", "nucleophilicity": "high", "color": "gray"}},
                    {"name": "NaOt-Bu", "properties": {"pKa": "18", "reactivity": "high", "solubility(THF)": "good", "color": "gray"}},
                    {"name": "KOt-Bu", "properties": {"pKa": "18", "reactivity": "high", "solubility(THF)": "good", "color": "gray"}},
                    {"name": "Pyridine", "properties": {"pKa": "5.2", "bp": "115°C", "nucleophilicity": "moderate", "color": "gray"}},
                    {"name": "DMAP", "properties": {"pKa": "9.7", "nucleophilicity": "high", "catalytic": "yes", "color": "gray"}},
                    {"name": "TMG", "properties": {"pKa": "13.6", "bp": "160°C", "nucleophilicity": "low", "color": "gray"}},
                    {"name": "MTBD", "properties": {"pKa": "15.0", "bp": "241°C", "nucleophilicity": "low", "color": "gray"}},
                    {"name": "Proton Sponge", "properties": {"pKa": "12.1", "nucleophilicity": "very low", "steric": "hindered", "color": "gray"}},
                    {"name": "KHMDS", "properties": {"pKa": "26", "reactivity": "high", "selectivity": "good", "color": "gray"}},
                    {"name": "LiHMDS", "properties": {"pKa": "26", "reactivity": "moderate", "selectivity": "excellent", "color": "gray"}}
                ]
            },
            "additives": {
                "Lewis Acids": [
                    {"name": "BF3•Et2O", "properties": {"acidity": "strong", "handling": "moisture-sensitive", "volatility": "high", "color": "gray"}},
                    {"name": "AlCl3", "properties": {"acidity": "strong", "handling": "moisture-sensitive", "reactivity": "high", "color": "gray"}},
                    {"name": "ZnCl2", "properties": {"acidity": "moderate", "handling": "hygroscopic", "compatibility": "wide", "color": "gray"}},
                    {"name": "MgCl2", "properties": {"acidity": "mild", "handling": "hygroscopic", "solubility": "limited", "color": "gray"}},
                    {"name": "TiCl4", "properties": {"acidity": "strong", "handling": "corrosive", "volatility": "moderate", "color": "gray"}},
                    {"name": "FeCl3", "properties": {"acidity": "strong", "handling": "hygroscopic", "oxidant": "yes", "color": "gray"}},
                    {"name": "Sc(OTf)3", "properties": {"acidity": "strong", "stability": "water-stable", "tolerance": "high", "color": "gray"}},
                    {"name": "Cu(OTf)2", "properties": {"acidity": "moderate", "stability": "water-stable", "pi-acid": "yes", "color": "gray"}},
                    {"name": "YbCl3", "properties": {"acidity": "moderate", "stability": "good", "coordination": "selective", "color": "gray"}},
                    {"name": "Zn(OTf)2", "properties": {"acidity": "moderate", "stability": "water-stable", "tolerance": "high", "color": "gray"}}
                ],
                "Acids/Bases": [
                    {"name": "TFA", "properties": {"pKa": "0.23", "bp": "72°C", "volatility": "high", "color": "gray"}},
                    {"name": "MsOH", "properties": {"pKa": "-1.9", "handling": "corrosive", "strength": "high", "color": "gray"}},
                    {"name": "CSA", "properties": {"pKa": "1.2", "type": "chiral acid", "solubility": "moderate", "color": "gray"}},
                    {"name": "Acetic Acid", "properties": {"pKa": "4.76", "bp": "118°C", "mildness": "high", "color": "gray"}},
                    {"name": "Formic Acid", "properties": {"pKa": "3.77", "bp": "101°C", "reducing agent": "yes", "color": "gray"}},
                    {"name": "DMAP", "properties": {"pKa": "9.7", "nucleophilicity": "high", "catalytic": "yes", "color": "gray"}},
                    {"name": "DABCO", "properties": {"pKa": "8.8", "nucleophilicity": "high", "basicity": "moderate", "color": "gray"}},
                    {"name": "p-TsOH", "properties": {"pKa": "-2.8", "handling": "hygroscopic", "solubility": "good", "color": "gray"}},
                    {"name": "HCl", "properties": {"pKa": "-6.3", "handling": "corrosive", "volatility": "high", "color": "gray"}},
                    {"name": "H2SO4", "properties": {"pKa": "-3.0", "handling": "corrosive", "dehydrating": "yes", "color": "gray"}}
                ],
                "Oxidants/Reductants": [
                    {"name": "H2O2", "properties": {"type": "oxidant", "handling": "30% aq. solution", "mildness": "moderate", "color": "gray"}},
                    {"name": "TBHP", "properties": {"type": "oxidant", "stability": "moderate", "solubility": "good", "color": "gray"}},
                    {"name": "mCPBA", "properties": {"type": "oxidant", "purity": "70-77%", "selectivity": "good", "color": "gray"}},
                    {"name": "NaBH4", "properties": {"type": "reductant", "handling": "moisture-sensitive", "mildness": "high", "color": "gray"}},
                    {"name": "DIBAL-H", "properties": {"type": "reductant", "handling": "pyrophoric", "selectivity": "good", "color": "gray"}},
                    {"name": "LiAlH4", "properties": {"type": "reductant", "handling": "pyrophoric", "strength": "high", "color": "gray"}},
                    {"name": "Oxone", "properties": {"type": "oxidant", "handling": "hygroscopic", "water soluble": "yes", "color": "gray"}},
                    {"name": "DDQ", "properties": {"type": "oxidant", "solubility": "low", "pi-bond": "selective", "color": "gray"}},
                    {"name": "NBS", "properties": {"type": "oxidant", "handling": "light-sensitive", "brominating": "yes", "color": "gray"}},
                    {"name": "Dess-Martin", "properties": {"type": "oxidant", "selectivity": "high", "mildness": "high", "color": "gray"}},
                    {"name": "IBX", "properties": {"type": "oxidant", "solubility": "limited", "stability": "explosive", "color": "gray"}}
                ],
                "Phase-Transfer": [
                    {"name": "TBAI", "properties": {"type": "phase-transfer catalyst", "solubility": "organic", "iodide": "yes", "color": "gray"}},
                    {"name": "NaI", "properties": {"type": "nucleophile", "solubility(acetone)": "high", "oxidizable": "yes", "color": "gray"}},
                    {"name": "LiCl", "properties": {"type": "additive", "solubility(THF)": "moderate", "coordination": "strong", "color": "gray"}},
                    {"name": "TBAB", "properties": {"type": "phase-transfer catalyst", "solubility": "wide", "bromide": "yes", "color": "gray"}},
                    {"name": "Aliquat 336", "properties": {"type": "phase-transfer catalyst", "stability": "high", "water-organic": "good", "color": "gray"}}
                ],
                "Other": [
                    {"name": "MgSO4", "properties": {"type": "drying agent", "capacity": "high", "neutral": "yes", "color": "gray"}},
                    {"name": "Na2SO4", "properties": {"type": "drying agent", "capacity": "moderate", "neutral": "yes", "color": "gray"}},
                    {"name": "TMEDA", "properties": {"type": "ligand", "bp": "121°C", "chelating": "yes", "color": "gray"}},
                    {"name": "4Å MS", "properties": {"type": "drying agent", "format": "beads/powder", "capacity": "high", "color": "gray"}},
                    {"name": "3Å MS", "properties": {"type": "drying agent", "format": "beads/powder", "capacity": "moderate", "color": "gray"}},
                    {"name": "BHT", "properties": {"type": "radical inhibitor", "mp": "70°C", "solubility": "organic", "color": "gray"}},
                    {"name": "BQ", "properties": {"type": "oxidant", "mp": "115°C", "reoxidant": "yes", "color": "gray"}},
                    {"name": "AIBN", "properties": {"type": "radical initiator", "activation": "heat/light", "half-life(80°C)": "1h", "color": "gray"}},
                    {"name": "CuSO4", "properties": {"type": "click chemistry", "oxidation state": "+2", "water soluble": "yes", "color": "gray"}},
                    {"name": "Ascorbic acid", "properties": {"type": "reductant", "biocompatible": "yes", "water soluble": "yes", "color": "gray"}},
                    {"name": "TEMPO", "properties": {"type": "radical scavenger", "mp": "38-41°C", "redox active": "yes", "color": "gray"}},
                    {"name": "TBAI", "properties": {"type": "nucleophile", "mp": "147-150°C", "iodide source": "yes", "color": "gray"}},
                    {"name": "CaH2", "properties": {"type": "drying agent", "reactivity": "high", "solvent drying": "excellent", "color": "gray"}},
                    {"name": "Celite", "properties": {"type": "filtration aid", "particle size": "fine", "inert": "yes", "color": "gray"}},
                    {"name": "DTBP", "properties": {"type": "radical initiator", "bp": "111°C", "half-life(130°C)": "1h", "color": "gray"}},
                    {"name": "Silica gel", "properties": {"type": "drying/chromatography", "capacity": "high", "acidic sites": "yes", "color": "gray"}}
                ]
            }
        }
        
        self.registry_file = self._get_registry_file_path()
        self.initialize_registry()
        
    def _get_registry_file_path(self) -> str:
        app_data_dir = os.path.expanduser("~/.bayesiandoe")
        if not os.path.exists(app_data_dir):
            os.makedirs(app_data_dir)
        return os.path.join(app_data_dir, "chemical_registry.json")
    
    def _load_registry(self) -> None:
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    loaded_registry = json.load(f)
                    self.registry.update(loaded_registry)
        except Exception as e:
            print(f"Error loading registry: {e}")
    
    def _save_registry(self) -> None:
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def get_categories(self, reg_type: str) -> List[str]:
        if reg_type in self.registry:
            return list(self.registry[reg_type].keys())
        return []
    
    def get_items(self, reg_type: str, category: str) -> List[Dict[str, Any]]:
        if reg_type in self.registry and category in self.registry[reg_type]:
            return self.registry[reg_type][category]
        return []
    
    def get_item_names(self, reg_type: str, category: str) -> List[str]:
        items = self.get_items(reg_type, category)
        return [item["name"] for item in items]
    
    def get_item_properties(self, reg_type: str, category: str, item_name: str) -> Dict[str, Any]:
        items = self.get_items(reg_type, category)
        for item in items:
            if item["name"] == item_name:
                return item.get("properties", {})
        return {}
    
    def get_standardized_properties(self, reg_type: str, category: str, item_name: str) -> Dict[str, Any]:
        """
        Get properties for an item with standardized formatting.
        Ensures all expected properties are present and consistently formatted.
        """
        properties = self.get_item_properties(reg_type, category, item_name)
        
        # Ensure color property is present
        if "color" not in properties:
            properties["color"] = "gray"
            
        # Add any other required standard properties based on reg_type
        if reg_type == "catalysts":
            if "loading" not in properties:
                properties["loading"] = "1-10 mol%"
                
        elif reg_type == "solvents":
            if "bp" not in properties:
                properties["bp"] = "N/A"
            if "ε" not in properties:
                properties["ε"] = "N/A"
        
        # Return the standardized properties
        return properties
    
    def add_item(self, reg_type: str, category: str, item_name: str, properties: Dict[str, str] = None) -> bool:
        if not reg_type or not category or not item_name:
            return False
            
        if reg_type not in self.registry:
            self.registry[reg_type] = {}
            
        if category not in self.registry[reg_type]:
            self.registry[reg_type][category] = []
        
        # Check if item already exists
        for item in self.registry[reg_type][category]:
            if item["name"] == item_name:
                return False
        
        # Initialize properties with defaults if not provided
        if properties is None:
            properties = {}
            
        # Ensure color property is set
        if "color" not in properties:
            properties["color"] = "gray"
            
        # Add type-specific default properties
        if reg_type == "catalysts" and "loading" not in properties:
            properties["loading"] = "1-10 mol%"
        elif reg_type == "solvents":
            if "bp" not in properties:
                properties["bp"] = "N/A"
            if "ε" not in properties:
                properties["ε"] = "N/A"
        
        # Add new item with properties
        new_item = {"name": item_name, "properties": properties}
        self.registry[reg_type][category].append(new_item)
        self._save_registry()
        return True
    
    def remove_item(self, reg_type: str, category: str, item_name: str) -> bool:
        if reg_type not in self.registry or category not in self.registry[reg_type]:
            return False
        
        for i, item in enumerate(self.registry[reg_type][category]):
            if item["name"] == item_name:
                del self.registry[reg_type][category][i]
                self._save_registry()
                return True
        
        return False
    
    def update_item_properties(self, reg_type: str, category: str, item_name: str, 
                               properties: Dict[str, str]) -> bool:
        if reg_type not in self.registry or category not in self.registry[reg_type]:
            return False
        
        for item in self.registry[reg_type][category]:
            if item["name"] == item_name:
                # Preserve color property if it's not in the new properties
                if "color" not in properties and "properties" in item and "color" in item["properties"]:
                    properties["color"] = item["properties"]["color"]
                elif "color" not in properties:
                    properties["color"] = "gray"
                    
                item["properties"] = properties
                self._save_registry()
                return True
        
        return False
    
    def add_category(self, reg_type: str, category: str) -> bool:
        if not reg_type or not category:
            return False
            
        if reg_type not in self.registry:
            self.registry[reg_type] = {}
            
        if category not in self.registry[reg_type]:
            self.registry[reg_type][category] = []
            self._save_registry()
            return True
            
        return False
    
    def get_all_types(self) -> List[str]:
        return list(self.registry.keys())
    
    def get_full_registry(self) -> Dict:
        return self.registry 
    
    def add_registry_type(self, reg_type: str) -> bool:
        """Add a new registry type."""
        if not reg_type or reg_type in self.registry:
            return False
        
        self.registry[reg_type] = {}
        self._save_registry()
        return True
    
    def rename_item(self, reg_type: str, category: str, old_item_name: str, new_item_name: str) -> bool:
        """Rename an item in the registry."""
        if reg_type not in self.registry or category not in self.registry[reg_type]:
            return False
        
        for item in self.registry[reg_type][category]:
            if item["name"] == old_item_name:
                item["name"] = new_item_name
                self._save_registry()
                return True
        
        return False
    
    def rename_category(self, reg_type: str, old_category: str, new_category: str) -> bool:
        """Rename a category in the registry."""
        if (reg_type in self.registry and 
            old_category in self.registry[reg_type] and
            new_category not in self.registry[reg_type]):
            
            self.registry[reg_type][new_category] = self.registry[reg_type].pop(old_category)
            self._save_registry()
            return True
        
        return False
    
    def remove_category(self, reg_type: str, category: str) -> bool:
        """Remove a category from the registry."""
        if reg_type in self.registry and category in self.registry[reg_type]:
            del self.registry[reg_type][category]
            self._save_registry()
            return True
        
        return False
    
    def remove_registry_type(self, reg_type: str) -> bool:
        """Remove a registry type."""
        if reg_type in self.registry:
            del self.registry[reg_type]
            self._save_registry()
            return True
        
        return False
    
    def merge_categories(self, reg_type: str, source_category: str, target_category: str) -> bool:
        """Merge items from source category into target category."""
        if (reg_type in self.registry and 
            source_category in self.registry[reg_type] and
            target_category in self.registry[reg_type]):
            
            # Add items from source to target if they don't already exist
            source_items = self.registry[reg_type][source_category]
            target_items = self.registry[reg_type][target_category]
            
            target_names = [item["name"] for item in target_items]
            
            for item in source_items:
                if item["name"] not in target_names:
                    target_items.append(item)
            
            # Remove source category
            del self.registry[reg_type][source_category]
            self._save_registry()
            return True
        
        return False
    
    def normalize_item_properties(self) -> None:
        """Ensure all items have consistent properties."""
        for reg_type in self.registry:
            for category in self.registry[reg_type]:
                for item in self.registry[reg_type][category]:
                    # Ensure properties dictionary exists
                    if "properties" not in item:
                        item["properties"] = {}
                    
                    # Add color property if missing
                    if "color" not in item["properties"]:
                        item["properties"]["color"] = "gray"
        
        # Save the normalized registry
        self._save_registry()
    
    def initialize_registry(self) -> None:
        """Initialize registry with consistent properties."""
        self._load_registry()
        self.normalize_item_properties() 