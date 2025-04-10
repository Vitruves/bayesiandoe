from typing import Dict, List, Any, Optional, Tuple
import json
import os

class ChemicalRegistry:
    def __init__(self):
        self.registry = {
            "solvents": {
                "Polar Protic": [
                    {"name": "Water", "properties": {"bp": "100°C", "ε": "80.1", "log P": "-1.38", "color": "blue"}},
                    {"name": "Methanol", "properties": {"bp": "64.7°C", "ε": "32.7", "log P": "-0.77", "color": "lightblue"}},
                    {"name": "Ethanol", "properties": {"bp": "78.5°C", "ε": "24.5", "log P": "-0.31", "color": "lightblue"}},
                    {"name": "n-Propanol", "properties": {"bp": "97°C", "ε": "20.1", "log P": "0.25", "color": "lightblue"}},
                    {"name": "i-Propanol", "properties": {"bp": "82.6°C", "ε": "19.9", "log P": "0.05", "color": "lightblue"}},
                    {"name": "n-Butanol", "properties": {"bp": "117.7°C", "ε": "17.8", "log P": "0.88", "color": "lightblue"}},
                    {"name": "t-Butanol", "properties": {"bp": "82.6°C", "ε": "12.5", "log P": "0.35", "color": "lightblue"}},
                    {"name": "Acetic Acid", "properties": {"bp": "118°C", "ε": "6.2", "log P": "-0.17", "color": "lightcoral"}},
                    {"name": "Formic Acid", "properties": {"bp": "100.8°C", "ε": "58.5", "log P": "-0.54", "color": "lightcoral"}},
                    {"name": "TFE", "properties": {"bp": "73.6°C", "ε": "26.1", "log P": "0.41", "color": "lightblue"}},
                    {"name": "HFIP", "properties": {"bp": "58.2°C", "ε": "16.7", "log P": "1.96", "color": "lightblue"}},
                    {"name": "Glycerol", "properties": {"bp": "290°C", "ε": "42.5", "log P": "-1.76", "color": "lightblue"}},
                    {"name": "Ethylene Glycol", "properties": {"bp": "197°C", "ε": "37.7", "log P": "-1.36", "color": "lightblue"}},
                    {"name": "Propylene Glycol", "properties": {"bp": "188°C", "ε": "32.0", "log P": "-0.92", "color": "lightblue"}},
                    {"name": "Deuterium Oxide (D2O)", "properties": {"bp": "101.4°C", "ε": "78.6", "log P": "-1.38", "color": "blue"}},
                    {"name": "Benzyl Alcohol", "properties": {"bp": "205°C", "ε": "13.0", "log P": "1.10", "color": "lightblue"}},
                    {"name": "Pentafluorophenol", "properties": {"bp": "143°C", "ε": "16.7", "log P": "2.03", "color": "lightcoral"}},
                    {"name": "t-Amyl Alcohol", "properties": {"bp": "102°C", "ε": "5.8", "log P": "1.09", "color": "lightblue"}},
                    {"name": "2,2,2-Trifluoroethanol", "properties": {"bp": "73.6°C", "ε": "26.1", "log P": "0.41", "color": "lightblue"}},
                    {"name": "Cyclopentanol", "properties": {"bp": "140°C", "ε": "16.7", "log P": "1.15", "color": "lightblue"}},
                    {"name": "2-Methoxyethanol", "properties": {"bp": "124°C", "ε": "16.9", "log P": "-0.77", "color": "lightblue"}},
                ],
                "Polar Aprotic": [
                    {"name": "Acetone", "properties": {"bp": "56°C", "ε": "20.7", "log P": "-0.24", "color": "mediumpurple"}},
                    {"name": "Acetonitrile", "properties": {"bp": "82°C", "ε": "37.5", "log P": "-0.34", "color": "mediumpurple"}},
                    {"name": "DMF", "properties": {"bp": "153°C", "ε": "36.7", "log P": "-1.01", "color": "mediumpurple"}},
                    {"name": "DMSO", "properties": {"bp": "189°C", "ε": "46.7", "log P": "-1.35", "color": "mediumpurple"}},
                    {"name": "NMP", "properties": {"bp": "202°C", "ε": "32.2", "log P": "-0.46", "color": "mediumpurple"}},
                    {"name": "Sulfolane", "properties": {"bp": "285°C", "ε": "43.3", "log P": "-0.77", "color": "mediumpurple"}},
                    {"name": "HMPA", "properties": {"bp": "235°C", "ε": "30.0", "log P": "0.28", "color": "mediumpurple"}},
                    {"name": "DMAc", "properties": {"bp": "165°C", "ε": "37.8", "log P": "-0.77", "color": "mediumpurple"}},
                    {"name": "Propylene Carbonate", "properties": {"bp": "242°C", "ε": "64.9", "log P": "-0.41", "color": "mediumpurple"}},
                    {"name": "Nitromethane", "properties": {"bp": "101°C", "ε": "35.9", "log P": "-0.33", "color": "mediumpurple"}},
                    {"name": "TMU", "properties": {"bp": "177°C", "ε": "23.1", "log P": "-0.14", "color": "mediumpurple"}},
                    {"name": "DMPU", "properties": {"bp": "246°C", "ε": "36.1", "log P": "-0.31", "color": "mediumpurple"}},
                    {"name": "Diglyme", "properties": {"bp": "162°C", "ε": "7.2", "log P": "-0.36", "color": "mediumpurple"}},
                    {"name": "Triglyme", "properties": {"bp": "216°C", "ε": "7.6", "log P": "-0.19", "color": "mediumpurple"}},
                    {"name": "Propionitrile", "properties": {"bp": "97°C", "ε": "27.2", "log P": "0.16", "color": "mediumpurple"}},
                    {"name": "Ethylene Carbonate", "properties": {"bp": "248°C", "ε": "89.6", "log P": "-0.90", "color": "mediumpurple"}},
                    {"name": "N-Formylmorpholine", "properties": {"bp": "225°C", "ε": "171", "log P": "-0.97", "color": "mediumpurple"}},
                    {"name": "Hexamethylphosphoramide", "properties": {"bp": "235°C", "ε": "30.0", "log P": "0.28", "color": "mediumpurple"}},
                    {"name": "1,3-Dimethyl-2-imidazolidinone", "properties": {"bp": "225°C", "ε": "37.6", "log P": "-0.19", "color": "mediumpurple"}},
                    {"name": "Butyronitrile", "properties": {"bp": "117°C", "ε": "20.3", "log P": "0.60", "color": "mediumpurple"}},
                    {"name": "N-Methyl-2-pyrrolidone", "properties": {"bp": "202°C", "ε": "32.2", "log P": "-0.46", "color": "mediumpurple"}},
                ],
                "Non-Polar": [
                    {"name": "Toluene", "properties": {"bp": "111°C", "ε": "2.4", "log P": "2.73", "color": "darkkhaki"}},
                    {"name": "Benzene", "properties": {"bp": "80°C", "ε": "2.3", "log P": "2.13", "color": "darkkhaki"}},
                    {"name": "Hexane", "properties": {"bp": "69°C", "ε": "1.9", "log P": "3.76", "color": "darkkhaki"}},
                    {"name": "Heptane", "properties": {"bp": "98°C", "ε": "1.9", "log P": "4.27", "color": "darkkhaki"}},
                    {"name": "Cyclohexane", "properties": {"bp": "81°C", "ε": "2.0", "log P": "3.44", "color": "darkkhaki"}},
                    {"name": "Pentane", "properties": {"bp": "36°C", "ε": "1.8", "log P": "3.45", "color": "darkkhaki"}},
                    {"name": "Diethyl Ether", "properties": {"bp": "34.6°C", "ε": "4.3", "log P": "0.89", "color": "darkkhaki"}},
                    {"name": "Chloroform", "properties": {"bp": "61°C", "ε": "4.8", "log P": "1.97", "color": "darkkhaki"}},
                    {"name": "DCM", "properties": {"bp": "40°C", "ε": "9.1", "log P": "1.25", "color": "darkkhaki"}},
                    {"name": "THF", "properties": {"bp": "66°C", "ε": "7.6", "log P": "0.46", "color": "darkkhaki"}},
                    {"name": "Dioxane", "properties": {"bp": "101°C", "ε": "2.3", "log P": "-0.27", "color": "darkkhaki"}},
                    {"name": "Ethyl Acetate", "properties": {"bp": "77°C", "ε": "6.0", "log P": "0.73", "color": "darkkhaki"}},
                    {"name": "MTBE", "properties": {"bp": "55°C", "ε": "2.6", "log P": "0.94", "color": "darkkhaki"}},
                    {"name": "Xylene", "properties": {"bp": "138-144°C", "ε": "2.3", "log P": "3.15", "color": "darkkhaki"}},
                    {"name": "Cyclopentyl methyl ether", "properties": {"bp": "106°C", "ε": "4.8", "log P": "1.59", "color": "darkkhaki"}},
                    {"name": "Carbon Tetrachloride", "properties": {"bp": "77°C", "ε": "2.2", "log P": "2.83", "color": "darkkhaki"}},
                    {"name": "DCE (1,2-Dichloroethane)", "properties": {"bp": "83.5°C", "ε": "10.4", "log P": "1.48", "color": "darkkhaki"}},
                    {"name": "Perfluorohexane", "properties": {"bp": "57°C", "ε": "1.7", "log P": "5.99", "color": "darkkhaki"}},
                    {"name": "Cyclopentane", "properties": {"bp": "49°C", "ε": "1.9", "log P": "3.00", "color": "darkkhaki"}},
                    {"name": "Dibutyl ether", "properties": {"bp": "142°C", "ε": "3.1", "log P": "3.21", "color": "darkkhaki"}},
                    {"name": "Hexafluorobenzene", "properties": {"bp": "80°C", "ε": "2.0", "log P": "2.31", "color": "darkkhaki"}},
                    {"name": "Isooctane", "properties": {"bp": "99°C", "ε": "1.9", "log P": "4.50", "color": "darkkhaki"}},
                    {"name": "Mesitylene", "properties": {"bp": "165°C", "ε": "2.3", "log P": "3.42", "color": "darkkhaki"}},
                    {"name": "Cumene", "properties": {"bp": "152°C", "ε": "2.4", "log P": "3.66", "color": "darkkhaki"}},
                    {"name": "Fluorobenzene", "properties": {"bp": "85°C", "ε": "5.4", "log P": "2.27", "color": "darkkhaki"}},
                ],
                "Green Solvents": [
                    {"name": "2-MeTHF", "properties": {"bp": "80°C", "ε": "6.2", "log P": "0.53", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "Cyrene", "properties": {"bp": "203°C", "ε": "10.6", "log P": "-0.13", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "PEG-400", "properties": {"bp": ">200°C", "ε": "13.6", "renewability": "medium", "color": "mediumseagreen"}},
                    {"name": "Ethyl lactate", "properties": {"bp": "154°C", "ε": "15.4", "log P": "0.19", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "Ethyl acetate", "properties": {"bp": "77°C", "ε": "6.0", "log P": "0.73", "renewability": "medium", "color": "mediumseagreen"}},
                    {"name": "Limonene", "properties": {"bp": "176°C", "ε": "2.3", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "Gamma-valerolactone", "properties": {"bp": "207°C", "ε": "36.5", "log P": "0.08", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "Glycerol carbonate", "properties": {"bp": "110-115°C", "ε": "109.7", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "Dimethyl carbonate", "properties": {"bp": "90°C", "ε": "3.1", "log P": "0.15", "renewability": "medium", "color": "mediumseagreen"}},
                    {"name": "Anisole", "properties": {"bp": "154°C", "ε": "4.3", "renewability": "moderate", "color": "mediumseagreen"}},
                    {"name": "Solketal", "properties": {"bp": "188°C", "ε": "12.2", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "Tetrahydrofurfuryl alcohol", "properties": {"bp": "178°C", "ε": "19.6", "log P": "-0.89", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "p-Cymene", "properties": {"bp": "177°C", "ε": "2.3", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "1,3-Propanediol", "properties": {"bp": "214°C", "ε": "35.0", "renewability": "high", "color": "mediumseagreen"}},
                    {"name": "D-Limonene", "properties": {"bp": "176°C", "ε": "2.3", "renewability": "high", "log P": "4.57", "color": "mediumseagreen"}},
                    {"name": "Methyl lactate", "properties": {"bp": "144°C", "ε": "11.3", "renewability": "high", "log P": "-0.53", "color": "mediumseagreen"}},
                    {"name": "Ethyl levulinate", "properties": {"bp": "206°C", "ε": "7.8", "renewability": "high", "log P": "0.29", "color": "mediumseagreen"}},
                    {"name": "γ-Butyrolactone", "properties": {"bp": "204°C", "ε": "39.1", "renewability": "moderate", "log P": "-0.64", "color": "mediumseagreen"}},
                ]
            },
            "catalysts": {
                "Palladium": [
                    {"name": "Pd(OAc)2", "properties": {"loading": "1-5 mol%", "activation": "easy", "cost": "medium", "color": "darkred"}},
                    {"name": "Pd(PPh3)4", "properties": {"loading": "0.5-5 mol%", "stability": "air-sensitive", "activation": "none", "color": "darkred"}},
                    {"name": "Pd2(dba)3", "properties": {"loading": "1-5 mol%", "stability": "moderate", "activation": "needs ligand", "color": "darkred"}},
                    {"name": "Pd/C", "properties": {"loading": "5-10 wt%", "type": "heterogeneous", "filtration": "easy", "color": "black"}},
                    {"name": "PdCl2", "properties": {"loading": "1-10 mol%", "activation": "needs ligand", "cost": "low", "color": "darkred"}},
                    {"name": "Pd(dppf)Cl2", "properties": {"loading": "1-5 mol%", "precatalyst": "yes", "stability": "high", "color": "darkred"}},
                    {"name": "Pd(OAc)2/PPh3", "properties": {"loading": "1-5 mol%", "L:M": "2:1 to 4:1", "stability": "moderate", "color": "darkred"}},
                    {"name": "Pd(TFA)2", "properties": {"loading": "1-5 mol%", "stability": "moderate", "activation": "fast", "color": "darkred"}},
                    {"name": "PEPPSI-IPr", "properties": {"loading": "0.5-2 mol%", "precatalyst": "yes", "stability": "high", "color": "darkred"}},
                    {"name": "PdCl2(MeCN)2", "properties": {"loading": "1-5 mol%", "activation": "moderate", "needs ligand": "yes", "color": "darkred"}},
                    {"name": "Pd(PPh3)2Cl2", "properties": {"loading": "1-5 mol%", "stability": "high", "precatalyst": "yes", "color": "darkred"}},
                    {"name": "Pd(dba)2", "properties": {"loading": "1-5 mol%", "stability": "moderate", "activation": "needs ligand", "color": "darkred"}},
                    {"name": "[(allyl)PdCl]2", "properties": {"loading": "1-5 mol%", "activation": "moderate", "dimer": "yes", "color": "darkred"}},
                    {"name": "Pd(OAc)2(dppb)", "properties": {"loading": "1-3 mol%", "precatalyst": "yes", "stability": "good", "color": "darkred"}},
                    {"name": "Pd-PEPPSI-IPent", "properties": {"loading": "0.1-1 mol%", "precatalyst": "yes", "stability": "excellent", "color": "darkred"}},
                ],
                "Nickel": [
                    {"name": "Ni(COD)2", "properties": {"loading": "5-10 mol%", "stability": "air-sensitive", "glove box": "yes", "color": "darkgreen"}},
                    {"name": "NiCl2·DME", "properties": {"loading": "5-10 mol%", "activation": "needs reductant", "cost": "low", "color": "darkgreen"}},
                    {"name": "NiCl2·6H2O", "properties": {"loading": "5-10 mol%", "activation": "needs reductant", "water stable": "yes", "color": "darkgreen"}},
                    {"name": "Ni(acac)2", "properties": {"loading": "5-10 mol%", "activation": "moderate", "cost": "moderate", "color": "darkgreen"}},
                    {"name": "NiBr2", "properties": {"loading": "5-10 mol%", "activation": "needs reductant", "hygroscopic": "yes", "color": "darkgreen"}},
                    {"name": "Ni(dppp)Cl2", "properties": {"loading": "1-5 mol%", "precatalyst": "yes", "stability": "high", "color": "darkgreen"}},
                    {"name": "NiCl2(PCy3)2", "properties": {"loading": "1-5 mol%", "precatalyst": "yes", "air stability": "moderate", "color": "darkgreen"}},
                    {"name": "NiCl2(PPh3)2", "properties": {"loading": "1-5 mol%", "precatalyst": "yes", "stability": "moderate", "color": "darkgreen"}},
                    {"name": "Raney Nickel", "properties": {"loading": "catalytic", "type": "heterogeneous", "hydrogenation": "yes", "color": "black"}},
                    {"name": "NiCl2(dppe)", "properties": {"loading": "5-10 mol%", "precatalyst": "yes", "stability": "good", "color": "darkgreen"}},
                    {"name": "Ni(acac)2/PCy3", "properties": {"loading": "5-10 mol%", "L:M": "2:1", "activation": "needs reductant", "color": "darkgreen"}},
                    {"name": "NiI2", "properties": {"loading": "5-10 mol%", "activation": "needs reductant", "hygroscopic": "yes", "color": "darkgreen"}},
                ],
                "Copper": [
                    {"name": "CuI", "properties": {"loading": "5-20 mol%", "type": "Cu(I)", "light sensitive": "yes", "color": "darkgoldenrod"}},
                    {"name": "CuCl", "properties": {"loading": "5-20 mol%", "type": "Cu(I)", "stability": "moderate", "color": "darkgoldenrod"}},
                    {"name": "CuBr", "properties": {"loading": "5-20 mol%", "type": "Cu(I)", "stability": "moderate", "color": "darkgoldenrod"}},
                    {"name": "Cu(OAc)2", "properties": {"loading": "5-20 mol%", "type": "Cu(II)", "stability": "high", "color": "darkgoldenrod"}},
                    {"name": "CuSO4", "properties": {"loading": "5-20 mol%", "type": "Cu(II)", "water soluble": "yes", "color": "darkgoldenrod"}},
                    {"name": "Cu(OTf)2", "properties": {"loading": "1-10 mol%", "type": "Cu(II)", "Lewis acid": "strong", "color": "darkgoldenrod"}},
                    {"name": "Cu2O", "properties": {"loading": "5-10 mol%", "type": "Cu(I)", "heterogeneous": "yes", "color": "darkgoldenrod"}},
                    {"name": "Cu powder", "properties": {"loading": "1-2 equiv", "type": "Cu(0)", "surface area": "important", "color": "darkgoldenrod"}},
                    {"name": "Cu(acac)2", "properties": {"loading": "5-10 mol%", "type": "Cu(II)", "solubility": "organic", "color": "darkgoldenrod"}},
                    {"name": "CuBr·SMe2", "properties": {"loading": "5-10 mol%", "type": "Cu(I)", "stability": "moderate", "color": "darkgoldenrod"}},
                    {"name": "CuTC (thiophene carboxylate)", "properties": {"loading": "1-10 mol%", "type": "Cu(I)", "stability": "moderate", "color": "darkgoldenrod"}},
                    {"name": "Cu(MeCN)4PF6", "properties": {"loading": "1-10 mol%", "type": "Cu(I)", "pi-acid": "good", "color": "darkgoldenrod"}},
                    {"name": "Cu(BF4)2", "properties": {"loading": "5-10 mol%", "type": "Cu(II)", "Lewis acid": "strong", "color": "darkgoldenrod"}},
                    {"name": "Cu(OTf)(C6H6)", "properties": {"loading": "1-5 mol%", "type": "Cu(I)", "pi-acid": "strong", "color": "darkgoldenrod"}},
                ],
                "Other Metals": [
                    {"name": "FeCl3", "properties": {"loading": "5-20 mol%", "type": "Fe(III)", "Lewis acid": "strong", "color": "maroon"}},
                    {"name": "ZnCl2", "properties": {"loading": "10-20 mol%", "type": "Zn(II)", "Lewis acid": "moderate", "color": "darkgray"}},
                    {"name": "AuCl3", "properties": {"loading": "1-5 mol%", "type": "Au(III)", "pi-acid": "strong", "color": "gold"}},
                    {"name": "Ru(bpy)3Cl2", "properties": {"loading": "1-5 mol%", "photocatalyst": "yes", "redox": "versatile", "color": "orangered"}},
                    {"name": "Ir(ppy)3", "properties": {"loading": "0.1-2 mol%", "photocatalyst": "yes", "excited state": "long", "color": "yellowgreen"}},
                    {"name": "AlCl3", "properties": {"loading": "1-2 equiv", "Lewis acid": "strong", "moisture sensitive": "high", "color": "darkgray"}},
                    {"name": "TiCl4", "properties": {"loading": "1-2 equiv", "Lewis acid": "strong", "handling": "difficult", "color": "darkgray"}},
                    {"name": "RhCl3", "properties": {"loading": "1-5 mol%", "stability": "high", "cost": "high", "color": "indianred"}},
                    {"name": "Sc(OTf)3", "properties": {"loading": "5-20 mol%", "Lewis acid": "strong", "water stable": "yes", "color": "darkgray"}},
                    {"name": "In(OTf)3", "properties": {"loading": "5-20 mol%", "Lewis acid": "moderate", "water stable": "yes", "color": "darkgray"}},
                    {"name": "Fe(acac)3", "properties": {"loading": "5-10 mol%", "type": "Fe(III)", "cost": "low", "color": "maroon"}},
                    {"name": "Pt/C", "properties": {"loading": "5-10 wt%", "type": "heterogeneous", "hydrogenation": "yes", "color": "black"}},
                    {"name": "CoCl2", "properties": {"loading": "5-10 mol%", "type": "Co(II)", "cost": "low", "color": "darkblue"}},
                    {"name": "Fe(CO)5", "properties": {"loading": "1-5 mol%", "type": "Fe(0)", "toxicity": "high", "color": "sienna"}},
                    {"name": "FeCl2", "properties": {"loading": "5-10 mol%", "type": "Fe(II)", "air sensitive": "yes", "color": "sienna"}},
                    {"name": "Fe(OTf)2", "properties": {"loading": "1-10 mol%", "type": "Fe(II)", "Lewis acid": "strong", "color": "sienna"}},
                ],
                "Organocatalysts": [
                    {"name": "L-Proline", "properties": {"loading": "10-30 mol%", "type": "amino acid", "mode": "enamine/iminium", "color": "teal"}},
                    {"name": "MacMillan", "properties": {"loading": "10-20 mol%", "type": "imidazolidinone", "mode": "iminium", "color": "teal"}},
                    {"name": "BINOL", "properties": {"loading": "5-20 mol%", "type": "diol", "mode": "hydrogen-bonding", "color": "teal"}},
                    {"name": "Quinine", "properties": {"loading": "5-10 mol%", "type": "cinchona", "mode": "base/H-bond", "color": "teal"}},
                    {"name": "Thiourea", "properties": {"loading": "5-10 mol%", "mode": "hydrogen-bonding", "dual activation": "yes", "color": "teal"}},
                    {"name": "DABCO", "properties": {"loading": "5-20 mol%", "type": "diamine", "mode": "nucleophilic", "color": "teal"}},
                    {"name": "Proline-tetrazole", "properties": {"loading": "5-15 mol%", "type": "amino acid", "mode": "enamine", "color": "teal"}},
                    {"name": "Cinchonidine", "properties": {"loading": "5-10 mol%", "type": "cinchona", "mode": "base/H-bond", "color": "teal"}},
                    {"name": "Squaramide", "properties": {"loading": "2-10 mol%", "mode": "hydrogen-bonding", "dual activation": "yes", "color": "teal"}},
                    {"name": "DMAP", "properties": {"loading": "1-10 mol%", "type": "pyridine", "mode": "nucleophilic", "color": "teal"}},
                    {"name": "Jacobsen Thiourea", "properties": {"loading": "1-5 mol%", "type": "thiourea", "chiral": "yes", "color": "teal"}},
                    {"name": "Takemoto Catalyst", "properties": {"loading": "1-10 mol%", "type": "bifunctional", "mode": "H-bond/base", "color": "teal"}},
                    {"name": "Maruoka Catalyst", "properties": {"loading": "1-5 mol%", "type": "BINOL-derived", "mode": "phase-transfer", "color": "teal"}},
                    {"name": "D-Proline", "properties": {"loading": "10-30 mol%", "type": "amino acid", "mode": "enamine/iminium", "color": "teal"}},
                    {"name": "Hayashi-Jørgensen", "properties": {"loading": "5-20 mol%", "type": "prolinol", "mode": "enamine", "color": "teal"}},
                    {"name": "Cinchonine", "properties": {"loading": "5-10 mol%", "type": "cinchona", "mode": "base/H-bond", "color": "teal"}},
                    {"name": "Quinidine", "properties": {"loading": "5-10 mol%", "type": "cinchona", "mode": "base/H-bond", "color": "teal"}},
                    {"name": "Prolinol", "properties": {"loading": "5-20 mol%", "type": "amino alcohol", "mode": "enamine", "color": "teal"}},
                    {"name": "TBD", "properties": {"loading": "1-10 mol%", "type": "guanidine", "mode": "base", "color": "teal"}},
                    {"name": "DBU", "properties": {"loading": "1-20 mol%", "type": "amidine", "mode": "base", "color": "teal"}},
                    {"name": "TEMPO", "properties": {"loading": "1-10 mol%", "type": "nitroxyl radical", "mode": "oxidation", "color": "red"}},
                ],
                "Iron": [
                    {"name": "Fe(acac)3", "properties": {"loading": "5-10 mol%", "type": "Fe(III)", "cost": "low", "color": "sienna"}},
                    {"name": "Fe(CO)5", "properties": {"loading": "1-5 mol%", "type": "Fe(0)", "toxicity": "high", "color": "sienna"}},
                    {"name": "FeCl2", "properties": {"loading": "5-10 mol%", "type": "Fe(II)", "air sensitive": "yes", "color": "sienna"}},
                    {"name": "FeCl3", "properties": {"loading": "5-10 mol%", "type": "Fe(III)", "Lewis acid": "strong", "color": "sienna"}},
                    {"name": "Fe(OTf)2", "properties": {"loading": "1-10 mol%", "type": "Fe(II)", "Lewis acid": "strong", "color": "sienna"}},
                ],
                "Buchwald Precatalysts": [
                    {"name": "SPhos Pd G1", "properties": {"generation": "1st", "ligand": "SPhos", "activation": "moderate", "stability": "good", "color": "darkred"}},
                    {"name": "SPhos Pd G2", "properties": {"generation": "2nd", "ligand": "SPhos", "activation": "fast", "stability": "good", "color": "darkred"}},
                    {"name": "SPhos Pd G3", "properties": {"generation": "3rd", "ligand": "SPhos", "activation": "rapid", "stability": "excellent", "color": "darkred"}},
                    {"name": "XPhos Pd G1", "properties": {"generation": "1st", "ligand": "XPhos", "activation": "moderate", "stability": "good", "color": "darkred"}},
                    {"name": "XPhos Pd G2", "properties": {"generation": "2nd", "ligand": "XPhos", "activation": "fast", "stability": "good", "color": "darkred"}},
                    {"name": "XPhos Pd G3", "properties": {"generation": "3rd", "ligand": "XPhos", "activation": "rapid", "stability": "excellent", "color": "darkred"}},
                    {"name": "RuPhos Pd G1", "properties": {"generation": "1st", "ligand": "RuPhos", "activation": "moderate", "stability": "good", "color": "darkred"}},
                    {"name": "RuPhos Pd G2", "properties": {"generation": "2nd", "ligand": "RuPhos", "activation": "fast", "stability": "good", "color": "darkred"}},
                    {"name": "RuPhos Pd G3", "properties": {"generation": "3rd", "ligand": "RuPhos", "activation": "rapid", "stability": "excellent", "color": "darkred"}},
                    {"name": "BrettPhos Pd G3", "properties": {"generation": "3rd", "ligand": "BrettPhos", "activation": "rapid", "stability": "excellent", "color": "darkred"}},
                    {"name": "tBuXPhos Pd G3", "properties": {"generation": "3rd", "ligand": "tBuXPhos", "activation": "rapid", "stability": "excellent", "color": "darkred"}},
                    {"name": "AdBrettPhos Pd G3", "properties": {"generation": "3rd", "ligand": "AdBrettPhos", "activation": "rapid", "stability": "excellent", "color": "darkred"}}
                ],
            },
            "ligands": {
                "Phosphines": [
                    {"name": "PPh3", "properties": {"bite angle": "n/a", "donor": "P", "cone angle": "145°", "color": "darkorange"}},
                    {"name": "BINAP", "properties": {"bite angle": "92°", "chiral": "yes", "rigidity": "high", "color": "darkorange"}},
                    {"name": "Xantphos", "properties": {"bite angle": "108°", "flexibility": "rigid", "electronics": "moderate", "color": "darkorange"}},
                    {"name": "dppf", "properties": {"bite angle": "99°", "flexibility": "moderate", "electronics": "moderate", "color": "darkorange"}},
                    {"name": "XPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "rich", "cone angle": "212°", "color": "darkorange"}},
                    {"name": "SPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "rich", "cone angle": "206°", "color": "darkorange"}},
                    {"name": "PCy3", "properties": {"bite angle": "n/a", "electron-rich": "yes", "cone angle": "170°", "color": "darkorange"}},
                    {"name": "dppe", "properties": {"bite angle": "85°", "flexibility": "rigid", "chelation": "strong", "color": "darkorange"}},
                    {"name": "dppp", "properties": {"bite angle": "91°", "flexibility": "moderate", "chelation": "strong", "color": "darkorange"}},
                    {"name": "JohnPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "rich", "cone angle": "203°", "color": "darkorange"}},
                    {"name": "DavePhos", "properties": {"bite angle": "n/a", "steric": "moderate", "electronics": "rich", "cone angle": "187°", "color": "darkorange"}},
                    {"name": "CyJohnPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "rich", "cone angle": "238°", "color": "darkorange"}},
                    {"name": "tBuXPhos", "properties": {"bite angle": "n/a", "steric": "very bulky", "electronics": "rich", "cone angle": "258°", "color": "darkorange"}},
                    {"name": "BrettPhos", "properties": {"bite angle": "n/a", "steric": "very bulky", "electronics": "rich", "cone angle": "265°", "color": "darkorange"}},
                    {"name": "RuPhos", "properties": {"bite angle": "n/a", "steric": "bulky", "electronics": "moderate", "cone angle": "241°", "color": "darkorange"}},
                    {"name": "P(tBu)3", "properties": {"bite angle": "n/a", "electron-rich": "very", "cone angle": "182°", "color": "darkorange"}},
                    {"name": "MePhos", "properties": {"bite angle": "n/a", "steric": "small", "electronics": "moderate", "cone angle": "136°", "color": "darkorange"}},
                    {"name": "DPEPhos", "properties": {"bite angle": "101°", "flexibility": "flexible", "electronics": "moderate", "color": "darkorange"}},
                    {"name": "DPPF", "properties": {"bite angle": "99°", "flexibility": "moderate", "ferrocene": "yes", "color": "darkorange"}},
                    {"name": "DCPP", "properties": {"bite angle": "102°", "flexibility": "moderate", "electron-rich": "yes", "color": "darkorange"}},
                    {"name": "DTBPF", "properties": {"bite angle": "99°", "steric": "very bulky", "electron-rich": "very", "color": "darkorange"}},
                ],
                "Nitrogen-based": [
                    {"name": "BiPy", "properties": {"bite angle": "78°", "rigidity": "rigid", "pi-acceptor": "good", "color": "darkcyan"}},
                    {"name": "Phen", "properties": {"bite angle": "82°", "rigidity": "rigid", "pi-acceptor": "good", "color": "darkcyan"}},
                    {"name": "TMEDA", "properties": {"bite angle": "85°", "flexibility": "high", "basicity": "high", "color": "darkcyan"}},
                    {"name": "L-Proline", "properties": {"type": "organocatalyst", "chirality": "L", "mode": "enamine/iminium", "color": "darkcyan"}},
                    {"name": "Pyridine", "properties": {"denticity": "monodentate", "pi-acceptor": "good", "basicity": "moderate", "color": "darkcyan"}},
                    {"name": "DABCO", "properties": {"denticity": "bidentate", "basicity": "high", "nucleophilicity": "high", "color": "darkcyan"}},
                    {"name": "DMAP", "properties": {"denticity": "monodentate", "nucleophilicity": "high", "basicity": "high", "color": "darkcyan"}},
                    {"name": "DBU", "properties": {"basicity": "very high", "nucleophilicity": "low", "type": "amidine", "color": "darkcyan"}},
                    {"name": "Quinuclidine", "properties": {"basicity": "high", "nucleophilicity": "high", "structure": "rigid", "color": "darkcyan"}},
                    {"name": "DPEN", "properties": {"type": "diamine", "chiral": "yes", "reduction": "transfer hydrogenation", "color": "darkcyan"}}
                ],
                "N-Heterocyclic Carbenes": [
                    {"name": "IMes", "properties": {"type": "imidazolium", "steric": "moderate", "electronics": "strongly donating", "color": "purple"}},
                    {"name": "IPr", "properties": {"type": "imidazolium", "steric": "bulky", "electronics": "strongly donating", "color": "purple"}},
                    {"name": "SIMes", "properties": {"type": "imidazolinium", "steric": "moderate", "electronics": "very donating", "color": "purple"}},
                    {"name": "SIPr", "properties": {"type": "imidazolinium", "steric": "bulky", "electronics": "very donating", "color": "purple"}},
                    {"name": "ICy", "properties": {"type": "imidazolium", "steric": "moderate", "electronics": "donating", "color": "purple"}},
                    {"name": "ItBu", "properties": {"type": "imidazolium", "steric": "very bulky", "electronics": "donating", "color": "purple"}},
                    {"name": "IAd", "properties": {"type": "imidazolium", "steric": "extremely bulky", "electronics": "donating", "color": "purple"}},
                    {"name": "IBioxMe4", "properties": {"type": "benzimidazolium", "steric": "moderate", "rigidity": "high", "color": "purple"}},
                    {"name": "IPrCl", "properties": {"type": "imidazolium", "steric": "bulky", "electronics": "less donating", "color": "purple"}},
                    {"name": "IPent", "properties": {"type": "imidazolium", "steric": "very bulky", "electronics": "strongly donating", "color": "purple"}},
                    {"name": "IiPr", "properties": {"type": "imidazolium", "steric": "moderate", "electronics": "donating", "color": "purple"}},
                    {"name": "CAAC", "properties": {"type": "cyclic alkylaminocarbene", "steric": "very bulky", "electronics": "extremely donating", "color": "purple"}},
                    {"name": "MIC", "properties": {"type": "mesoionic carbene", "steric": "variable", "electronics": "strongly donating", "color": "purple"}}
                ],
                "P,N-Ligands": [
                    {"name": "PHOX", "properties": {"type": "P,N-ligand", "chiral": "yes", "hemilabile": "yes", "color": "saddlebrown"}},
                    {"name": "PyPhos", "properties": {"type": "P,N-ligand", "hemilabile": "yes", "flexibility": "moderate", "color": "saddlebrown"}},
                    {"name": "QUINAP", "properties": {"type": "P,N-ligand", "chiral": "yes", "atropisomeric": "yes", "color": "saddlebrown"}},
                    {"name": "Josiphos", "properties": {"type": "P,P-ligand", "chiral": "yes", "ferrocene": "yes", "color": "saddlebrown"}}
                ],
                "Palladacycles": [
                    {"name": "Buchwald G3", "properties": {"generation": "3rd", "ligand": "various", "activation": "fast", "stability": "high", "color": "darkslategray"}},
                    {"name": "Herrmann-Beller", "properties": {"generation": "1st", "ligand": "P(o-tolyl)3", "activation": "slow", "stability": "moderate", "color": "darkslategray"}},
                    {"name": "Fuhrer Catalyst", "properties": {"generation": "mixed", "ligand": "various", "stability": "high", "color": "darkslategray"}},
                    {"name": "Nolan Catalysts (PEPPSI)", "properties": {"generation": "2nd/3rd", "ligand": "NHC", "activation": "fast", "stability": "high", "color": "darkslategray"}},
                    {"name": "Umicore Catalysts", "properties": {"generation": "various", "ligand": "NHC/phosphine", "stability": "high", "color": "darkslategray"}},
                    {"name": "Catacxium A", "properties": {"type": "palladacycle-based", "ligand": "phosphine", "stability": "high", "color": "darkslategray"}},
                    {"name": "Buchwald PG G1", "properties": {"generation": "1st", "ligand": "SPhos/XPhos/RuPhos", "activation": "moderate", "stability": "good", "color": "darkslategray"}},
                    {"name": "Buchwald PG G2", "properties": {"generation": "2nd", "ligand": "SPhos/XPhos/RuPhos", "activation": "fast", "stability": "good", "color": "darkslategray"}}, 
                    {"name": "Buchwald PG G3", "properties": {"generation": "3rd", "ligand": "SPhos/XPhos/RuPhos", "activation": "rapid", "stability": "excellent", "color": "darkslategray"}},
                    {"name": "Buchwald PG G4", "properties": {"generation": "4th", "ligand": "SPhos/XPhos/RuPhos", "activation": "very rapid", "stability": "excellent", "color": "darkslategray"}},
                    {"name": "Buchwald SPhos G3", "properties": {"generation": "3rd", "ligand": "SPhos", "activation": "rapid", "stability": "excellent", "color": "darkslategray"}},
                    {"name": "Buchwald XPhos G3", "properties": {"generation": "3rd", "ligand": "XPhos", "activation": "rapid", "stability": "excellent", "color": "darkslategray"}},
                    {"name": "Buchwald RuPhos G3", "properties": {"generation": "3rd", "ligand": "RuPhos", "activation": "rapid", "stability": "excellent", "color": "darkslategray"}},
                    {"name": "Buchwald BrettPhos G3", "properties": {"generation": "3rd", "ligand": "BrettPhos", "activation": "rapid", "stability": "excellent", "color": "darkslategray"}},
                ],
                "Bidentate N,N-Ligands": [
                    {"name": "Phenanthroline", "properties": {"bite angle": "82°", "rigidity": "high", "pi-acceptor": "good", "color": "mediumvioletred"}},
                    {"name": "BBEDA", "properties": {"bite angle": "78°", "flexibility": "moderate", "pi-donor": "good", "color": "mediumvioletred"}},
                    {"name": "TMEDA", "properties": {"bite angle": "85°", "flexibility": "high", "basicity": "high", "color": "mediumvioletred"}},
                    {"name": "Neocuproine", "properties": {"bite angle": "83°", "rigidity": "high", "steric": "moderate", "color": "mediumvioletred"}},
                    {"name": "Box ligands", "properties": {"bite angle": "85-90°", "chirality": "C2", "rigidity": "high", "color": "mediumvioletred"}}
                ],
            },
            "bases": {
                "Inorganic": [
                    {"name": "K2CO3", "properties": {"pKa": "10.3", "solubility(H2O)": "112 g/L", "solubility(organic)": "low", "color": "dimgray"}},
                    {"name": "Cs2CO3", "properties": {"pKa": "10.3", "solubility(H2O)": "261 g/L", "solubility(organic)": "moderate", "color": "dimgray"}},
                    {"name": "K3PO4", "properties": {"pKa": "12.3", "solubility(H2O)": "90 g/L", "solubility(organic)": "low", "color": "dimgray"}},
                    {"name": "NaH", "properties": {"pKa": "35", "reactivity": "high", "handling": "pyrophoric", "color": "dimgray"}},
                    {"name": "NaOH", "properties": {"pKa": "15.7", "solubility(H2O)": "1090 g/L", "hygroscopic": "yes", "color": "dimgray"}},
                    {"name": "KOH", "properties": {"pKa": "15.7", "solubility(H2O)": "1120 g/L", "hygroscopic": "yes", "color": "dimgray"}},
                    {"name": "NaHCO3", "properties": {"pKa": "6.4", "solubility(H2O)": "96 g/L", "mildness": "high", "color": "dimgray"}},
                    {"name": "Na2CO3", "properties": {"pKa": "10.3", "solubility(H2O)": "215 g/L", "solubility(organic)": "low", "color": "dimgray"}},
                    {"name": "KF", "properties": {"pKa": "10.8", "solubility(H2O)": "923 g/L", "nucleophilicity": "high", "color": "dimgray"}},
                    {"name": "CsF", "properties": {"pKa": "10.8", "solubility(H2O)": "367 g/L", "solubility(organic)": "moderate", "color": "dimgray"}},
                    {"name": "LiOH", "properties": {"pKa": "15.7", "solubility(H2O)": "128 g/L", "coordination": "strong", "color": "dimgray"}},
                    {"name": "Na2HPO4", "properties": {"pKa": "7.2", "solubility(H2O)": "77 g/L", "mildness": "high", "color": "dimgray"}},
                    {"name": "Li2CO3", "properties": {"pKa": "10.3", "solubility(H2O)": "13 g/L", "coordination": "strong", "color": "dimgray"}},
                    {"name": "Mg(OH)2", "properties": {"pKa": "11.4", "solubility(H2O)": "low", "mildness": "very high", "color": "dimgray"}},
                    {"name": "Ca(OH)2", "properties": {"pKa": "12.4", "solubility(H2O)": "low", "mildness": "high", "color": "dimgray"}},
                    {"name": "KHCO3", "properties": {"pKa": "6.4", "solubility(H2O)": "322 g/L", "mildness": "high", "color": "dimgray"}},
                    {"name": "LiOtBu", "properties": {"pKa": "18", "solubility(H2O)": "reacts", "solubility(THF)": "good", "color": "dimgray"}},
                    {"name": "K2CO3/18-crown-6", "properties": {"pKa": "10.3", "solubility(organic)": "enhanced", "phase transfer": "yes", "color": "dimgray"}},
                    {"name": "KH", "properties": {"pKa": "35", "reactivity": "very high", "handling": "pyrophoric", "color": "dimgray"}},
                    {"name": "Ba(OH)2", "properties": {"pKa": "13.4", "solubility(H2O)": "39 g/L", "mildness": "moderate", "color": "dimgray"}},
                    {"name": "RbOH", "properties": {"pKa": "15.7", "solubility(H2O)": "high", "hygroscopic": "very", "color": "dimgray"}},
                    {"name": "K3PO4·H2O", "properties": {"pKa": "12.3", "solubility(H2O)": "90 g/L", "solubility(organic)": "low", "color": "dimgray"}},
                    {"name": "KOtBu/18-crown-6", "properties": {"pKa": "18", "solubility(organic)": "enhanced", "reactivity": "high", "color": "dimgray"}},
                    {"name": "Sr(OH)2", "properties": {"pKa": "13.2", "solubility(H2O)": "moderate", "mildness": "moderate", "color": "dimgray"}},
                ],
                "Organic": [
                    {"name": "Et3N", "properties": {"pKa": "10.8", "bp": "89°C", "nucleophilicity": "moderate", "color": "darkslateblue"}},
                    {"name": "DIPEA", "properties": {"pKa": "11.4", "bp": "127°C", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "DBU", "properties": {"pKa": "13.5", "bp": "180°C", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "LDA", "properties": {"pKa": "36", "reactivity": "very high", "nucleophilicity": "high", "color": "darkslateblue"}},
                    {"name": "NaOt-Bu", "properties": {"pKa": "18", "reactivity": "high", "solubility(THF)": "good", "color": "darkslateblue"}},
                    {"name": "KOt-Bu", "properties": {"pKa": "18", "reactivity": "high", "solubility(THF)": "good", "color": "darkslateblue"}},
                    {"name": "Pyridine", "properties": {"pKa": "5.2", "bp": "115°C", "nucleophilicity": "moderate", "color": "darkslateblue"}},
                    {"name": "DMAP", "properties": {"pKa": "9.7", "nucleophilicity": "high", "catalytic": "yes", "color": "darkslateblue"}},
                    {"name": "TMG", "properties": {"pKa": "13.6", "bp": "160°C", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "MTBD", "properties": {"pKa": "15.0", "bp": "241°C", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "Proton Sponge", "properties": {"pKa": "12.1", "nucleophilicity": "very low", "steric": "hindered", "color": "darkslateblue"}},
                    {"name": "KHMDS", "properties": {"pKa": "26", "reactivity": "high", "selectivity": "good", "color": "darkslateblue"}},
                    {"name": "LiHMDS", "properties": {"pKa": "26", "reactivity": "moderate", "selectivity": "excellent", "color": "darkslateblue"}},
                    {"name": "NaHMDS", "properties": {"pKa": "26", "reactivity": "high", "selectivity": "good", "color": "darkslateblue"}},
                    {"name": "n-BuLi", "properties": {"pKa": "50", "reactivity": "very high", "nucleophilicity": "high", "color": "darkslateblue"}},
                    {"name": "s-BuLi", "properties": {"pKa": "51", "reactivity": "very high", "nucleophilicity": "high", "color": "darkslateblue"}},
                    {"name": "t-BuLi", "properties": {"pKa": "53", "reactivity": "extremely high", "nucleophilicity": "moderate", "color": "darkslateblue"}},
                    {"name": "LTMP", "properties": {"pKa": "37", "reactivity": "high", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "2,6-Lutidine", "properties": {"pKa": "6.7", "bp": "143°C", "nucleophilicity": "very low", "color": "darkslateblue"}},
                    {"name": "BTMG", "properties": {"pKa": "13.6", "bp": "190°C", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "P4-tBu", "properties": {"pKa": "33.5", "reactivity": "high", "non-nucleophilic": "yes", "color": "darkslateblue"}},
                    {"name": "Quinuclidine", "properties": {"pKa": "11.0", "bp": "162°C", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "BTPP", "properties": {"pKa": "28.4", "reactivity": "high", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "Barton's Base", "properties": {"pKa": "16.5", "nucleophilicity": "low", "steric": "hindered", "color": "darkslateblue"}},
                    {"name": "DBN", "properties": {"pKa": "13.5", "bp": "177°C", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "BEMP", "properties": {"pKa": "27.6", "reactivity": "high", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "BTTP", "properties": {"pKa": "28.4", "reactivity": "high", "nucleophilicity": "low", "color": "darkslateblue"}}
                ],
                "Strong Non-nucleophilic": [
                    {"name": "P1-tBu", "properties": {"pKa": "28.4", "reactivity": "high", "nucleophilicity": "very low", "color": "darkslateblue"}},
                    {"name": "P2-tBu", "properties": {"pKa": "30.2", "reactivity": "high", "nucleophilicity": "very low", "color": "darkslateblue"}},
                    {"name": "P4-tBu", "properties": {"pKa": "33.5", "reactivity": "very high", "nucleophilicity": "very low", "color": "darkslateblue"}},
                    {"name": "TMG", "properties": {"pKa": "13.6", "reactivity": "moderate", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "MTBD", "properties": {"pKa": "15.0", "reactivity": "moderate", "nucleophilicity": "low", "color": "darkslateblue"}},
                    {"name": "DBU/LiCl", "properties": {"pKa": "~14", "reactivity": "moderate", "nucleophilicity": "very low", "color": "darkslateblue"}},
                    {"name": "t-BuP4", "properties": {"pKa": "42.7", "reactivity": "extremely high", "nucleophilicity": "very low", "color": "darkslateblue"}}
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
                    {"name": "Zn(OTf)2", "properties": {"acidity": "moderate", "stability": "water-stable", "tolerance": "high", "color": "gray"}},
                    {"name": "B(C6F5)3", "properties": {"acidity": "very strong", "stability": "moisture-sensitive", "fluorophilicity": "high", "color": "lightslategray"}},
                    {"name": "InCl3", "properties": {"acidity": "moderate", "handling": "air-stable", "solubility": "good", "color": "lightslategray"}},
                    {"name": "TMSOTf", "properties": {"acidity": "strong", "handling": "moisture-sensitive", "silylating": "yes", "color": "lightslategray"}},
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
    
    def get_item_display_name(self, reg_type: str, category: str, item_name: str) -> str:
        """Get a display name for the item including its key property."""
        properties = self.get_item_properties(reg_type, category, item_name)
        key_prop_str = ""

        if not properties:
            return item_name

        try:
            if reg_type == "solvents":
                props = []
                if "bp" in properties:
                    props.append(f"bp={properties['bp']}")
                if "ε" in properties:
                    props.append(f"ε={properties['ε']}")
                if props:
                    key_prop_str = ", ".join(props)
            elif reg_type == "bases" or reg_type == "additives" and category == "Acids/Bases":
                if "pKa" in properties:
                    key_prop_str = f"pKa={properties['pKa']}"
            elif reg_type == "ligands":
                if category == "Phosphines" and "cone angle" in properties:
                    key_prop_str = f"θ={properties['cone angle']}"
                elif category == "N-Heterocyclic Carbenes" and "steric" in properties:
                    key_prop_str = f"steric={properties['steric']}"
                elif "steric" in properties: # Fallback for other ligands
                    key_prop_str = f"steric={properties['steric']}"
            elif reg_type == "catalysts":
                if "loading" in properties:
                    key_prop_str = f"load={properties['loading']}"
            elif reg_type == "additives":
                if "type" in properties:
                    key_prop_str = f"type={properties['type']}"
        except Exception as e:
            print(f"Error formatting display name for {item_name}: {e}")
            key_prop_str = ""

        if key_prop_str:
            return f"{item_name} ({key_prop_str})"
        else:
            return item_name
    
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