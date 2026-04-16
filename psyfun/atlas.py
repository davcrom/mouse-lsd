import numpy as np

from iblutil.numerical import ismember
from iblatlas.atlas import BrainRegions
import iblatlas.plots as anatomyplots
regions = BrainRegions()


def region_parcellation(acronyms):
    regions = np.full(len(acronyms), '', dtype=object)

    regions[
        np.isin(acronyms,
            [
                'ACA',
                'ACAd', 'ACAd1', 'ACAd2/3', 'ACAd5', 'ACAd6a', 'ACAd6b',
                'ACAv', 'ACAv1', 'ACAv2/3', 'ACAv5', 'ACAv6a','ACAv6b',
                'ILA', 'ILA1', 'ILA273', 'ILA5', 'ILA6a', 'ILA6b',
                'PL', 'PL1', 'PL2/3', 'PL5', 'PL6a', 'PL6b',
            ]
        )
    ] = 'mPFC'

    regions[
        np.isin(acronyms,
            [
                'MOp', 'MOp1', 'MOp2/3', 'MOp5', 'MOp6a', 'MOp6b'
            ]
        )
    ] = 'M1'

    regions[
        np.isin(acronyms,
            [
                'MOs', 'MOs1', 'MOs2/3', 'MOs5', 'MOs6a', 'MOs6b'
            ]
        )
    ] = 'M2'

    regions[
        np.isin(acronyms,
            [
                'DP', 'OLF', 'PIR', 'TTd'
            ]
        )
    ] = 'OLF'

    regions[
        np.isin(acronyms,
            [
                'SSp',
                'SSP-m', 'SSp-m1', 'SSp-m2/3', 'SSp-m4', 'SSp-m5', 'SSp-m6a', 'SSp-m6b',
                'SSP-n', 'SSp-n1', 'SSp-n2/3', 'SSp-n4', 'SSp-n5', 'SSp-n6a', 'SSp-n6b',
                'SSP-bfd', 'SSp-bfd1', 'SSp-bfd2/3', 'SSp-bfd4', 'SSp-bfd5', 'SSp-bfd6a', 'SSp-bfd6b',
                'SSP-tr', 'SSp-tr1', 'SSp-tr2/3', 'SSp-tr4', 'SSp-tr5', 'SSp-tr6a', 'SSp-tr6b',
                'SSP-ul', 'SSp-ul1', 'SSp-ul2/3', 'SSp-ul4', 'SSp-ul5', 'SSp-ul6a', 'SSp-ul6b',
                'SSP-ll', 'SSp-ll1', 'SSp-ll2/3', 'SSp-ll4', 'SSp-ll5', 'SSp-ll6a', 'SSp-ll6b',
                'SSP-un', 'SSp-un1', 'SSp-un2/3', 'SSp-un4', 'SSp-un5', 'SSp-un6a', 'SSp-un6b',
                'SSs', 'SSs1', 'SSs2/3', 'SSs4', 'SSs5', 'SSs6a', 'SSs6b',
            ]
        )
    ] = 'SS'

    regions[
        np.isin(acronyms,
            [
                'RSP',
                'RSPagl', 'RSPagl1', 'RSPagl2/3', 'RSPagl5', 'RSPagl6a', 'RSPagl6b',
                'RSPd', 'RSPd1', 'RSPd2/3', 'RSPd5', 'RSPd6a', 'RSPd6b',
                'RSPv', 'RSPv1', 'RSPv2/3', 'RSPv5', 'RSPv6a', 'RSPv6b'
            ]
        )
    ] = 'RSP'

    regions[
        np.isin(acronyms,
            [
                'ECT', 'ECT1', 'ECT2/3', 'ECT5', 'ECT6a', 'ECT6b',
                'TEa', 'TEa1', 'TEa2/3', 'TEa4', 'TEa5', 'TEa6a', 'TEa6b'
            ]
        )
    ] = 'TMP'

    regions[
        np.isin(acronyms,
            [
                'AUD',
                'AUDp', 'AUDp1', 'AUDp2/3', 'AUDp4', 'AUDp5', 'AUDp6a', 'AUDp6b',
                'AUDd', 'AUDd1', 'AUDd2/3', 'AUDd4', 'AUDd5', 'AUDd6a', 'AUDd6b',
                'AUDv', 'AUDv1', 'AUDv2/3', 'AUDv4', 'AUDv5', 'AUDv6a', 'AUDv6b',
                'AUDpo', 'AUDpo1', 'AUDpo2/3', 'AUDpo4', 'AUDpo5', 'AUDpo6a', 'AUDpo6b',
                'AUDd5', 'AUDp5', 'AUDp6b', 'AUDv4', 'AUDv5', 'AUDv6a', 'AUDv6b'])
    ] = 'AUD'

    regions[
        np.isin(acronyms,
            [
                'VIS',
                'VISp', 'VISp1', 'VISp2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b',
                'VISa', 'VISa1', 'VISa2/3', 'VISa4', 'VISa5', 'VISa6a', 'VISa6b',
                'VISal', 'VISal1', 'VISal2/3', 'VISal4', 'VISal5', 'VISal6a', 'VISal6b',
                'VISam', 'VISam1', 'VISam2/3', 'VISam4', 'VISam5', 'VISam6a', 'VISam6b',
                'VISl', 'VISl1', 'VISl2/3', 'VISl4', 'VISl5', 'VISl6a', 'VISl6b',
                'VISpl', 'VISpl1', 'VISpl2/3', 'VISpl4', 'VISpl5', 'VISpl6a', 'VISpl6b',
                'VISpm', 'VISpm1', 'VISpm2/3', 'VISpm4', 'VISpm5', 'VISpm6a', 'VISpm6b',
                'VISli', 'VISli1', 'VISli2/3', 'VISli4', 'VISli5', 'VISli6a', 'VISli6b',
                'VISpor', 'VISpor1', 'VISpor2/3', 'VISpor4', 'VISpor5', 'VISpor6a', 'VISpor6b',
                'VISrl', 'VISrl1', 'VISrl2/3', 'VISrl4', 'VISrl5', 'VISrl6a', 'VISrl6b'
            ]
        )
    ] = 'VIS'

    regions[
        np.isin(acronyms,
            [
                'TH'
            ]
        )
    ] = 'TH'

    regions[
        np.isin(acronyms,
            [
                'VPL', 'VPLpc', 'VPM', 'VPMpc', 'PO'
            ]
        )
    ] = 'THss'

    regions[
        np.isin(acronyms,
            [
                'LGv', 'LGd', 'LGd-co', 'LGd-ip', 'LGd-sh', 'LP'
            ]
        )
    ] = 'THvis'

    regions[
        np.isin(acronyms,
            [
                'MD',  # frontal cortex, basal ganglia
                'LD',  # retrosplenial cortex
                'VAL', 'VM'  # motor cortex, basal ganglia
            ]
        )
    ] = 'THcog'

    regions[
        np.isin(acronyms,
            [
                'RE', 'PR', 'RH',
                'AM', 'AMd', 'AMv', 'IAD', 'IAM'
            ]
        )
    ] = 'THhpc'

    regions[
        np.isin(acronyms,
            [
                'PVT', 'IMD', 'Xi', 'Eth'
            ]
        )
    ] = 'THmid'

    regions[
        np.isin(acronyms,
            [
                'CL', 'CM', 'PCN', 'SMT'
            ]
        )
    ] = 'THil'

    regions[
        np.isin(acronyms,
            [
                'RT'
            ]
        )
    ] = 'TRN'

    regions[
        np.isin(acronyms,
            [
                'BLA','BLAa', 'BLAp', 'BLAv',
                'BMA', 'BMAa', 'BMAp',
                'CEA', 'CEAc', 'CEAl', 'CEAm',
                'COA', 'COAp', 'COAa', 'COApm', 'COApl',
                'IA', 'LA', 'MEA', 'PA', 'PAA',
                'BST'
            ]
        )
    ] = 'AMY'

    regions[
        np.isin(acronyms,
            [
                'HPF',
                'CA1', 'CA2', 'CA3',
                'DG-mo', 'DG-po', 'DG-sg',
                'ENTl', 'ENTl1', 'ENTl2', 'ENTl3', 'ENTl5', 'ENTl6a',
                'SUB', 'IG'
            ]
        )
    ] = 'HPC'

    regions[
        np.isin(acronyms,
            [
                'STR', 'CP', 'ACB',
                'PAL', 'GPe', 'GPi',
            ]
        )
    ] = 'STR'

    regions[
        np.isin(acronyms,
            [
                'HY', 'DMH',
                'AVP', 'ADP', 'MPO', 'LPO', 'VMPO', 'VMPO', 'MEPO',
                'PVH', 'PVHd', 'PVi',
            ]
        )
    ] = 'HYP'

    regions[
        np.isin(acronyms,
            [
                'ZI',
            ]
        )
    ] = 'ZI'

    regions[
        np.isin(acronyms, ['MH', 'LH'])
    ] = 'HB'

    regions[
        np.isin(acronyms,
            [
                'LS', 'LSX', 'LSc', 'LSr', 'LSv',
                'MS',
                'SF', 'SH'
                ]
            )
    ] = 'SEP'

    regions[
        np.isin(acronyms, ['NDB', 'SI'])
    ] = 'BF'

    regions[
        np.isin(acronyms,
            [
                'CLA',  # claustrum proper
                'CTXsp',  # subplate, developmental
                'EPd', 'EPv'  # endopiriform, maybe OLF
            ]
        )
    ] = 'CLA'

    regions[
        np.isin(acronyms,
            [
                'SEZ', 'VL',
                'aco', 'alv', 'amc', 'ar', 'ccb', 'ccg', 'ccs', 'chpl', 'cing',
                'ec', 'ee', 'em', 'fa', 'fi', 'fiber tracts', 'fp', 'fr', 'int',
                'opt', 'or', 'scwm', 'sm', 'st'
            ]
        )
    ] = 'fiber'

    regions[
        np.isin(acronyms, ['V3'])
    ] = 'ventricle'

    regions[
        np.isin(acronyms, ['root', 'void', None])
    ] = 'none'

    if any(regions == ''):
        print(f"WARNING: regions not included in parcellation: {acronyms[regions == '']}")

    return regions


def remap_names(acronyms, source='Allen', dest='Beryl'):
    br = BrainRegions()
    _, inds = ismember(br.acronym2id(acronyms), br.id[br.mappings[source]])
    remapped_acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    return remapped_acronyms

def combine_regions(allen_acronyms, split_thalamus=True, abbreviate=True):
    acronyms = remap_names(allen_acronyms)
    regions = np.array(['root'] * len(acronyms), dtype=object)
    if abbreviate:
        regions[np.isin(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'mPFC'
        regions[np.isin(acronyms, ['MOs'])] = 'M2'
        regions[np.isin(acronyms, ['ORBl', 'ORBm'])] = 'OFC'
        if split_thalamus:
            regions[np.isin(acronyms, ['PO'])] = 'PO'
            regions[np.isin(acronyms, ['LP'])] = 'LP'
            regions[np.isin(acronyms, ['LD'])] = 'LD'
            regions[np.isin(acronyms, ['RT'])] = 'RT'
            regions[np.isin(acronyms, ['VAL'])] = 'VAL'
        else:
            regions[np.isin(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thal'
        regions[np.isin(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'SC'
        regions[np.isin(acronyms, ['RSPv', 'RSPd'])] = 'RSP'
        regions[np.isin(acronyms, ['MRN'])] = 'MRN'
        regions[np.isin(acronyms, ['ZI'])] = 'ZI'
        regions[np.isin(acronyms, ['PAG'])] = 'PAG'
        regions[np.isin(acronyms, ['SSp-bfd'])] = 'BC'
        #regions[np.isin(acronyms, ['LGv', 'LGd'])] = 'LG'
        regions[np.isin(acronyms, ['PIR'])] = 'Pir'
        #regions[np.isin(acronyms, ['SNr', 'SNc', 'SNl'])] = 'SN'
        regions[np.isin(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'VIS'
        regions[np.isin(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amyg'
        regions[np.isin(acronyms, ['AON', 'TTd', 'DP'])] = 'OLF'
        regions[np.isin(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Str'
        regions[np.isin(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hipp'
    else:
        regions[np.isin(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'Medial prefrontal cortex'
        regions[np.isin(acronyms, ['MOs'])] = 'Secondary motor cortex'
        regions[np.isin(acronyms, ['ORBl', 'ORBm'])] = 'Orbitofrontal cortex'
        if split_thalamus:
            regions[np.isin(acronyms, ['PO'])] = 'Thalamus (PO)'
            regions[np.isin(acronyms, ['LP'])] = 'Thalamus (LP)'
            regions[np.isin(acronyms, ['LD'])] = 'Thalamus (LD)'
            regions[np.isin(acronyms, ['RT'])] = 'Thalamus (RT)'
            regions[np.isin(acronyms, ['VAL'])] = 'Thalamus (VAL)'
        else:
            regions[np.isin(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thalamus'
        regions[np.isin(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'Superior colliculus'
        regions[np.isin(acronyms, ['RSPv', 'RSPd'])] = 'Retrosplenial cortex'
        regions[np.isin(acronyms, ['MRN'])] = 'Midbrain reticular nucleus'
        regions[np.isin(acronyms, ['AON', 'TTd', 'DP'])] = 'Olfactory areas'
        regions[np.isin(acronyms, ['ZI'])] = 'Zona incerta'
        regions[np.isin(acronyms, ['PAG'])] = 'Periaqueductal gray'
        regions[np.isin(acronyms, ['SSp-bfd'])] = 'Barrel cortex'
        #regions[np.isin(acronyms, ['LGv', 'LGd'])] = 'Lateral geniculate'
        regions[np.isin(acronyms, ['PIR'])] = 'Piriform'
        #regions[np.isin(acronyms, ['SNr', 'SNc', 'SNl'])] = 'Substantia nigra'
        regions[np.isin(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'Visual cortex'
        regions[np.isin(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amygdala'
        regions[np.isin(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Tail of the striatum'
        regions[np.isin(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hippocampus'
    return regions
