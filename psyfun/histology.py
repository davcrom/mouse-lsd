"""Histology-pipeline status predicates derived from Alyx insertion records.

These helpers describe pipeline progress for a probe insertion: whether
histology image stacks have been uploaded for the subject, whether the
probe track has been traced, and whether alignments have been uploaded
and resolved. They read Alyx records only — no bulk data is downloaded.
"""
import re

import requests


HIST_REL = 'histology/{lab}/{subject}/downsampledStacks_25/sample2ARA/'


def list_histology_tifs(subject: str, lab: str, par) -> list[str]:
    """Return the list of ``.tif`` filenames published for ``subject`` in ``lab``.

    Empty list if the directory does not exist or the request fails.
    """
    url = f'{par.HTTP_DATA_SERVER}/' + HIST_REL.format(lab=lab, subject=subject)
    try:
        r = requests.get(
            url,
            auth=(par.HTTP_DATA_SERVER_LOGIN, par.HTTP_DATA_SERVER_PWD),
            timeout=15,
        )
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    return [m + '.tif' for m in re.findall(r'href="(.*).tif"', r.text)]


def insertion_picks(insertion: dict) -> bool:
    return bool((insertion.get('json') or {}).get('xyz_picks') or [])


def insertion_alignment_uploaded(insertion: dict) -> bool:
    eq = ((insertion.get('json') or {}).get('extended_qc') or {})
    n = eq.get('alignment_count')
    return bool(n) and n > 0


def insertion_alignment_resolved(insertion: dict) -> bool:
    eq = ((insertion.get('json') or {}).get('extended_qc') or {})
    return eq.get('alignment_resolved') is True
