""" from https://github.com/keithito/tacotron """

import re


valid_symbols = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
    'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
    'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
    'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
    'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
    'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
    'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
] + [
    'unk', 'ia5', 'ua5', 'iou3', 'io4', 'zh', 'iou1', 'g', 'a1', 'ua4', 'ai4', 'o2', 
    'iao3', 'ue3', 'q', 'va4', 'ue5', 'ii4', 'uei3', 'iu3', 'va3', 'l', 'iao2', 'uo4', 'ei4', 
    'j', 'e2', 'o3', 'ei3', 'iu2', 'e5', 'ao2', 'ie4', 'ie2', 'ua1', 'ia4', 'k', 'ua3', 've1',
    'va1', 'p', 'uai1', 'ou3', 'uei2', 've2', 'u1', 'ie1', 'm', 'ai3', 'i4', 'u4', 'v1', 'uo5',
    'a2', 'uei1', 'ao4', 'ch', 'io3', 'ua2', 'io2', 's', 'ia3', 'ou4', 'i1', 'v2', 'e1', 'uo3',
    'i3', 'u3', 'i2', 'e4', 't', 'd', 'ao3', 'ie3', 'iou2', 'a5', 'ii2', 'i5', 'ng', 'uei4', 'uai4',
    'c', 'uo2', 'v4', 'o4', 'x', 'ue2', 'ia2', 'uai2', 'ii1', 'ou2', 'f', 'ei2', 've4', 'a3', 'uo1',
    'io1', 'iu1', 'ei1', 'ia1', 'ue4', 'iao4', 'ai2', 'iu4', 'ue1', 'ou5', 'ai1', 'iou4', 'b', 'o1', 
    'iao1', 'ii5', 'va2', 'ao1', 'u2', 'n', 've3', 'o5', 'ii3', 'r', 'ou1', 'h', 'v3', 'sh', 'z', 'e3', 'a4']

_valid_symbol_set = set(valid_symbols)


class CMUDict:
    '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''

    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding='latin-1') as f:
                entries = _parse_cmudict(f)
        else:
            entries = _parse_cmudict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word,
                       pron in entries.items() if len(pron) == 1}
        self._entries = entries

    def __len__(self):
        return len(self._entries)

    def lookup(self, word):
        '''Returns list of ARPAbet pronunciations of the given word.'''
        return self._entries.get(word.upper())


_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
    cmudict = {}
    for line in file:
        if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
            parts = line.split('  ')
            word = re.sub(_alt_re, '', parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in cmudict:
                    cmudict[word].append(pronunciation)
                else:
                    cmudict[word] = [pronunciation]
    return cmudict


def _get_pronunciation(s):
    parts = s.strip().split(' ')
    for part in parts:
        if part not in _valid_symbol_set:
            return None
    return ' '.join(parts)
