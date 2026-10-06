import json
from pathlib import Path
import re
import random

import pandas as pd
from tqdm import tqdm
import gzip

random.seed(42)


fn = '/home/morano/SW/data-processing/__detailed_data/vibes_for_clip2.csv'
fn = '/home/morano/SW/data-processing/vibes_for_clip3.csv'


REPEAT = True

# first line is column names
# pat_id;eye;scan;gender;current_age;diagnosis_group;diagnoses;visual_acuities;clinical_findings;treatments
df = pd.read_csv(fn, header=0, sep=';')

print(df.head())

# Print top 5 diagnoses
# print(df['diagnoses'].value_counts().head())

# new_df = df.dropna(subset=['clinical_findings'])
# print(new_df.head())
# # Count number of samples
# print(f'Number of samples with clinical_findings: {len(new_df)}')

eye_mapping = {
    'right': 'OD',
    'left': 'OS',
}

with open('./__new_dataset_v2/all_fsid_list.json', 'r') as f:
    all_fsid_list = json.load(f)

print(f'Number of FSIDs in the dataset: {len(all_fsid_list)}')


if Path('__v3/diagnostic_data.json.gz').exists() and not REPEAT:
    with gzip.open('__v3/diagnostic_data.json.gz', 'rt') as f:
        diagnostic_data = json.load(f)
else:
    diagnostic_data = {}
    for i, row in tqdm(df.iterrows()):
        try:
            fsid = row['scan']
            if fsid not in all_fsid_list:
                continue
            eye = eye_mapping[row['eye']]
            patient_id = row['pat_id']
        except Exception as _e:
            continue
        try:
            age = row['current_age']
        except Exception as _e:
            age = "k.A."
        try:
            sex = row['gender']
            # if NaN, also set to "k.A."
            if pd.isna(sex):
                sex = "k.A."
        except Exception as _e:
            sex = "k.A."
        try:
            visual_acuities = eval(row['visual_acuities'])[0]['logmar_data']
        except Exception as _e:
            visual_acuities = -1
        try:
            clinical_findings = eval(row['clinical_findings'])[0]
        except Exception as _e:
            continue
        try:
            oct_findings = clinical_findings['additional_data'][f'{eye}_OCT']
        except Exception as _e:
            oct_findings = "k.A."
        try:
            clinical_term = str(clinical_findings['clinical_term'])
        except Exception as _e:
            clinical_term = "k.A."
        try:
            diagnoses = eval(row['diagnoses'])[0]['diagnosis']
        except Exception as _e:
            diagnoses = "k.A."
        if oct_findings == "k.A." and clinical_term == "k.A." and diagnoses == "k.A.":
            continue
        diagnostic_data[fsid] = {
            'patient_id': patient_id,
            'oct_findings': oct_findings,
            'visual_acuities': visual_acuities,
            'age': age,
            'sex': sex,
            'diagnoses': diagnoses,
            'clinical_term': clinical_term,
        }

    with gzip.open('__v3/diagnostic_data.json.gz', 'wt') as f:
        json.dump(diagnostic_data, f, indent=2, ensure_ascii=False)


print(f'Parsed diagnostic data for {len(diagnostic_data)} scans.')

for fsid, data in list(diagnostic_data.items())[:5]:
    print(json.dumps({fsid: data}, indent=2, ensure_ascii=False))


def medical_cleaner(text):
    if not isinstance(text, str) or not text.strip() or text.strip() in {
        '-', '0', 'keine Angaben', 'Keine Angaben', 'n/a', 'k.A.'
    }:
        return "k.A."

    _patterns = [
        (re.compile(p, re.IGNORECASE), r)
        for p, r in [
            (r'\s*\+\s*', 'und '),
            (r'\s*>>\s*', 'wesentlich stärker ausgeprägt als '),
            (r'\(\s*--\s*\)', '(deutlich abnehmend)'),
            (r'\(\s*-\s*\)', '(abnehmend)'),
            (r'\(\s*\+\s*\)', '(zunehmend)'),
            (r'\?{2,}', '?'),
            (r'\((\d+)Mo\)', '(\1 Monaten)'),
            (r'\((\d+)Wo\)', '(\1 Wochen)'),
            (r'\bMo\b', 'Monaten'),
            (r'\bWo\b', 'Wochen'),
            # Make new lines as ;
            (r'\n+', '; '),
            # Replace -> with daher
            (r'\s*->\s*', ' daher '),
            # Replace > with mehr als
            (r'\s*>\s*', ' mehr als '),
            # Replace < with weniger als
            (r'\s*<\s*', ' weniger als '),

            # --- LATERALITY (fused forms before standalone) ---
            (r'\bou(?=[A-ZÄÖÜ])', 'beidseits '),
            (r'\bos(?=[A-ZÄÖÜ])', 'linkes Auge '),
            (r'\bod(?=[A-ZÄÖÜ])', 'rechtes Auge '),
            (r'\bo\.u\.', 'beidseits'),
            (r'\bo\.s\.', 'linkes Auge'),
            (r'\bo\.d\.', 'rechtes Auge'),
            (r'\b(ou|bds?|BA)\b', 'beidseits'),
            (r'\b(os|LA)\b', 'linkes Auge'),
            (r'\b(od|RA)\b', 'rechtes Auge'),
            (r'\bR\b', 'rechtes Auge'),
            (r'\bL\b', 'linkes Auge'),

            # --- RPE COMPOUNDS (must come before bare RPE/PE rules) ---
            (r'\bRPE-Riss\b', 'Pigmentepithel-Riss'),
            (r'\bRPE-Irreg\.?\b', 'Pigmentepithel-Irregularität'),
            (r'\bRPE-Falten\b', 'Pigmentepithel-Falten'),
            (r'\bPE-Falten\b', 'Pigmentepithel-Falten'),
            (r'\bPE-Irreg(ul)?\.?\b', 'Pigmentepithel-Irregularität'),
            (r'\bRIP\b', 'Pigmentepithel-Riss'),
            (r'\bRPE\b', 'Retinales Pigmentepithel'),
            (r'(?<![A-Za-zÄÖÜäöü])\bPE\b', 'Retinales Pigmentepithel'),

            # --- DIAGNOSES & PATHOLOGY ---
            (r'\bMb\.?\s+Best\b', 'Morbus Best (vitelliforme Makuladystrophie)'),
            (r'\bKatarakt\.?\s+cort\.?\b', 'corticale Katarakt (Rinden-Linsentrübung)'),
            (r'\bRR\b', 'Blutdruck (Riva-Rocci)'),
            (r'\bPIC\b', 'Punctate Inner Choroidopathy (entzündliche Netzhauterkrankung)'),
            (r'\bPOWG\b', 'Primäres Offenwinkelglaukom (Grüner Star)'),
            (r'\bRetinales Pigmentepithel Irr\.?\b', 'Irregularitäten des Retinalen Pigmentepithels'),
            (r'\bV\.?\s?d\.?\b', 'Verdacht auf'),
            (r'\bHypertension\b', 'Bluthochdruck'), # Or 'Druckerhöhung' in an ocular context
            (r'\bPCO\b', 'Nachstar (Kapseleintrübung)'),
            (r'\bAH\s+Nävus\b', 'Aderhautnävus'),
            (r'\bCNV(?=[;.,:\s]|$)', 'choroidale Neovaskularisation'),
            (r'\bAMD(?=[;.,:\s]|$)', 'altersbedingte Makuladegeneration'),
            (r'\bGA\b', 'geographische Atrophie'),
            (r'\bVMTS\b', 'Vitreomakuläres Traktionssyndrom'),
            (r'\bVMT\b', 'vitreomakuläre Traktion'),
            (r'\bERM(?=[;.,:\s]|$)', 'epiretinale Membran'),
            (r'\bERG\b', 'epiretinale Gliose'),
            (r'\bNHA\b', 'Netzhautabhebung'),
            (r'\bHGKA\b', 'hintere Glaskörperabhebung'),
            (r'\bDPED\b', 'Drusenoide Pigmentepithelabhebung'),
            (r'\bPEDs\b', 'Pigmentepithelabhebungen'),
            (r'\bPED(?=[;.,:\s]|$)', 'Pigmentepithelabhebung'),
            (r'\bORT(?=[;.,:\s]|$)', 'Outer Retinal Tubulations'),
            (r'\bCCS(?=[;.,:\s]|$)', 'Chorioretinopathia Centralis Serosa'),
            (r'\bCRA(?=[;.,:\s]|$)', 'Chorioretinopathia centralis serosa'),
            (r'\bZVT(?=[;.,:\s]|$)', 'Zentralvenenthrombose'),
            (r'\bZVV(?=[;.,:\s]|$)', 'Zentralvenenverschluss'),
            (r'\bOHT\b', 'okuläre Hypertension (Augeninnendruckerhöhung)'),
            (r'\bPEX\b', 'Pseudoexfoliationssyndrom'),
            (r'\bRetinoschisis\b', 'Netzhautspaltung (Retinoschisis)'),
            (r'\bAVV\b', 'Zentralarterienverschluss'),
            (r'\bZAV\b', 'Zentralarterienverschluss'), # Common variant
            (r'\bBRVO\b|\bAVT\b|\bAVV\b', 'Astvenenverschluss'),
            (r'\bNPDR\b', 'nicht-proliferative diabetische Retinopathie'),
            (r'\bDME\b', 'diabetisches Makulaödem'),
            (r'\bPCV', 'Polypoidale choroidale Vaskularisation'),
            (r'\b(SRF|SFR)\b', 'subretinale Flüssigkeit'),
            (r'\bIRF\b', 'intraretinale Flüssigkeit'),
            (r'\bIR-Flüssigkeit\b', 'intraretinale Flüssigkeit'),
            (r'\b(IRZ|IRC)\b', 'intraretinale Zysten'),
            (r'\bIR-?Zysten\b', 'intraretinale Zysten'),
            (r'\b(CMÖ|CSMÖ|CME)(?=[;.,:\s]|$)', 'cystoides Makulaödem'),
            (r'\bzystisches\s+Ödem\b', 'cystoides Makulaödem'),
            (r'\bzystoides\b', 'cystoides'),
            # CME-Anstieg -> cystoides Makulaödem Anstieg
            (r'\bCME-Anstieg\b', 'cystoides Makulaödem Anstieg'),


            # --- ANATOMY ---
            (r'\bNH\b', 'Netzhaut'),
            (r'\bGK\b', 'Glaskörper'),
            (r'\bFAZ\b', 'foveale avaskuläre Zone'),
            (r'\bILM\b', 'Internal Limiting Membrane'),
            (r'\bONL\b', 'Outer Nuclear Layer'),
            (r'\bRNFL\b', 'Nervenfaserschicht'),
            (r'\bPRL\b', 'Photorezeptorschicht'),
            (r'\bPRZ\b', 'Photorezeptorschicht'),
            (r'\b(OGB|UGB|GB)\b', 'Gefäßbogen'),
            (r'\bCRT\b', 'zentrale Netzhautdicke'),

            # --- PROCEDURES & DRUGS ---
            (r'\bT&E(?=[;.,\s]|$)', 'Treat and Extend-Schema'),
            (r'\bTAE(?=[;.,\s]|$)', 'Treat and Extend-Schema'),
            (r'\bPDT(?=[;.,\s]|$)', 'Photodynamische Therapie'),
            (r'\bIVOM(?=[;.,\s]|$)', 'Intravitreale Operative Medikamentenapplikation'),
            (r'\bIVOMs(?=[;.,\s]|$)', 'Intravitreale Operative Medikamentenapplikationen'),
            (r'\bFLA(?=[;.,\s]|$)', 'Fluoreszeinangiographie'),
            (r'\bICGA(?=[;.,\s]|$)', 'Indocyaningrün-Angiographie'),
            (r'\b(EY|Ey)(?=[;.,\s]|$)', 'Eylea'),
            (r'\bInj\.?(?=[;.,\s]|$)', 'Injektion'),
            (r'\bKo\.?(?=[;.,\s]|$)', 'Kontrolle'),
            (r'\bHKL(?=[;.,\s]|$)', 'Hinterkammerlinse'),
            (r'\bOP(?=[;.,\s]|$)', 'Operation'),
            (r'\bYAG\s?L?KT(?=[;.,\s]|$)', 'YAG LKT'),

            # --- STATUS & CONTEXT ---
            (r'\bSt\.?\s?p\.?\s?', 'Zustand nach '),
            (r'\bZ\.?\s?n\.?\s?', 'Zustand nach '),
            (r'\bV\.?\s?a\.?\b', 'Verdacht auf'),
            (r'\bHW\b', 'Hinweis'),
            (r'\bdzt\.?\b', 'derzeit'),
            (r'\br>l\b', 'rechts vor links'),
            (r'\bl>r\b', 'links vor rechts'),

            # --- ABBREVIATION EXPANSIONS ---
            (r'\bneovask\.?\b', 'neovaskuläre'),
            (r'\bwg\.?\b', 'wegen'),
            (r'\bExsudationOS\b', 'Exsudation; linkes Auge'),
            (r'\bExsudationOD\b', 'Exsudation; rechtes Auge'),
            (r'\bFLAOS\b', 'Fluoreszeinangiographie; linkes Auge'),
            (r'\bAktivitätOS\b', 'Aktivität; linkes Auge'),
            (r'\bMaculabefundOS\b', 'Maculabefund; linkes Auge'),
            (r'\bLucentisinjktionOS\b', 'Lucentis-Injektion; linkes Auge'),
            (r'\bLucentisinjktionOD\b', 'Lucentis-Injektion; rechtes Auge'),
            (r'\bObservanzOS\b', 'Observanz; linkes Auge'),
            (r'\bCNVOS\b', 'choroidale Neovaskularisation; linkes Auge'),
            (r'\bRezidivOS\b', 'Rezidiv; linkes Auge'),
            (r'\bAmbOS\b', 'Amb; linkes Auge'),
            (r'\bTerminOS\b', 'Termin; linkes Auge'),
            (r'\bTerminOD\b', 'Termin; rechtes Auge'),
            (r'\bVeränderungOS\b', 'Veränderung; linkes Auge'),
            (r'\bstabilOS\b', 'stabil; linkes Auge'),
            (r'\bFibroseOS\b', 'Fibrose; linkes Auge'),
            (r'\btrockenOS\b', 'trocken; linkes Auge'),
            (r'\bAMDOS\b', 'altersbedingte Makuladegeneration; linkes Auge'),
            (r'\bMaculabefudOS\b', 'Maculabefund; linkes Auge'),
            (r'\bMaculabefundOS\b', 'Maculabefund; linkes Auge'),
            (r'\bMaculabefundOD\b', 'Maculabefund; rechtes Auge'),
            (r'\bAMDOD\b', 'altersbedingte Makuladegeneration; rechtes Auge'),
            (r'\btrockenOD\b', 'trocken; rechtes Auge'),
            (r'\bFibroseOD\b', 'Fibrose; rechtes Auge'),
            (r'\bstabilOD\b', 'stabil; rechtes Auge'),
            (r'\bVeränderungOD\b', 'Veränderung; rechtes Auge'),
            (r'\bAmbOD\b', 'Amb; rechtes Auge'),
            (r'\bRezidivOD\b', 'Rezidiv; rechtes Auge'),
            (r'\bCNVOD\b', 'choroidale Neovaskularisation; rechtes Auge'),
            (r'\bObservanzOD\b', 'Observanz; rechtes Auge'),
            (r'\bMaculabefundOD\b', 'Maculabefund; rechtes Auge'),
            (r'\bAktivitätOD\b', 'Aktivität; rechtes Auge'),
            (r'\bFLAOD\b', 'Fluoreszeinangiographie; rechtes Auge'),
            (r'\bdiab\.?\b', 'diabetische'),
            (r'\bFd\.?\b', 'Fundus'),
            (r'\bHH-narbe\b', 'Hornhautnarbe'),
            (r'\bMet\.?\b', 'Metastase'),
            (r'\bz\.?\s?a\.?\b', 'zentral anliegend (nicht abgehoben)'),
            (r'\bnuc\.?\b', 'nuclearis'),
            (r'\bet\b', 'und'),
            (r'\bo\.?\s?B\.?\b', 'ohne Befund (unauffällig)'),
            (r'\bexs\.?\b', 'exsudative (feuchte)'),
            (r'\bexsudative\b', 'exsudative (feuchte)'),
            (r'\bKhunt\s*Jun\b', 'Junius-Kuhnt'),
            (r'\bKhunt\b', 'Kuhnt'),
            (r'\bTh\.?\b', 'Therapie'),
            (r'\bKonj\.?\s?sicca\b', 'Conjunctivitis sicca (trockenes Auge)'),
            (r'\bPXG\b', 'Pseudoexfoliationsglaukom'),
            (r'\bAugeMaculaforamen\b', 'Auge; Maculaforamen'),
            (r'\bRAM\b', 'retinales arterielles Makroaneurysma'),
            (r'\AugeERM\b', 'Auge; epiretinale Membran'),
            (r'\bDD', 'Differentialdiagnose'),
            (r'\bVMTLaut\b', 'vitreomakuläre Traktion laut'),
            (r'\bRE:\b', 'rechtes Auge:'),
            (r'\bTelJun\b', 'Tel Junius'),
            (r'\bPatient\.\b', 'Patientin'),
            (r'\bGeogr\b', 'geographische'),
            (r'\bAugeKhunt\s+Junius\b', 'Auge Junius-Kuhnt'),
            (r'\bKhunt\s+Junius\b', 'Junius-Kuhnt'),
            (r'\bintravitr\.?\b', 'intravitreale'),
            (r'\bInj\s*OD\b', 'Injektion rechtes Auge'),
            (r'\bInj\s*OS\b', 'Injektion linkes Auge'),
            (r'\bMMC\b', 'Mitomycin C (Antifibrotikum)'),
            (r'\bTrab\b', 'Trabektomie (filternde Glaukom-Operation)'),
            (r'\bLucentisOD\b', 'Lucentis rechtes Auge'),
            (r'\bLucentisOS\b', 'Lucentis linkes Auge'),
            (r'\bAugeLA\b', 'linkes Auge'),
            (r'DefekteLA', 'Defekte; linkes Auge'),
            (r'o\.? u\.?J\.', 'beidseits. Junius-'),
            (r'AugeCNV', 'Auge; choroidale Neovaskularisation'),
            (r'\bBefundOD\b', 'Befund; rechtes Auge'),
            (r'\bbeidseitsCNV\b', 'beidseits choroidale Neovaskularisation'),
            (r'\bPCOund\b', 'Nachstar (Kapseleintrübung) und'),
            (r'\bErkrankungOD\b', 'Erkrankung; rechtes Auge'),
            (r'\bErkrankungOS\b', 'Erkrankung; linkes Auge'),
            (r'\bNarbenstadiumOD\b', 'Narbenstadium; rechtes Auge'),
            (r'\bNarbenstadiumOS\b', 'Narbenstadium; linkes Auge'),
            (r'\bBlutungOD\b', 'Blutung; rechtes Auge'),
            (r'\bBlutungOS\b', 'Blutung; linkes Auge'),
            (r'\bBefundOS\b', 'Befund; linkes Auge'),
            (r'\bOD:\s?Junius', 'rechtes Auge: Junius'),
            (r'\bderzt\.?\b', 'derzeit'),
            (r'\bIntervallOD\b', 'Intervall; rechtes Auge'),
            (r'\bNetzhautbefundOD\b', 'Netzhautbefund; rechtes Auge'),
            (r'\bNetzhautbefundOS\b', 'Netzhautbefund; linkes Auge'),
            (r'\bIntervallOS\b', 'Intervall; linkes Auge'),
            (r'\bAugeRA\b', 'rechtes Auge'),
            (r'InjektionRA', 'Injektion rechtes Auge'),
            (r'\bSK\s+Revision\b', 'Sickerkissen-Revision (Glaukom-Filterkissen-Chirurgie)'),
            (r'Macula', 'Makula'),
            (r'\bChron\b', 'chronische'),
            (r'\bchron\.?\b', 'chronische'),
            (r'\bC\s+guttata\b', 'Cornea guttata (Hornhaut-Endothelveränderung)'),
            (r'\bSek\b', 'sekundäre'),
            (r'\bbinok\.?\s+DB\b', 'binokulare Doppelbilder'),
            (r'\bBeg\.?\b', 'beginnende'),
            (r'\bCe\s+Entfernung\b', 'Cerclage-Entfernung'),
            (r'\bbei Bed\.?\b', 'bei Bedarf'),
            (r'\bFkt\b', 'Funktionell'),
            (r'\bfunkt\.?\b', 'funktionell'),
            (r'\bCat-OP\b', 'Katarakt-Operation'),
            (r'\bJun\.\s*Kuhnt\b', 'Junius-Kuhnt'),
            (r'\bJ\s*\.?\s*Kuhnt\b', 'Junius-Kuhnt'),
            (r'\bJ\.\s+Kuhnt\b', 'Junius-Kuhnt'),
            (r'\bJunius[-\s]+Kh?u[hn]+t\b', 'Junius-Kuhnt'),
            (r'\b(Cat|Kat)\.?\b', 'Katarakt'),
            (r'\bAugesenilis\b', 'Auge Katarakt senilis'),
            (r'\bKataraktsenilis\b', 'Katarakt senilis'),
            (r'\bsen\.?\b', 'senilis'),
            (r'\bincip\.?\b', 'incipiens'),
            # Kataraktincip.
            (r'\bKataraktincip\.?\b', 'Katarakt incipiens'),
            (r'\bIncipiens\b', 'beginnendes'),
            (r'\binc\b\.?', 'beginnende'),
            (r'\brez\.?\b', 'rezidivierend'),
            (r'\bfragl\.?\b', 'fraglich'),
            (r'\bev(tl)?\.?\b', 'eventuell'),
            (r'\bdzt\.?\b', 'derzeit'),
            (r'\bidem\b', 'gleichbleibend'),
            (r'\bca\.?\b', 'circa'),
            (r'\bim\s+Vgl\.?\b', 'im Vergleich'),
            (r'\bvgl\.?\b', 'vergleiche'),
            (r'\bnucl\.?\b', 'nuclearis'),
            (r'\bdeutl\.\s*', 'deutliche '),
            (r'\bdeutl\b', 'deutliche'),
            (r'\bdiff\.?\b', 'diffuses'),
            (r'\bgeringgr\.?\b', 'geringgradiger'),
            (r'\bmin\.\s*', 'minimale '),  # consumes dot + trailing space
            (r'\bmin\b', 'minimale'),   # dotless variant
            (r'\bmini\b', 'minimale'),
            (r'\bkl\.?\b', 'kleine'),
            (r'\bzart\.?\b', 'zarte'),
            (r'\betw\.?\b', 'etwas'),
            (r'\bann\.?\b', 'annähernd'),
            (r'\bger\.?\b', 'geringe'),
            (r'\btw\.?\b', 'teilweise'),
            (r'\btlw\.?\b', 'teilweise'),
            (r'\bbzw\.?\b', 'beziehungsweise'),
            (r'\btgl\.?\b', 'täglich'),
            (r'\bMac\s?Tel\b', 'makuläre Telangiektasien'),
            (r'\bzlw\.?\b', 'stellenweise'),
            (r'\bsup\.?\b', 'superior'),
            (r'\btemp\.?\b', 'temporal'),
            (r'\bzentr\.?\b', 'zentrale'),
            (r'\bsubfov\.?\b', 'subfoveale'),
            (r'\bparafov\.?\b', 'parafoveale'),
            (r'\bjuxtafov\.?\b', 'juxtafoveal'),
            (r'\bparapap\.?\b', 'parapapillär'),
            (r'\bperipap\.?\b', 'peripapilläres'),
            (r'\bintraret\.?\b', 'intraretinale'),
            (r'\bsubret\.?\b', 'subretinale'),
            (r'\bepiret\.?\b', 'epiretinale'),
            (r'\bfibrovask\.?\b', 'fibrovaskuläre'),
            (r'\bfibrot?\.?\b', 'fibrotische'),
            (r'\bfib\.?\b', 'fibrotische'),
            (r'\bgeograph\.?\b', 'geographische'),
            (r'\bMak\.?\b', 'Makula'),
            (r'\bAtroph\b', 'Atrophie'),
            (r'\bIrreg\.?\b', 'Irregularität'),
            (r'\bIrri\.?\b', 'Irregularität'),
            (r'\bUnregelm\.?\b', 'Unregelmäßigkeiten'),
            (r'\bHäm\.?\b', 'Blutung'),
            (r'\bHEs\b', 'Harte Exsudate'),
            (r'\bPEV\b', 'Pigmentepithel-Veränderung'),
            (r'\bPRL\b', 'Photorezeptorschicht'),
            (r'\bpapillomakuläre\b', 'papillomakulären'),
            (r'\bpara-?papilläre\b', 'parapapillär'),
            (r'\bPrazentrales\b', 'Parazentrale'),
            (r'\bsubj\b', 'subjektiv'),
            (r'\bSV\b', 'Sehverschlechterung'),
            # (mit Patient/in besprochen) -> nothing
            (r'\bmit Pat.*besprochen\b', ''),
            (r'\bPat\.?\b', 'Patient'),
            # (auf Wunsch der Pat. ) -> nothing
            (r'\bauf Wunsch der Pat\.?\b', ''),
            (r'\bmehrfach\s+IVOMs?\b', 'mehrfache intravitrealen operative Medikamentenapplikationen'),
            # (Pat. möchte noch bis nächste Kontrolle. warten) -> nothing
            (r'\bPat\.?\s*möchte\s*noch\s*bis\s*nächste\s*Kontrolle\s*warten\b', ''),
            (r'\bAAV\b', 'Arterienastverschluss'),
            (r'\bi\.?\s?B\.?\b', 'im Bereich'),
            (r'Fivrose', 'Fibrose'),


            # --- TYPO FIXES ---
            (r'\bAktivitätszeichenOs\b', 'Aktivitätszeichen; linkes Auge'),
            (r'\bAktivitätszeichenOd\b', 'Aktivitätszeichen; rechtes Auge'),
            (r'\bAugeOW-Glaukom\b', 'Auge; Offenwinkelglaukom'),
            (r'\bAMDGeographische\b', 'altersbedingte Makuladegeneration; geographische'),
            (r'\bObservanzF\b', 'Observanz; Fundusuntersuchung'),
            (r'Yag-KTERM', 'YAG KT; epiretinale Membran'),
            (r'\bauf\s+auf\b', 'auf'),
            (r'\bCNVwesentlich\b', 'choroidale Neovaskularisation wesentlich'),
            (r'\bSklerapatchesund\b', 'Sklera-Patches und'),
            (r'\bppVEund\s+MP\b', 'Pars-plana-Vitrektomie und Membranpeeling'),
            (r'\bppVE\b', 'Pars-plana-Vitrektomie'),
            (r'VEund\s+MPund', 'Vitrektomie und Membranpeeling und'),
            (r'\bVEund\b', 'Vitrektomie und'),
            (r'\bKryound\b', 'Kryopexie und'),
            (r'\bMPund\b', 'Membranpeeling und'),
            (r'\bHKLund\b', 'Hinterkammerlinse und'),
            (r'\bscar\b', 'Narbe'),
            (r'\bArbe\b', 'Narbe'),
            (r'Narebe', 'Narbe'),
            (r'\bNN(?=arbe)', 'N'),
            (r'\bnN(?=arbe)', 'N'),
            (r'\bPhakound\b', 'Phakoemulsifikation und'),
            (r'CMEund', 'cystoides Makulaödem und'),
            (r'Narbe\s+bereich', 'Narbenbereich'),
            (r'\bin\s+die\s+Narbenbereich\b', 'im Narbenbereich'),
            (r'\bSpatbildung\b', 'Spaltbildung'),
            (r'Verdicktung', 'Verdickung'),
            (r'subfofeal', 'subfoveal'),
            (r'\binraretinale\b', 'intraretinale'),
            (r'\bFibröse\b', 'fibrotische'),
            (r'firb\.', 'fibrotische'),
            (r'\bFibr\.?\b', 'fibrotische'),
            (r'peelingKeine', 'peeling. Keine'),
            (r'TrockenParafoveal', 'Trocken. Parafoveal'),
            (r'Arophie|Atrphie|Atophie|Atrophei', 'Atrophie'),
            (r'Flü[üß]igkeit', 'Flüssigkeit'),
            (r'Unregelmäs+igkeiten?', 'Unregelmäßigkeiten'),
            (r'unregelmäßigkeiten', 'Unregelmäßigkeiten'),
            (r'\bÖdme\b', 'Ödem'),
            (r'\bzysten\b', 'Zysten'),
            (r'\bsontst\b', 'sonst'),
            (r'TROCKEN|trockne|trockern|triocken|torcken|tocken', 'trocken'),
            (r'\bJun\.\s+Kuhnt\b', 'Junius-Kuhnt'),
            (r'CMÖund', 'cystoides Makulaödem und'),
            (r'SRFund', 'subretinale Flüssigkeit und'),
            (r'FLAund', 'Fluoreszeinangiographie und'),
            (r'RandrezidivLA', 'Randrezidiv; linkes Auge'),
            (r'\bOCTLA\b', 'OCT; linkes Auge'),
            (r'\bOCTRA\b', 'OCT; rechtes Auge'),
            (r'\bJ\.\s*KuhntLA\b', 'Junius-Kuhnt; linkes Auge'),
            (r'\bJ\.\s*KuhntRA\b', 'Junius-Kuhnt; rechtes Auge'),
            (r'AMDLA', 'altersbedingte Makuladegeneration; linkes Auge'),
            (r'\bCMELA\b', 'cystoides Makulaödem; linkes Auge'),
            (r'\bPCOLA\b', 'Nachstar; linkes Auge'),
            # LA attached to a previous word
            (r'StabilisierungLA', 'Stabilisierung; linkes Auge'),
            (r'CNV-Aktivität', 'choroidale Neovaskularisation Aktivität'),
            (r'ObservanzLA', 'Observanz; linkes Auge'),
            (r'LucentisLA', 'Lucentis; linkes Auge'),
            (r'superiorLA', 'superior; linkes Auge'),
            (r'INjLA', 'Injektion; linkes Auge'),
            (r'EyleaLA', 'Eylea; linkes Auge'),
            (r'VersuchLA', 'Versuch; linkes Auge'),
            (r'InjektionLA', 'Injektion; linkes Auge'),
            (r'sehverschlechterungLA', 'Sehverschlechterung; linkes Auge'),
            (r'SehverbesserungLA', 'Sehverbesserung; linkes Auge'),
            (r'zuwartenLA', 'zuwarten; linkes Auge'),
            (r'stabilLA', 'stabil; linkes Auge'),
            (r'ÖdemLA', 'Ödem; linkes Auge'),
            (r'MaculabefundLA', 'Maculabefund; linkes Auge'),
            (r'BefundLA', 'Befund; linkes Auge'),
            (r'HH-befundF', 'Hornhautbefund; Fundus'),
            (r'\bSpl\b', 'Spaltlampenuntersuchung'),
            (r'SVLA', 'Sehverschlechterung; linkes Auge'),
            (r'NetzhautbefundLA', 'Netzhautbefund; linkes Auge'),
            (r'idemLA', 'gleichbleibend; linkes Auge'),
            (r'CMÖLA', 'cystoides Makulaödem; linkes Auge'),
            (r'AbstandLA', 'Abstand; linkes Auge'),
            (r'trockenLA', 'trocken; linkes Auge'),
            (r'ÖdemsRA', 'Ödem; rechtes Auge'),
            (r'MaculabefundRA', 'Maculabefund; rechtes Auge'),
            (r'ObsrvanzRA', 'Observanz; rechtes Auge'),
            (r'Monokulusund', 'Monokulus und'),
            (r'ResorptionLA', 'Resorption; linkes Auge'),
            (r'SRFRA\b', 'subretinale Flüssigkeit; rechtes Auge'),
            (r'SRFLA\b', 'subretinale Flüssigkeit; linkes Auge'), # Good to have the pair
            # d. GF (des Gesichtsfelds)
            (r'\bd\.?\s*GF\b', 'des Gesichtsfelds'),
            # z. T. (zum Teil)
            (r'\bz\.?\s*T\.?\b', 'zum Teil'),
            (r'beidseitsCNV-Rezidiv', 'beidseits; choroidale Neovaskularisation Rezidiv'),
            (r'\bo\.sOkuläre', 'linkes Auge; okuläre'),
            (r'AugeCNV-Rezidiv', 'Auge; choroidale Neovaskularisation Rezidiv'),
            (r'\bo\.?\s*d\b', 'rechtes Auge'),
            (r'\bERneute\b', 'erneute'),
            (r'\bCatarakt\s+progression\b', 'fortschreitende Katarakt'),
            (r'\bCatarakt\b', 'Katarakt'),
            (r'\beind\.?\b', 'eindeutig'),
            (r'\banti-?VEGF\b', 'Anti-VEGF'),
            # Vitrektomie und MP -> Vitrektomie und Membranpeeling
            (r'\bVitrektomie\s+und\s+MP\b', 'Vitrektomie und Membranpeeling'),
            (r'\bNetzhautbefundPat\.?\b', 'Netzhautbefund; Patient'),
            # Rule for RA (rechtes Auge)
            (r'([a-zäöüßA-Z0-9]{1,})(RA)\b', r'\1; rechtes Auge'),
            # Rule for LA (linkes Auge)
            (r'([a-zäöüßA-Z0-9]{1,})(LA)\b', r'\1; linkes Auge'),
            # Rule for OD (Oculus Dexter / rechts)
            (r'([a-zäöüßA-Z0-9]{1,})(OD)\b', r'\1; rechtes Auge'),
            # Rule for OS (Oculus Sinister / links)
            (r'([a-zäöüßA-Z0-9]{1,})(OS)\b', r'\1; linkes Auge'),
            (r'([a-zäöüßA-Z0-9]{1,})(OU)\b', r'\1; beidseits'),
            (r'\bLACNV\b', 'linkes Auge: choroidale Neovaskularisation'),
            (r'\bRACNV\b', 'rechtes Auge: choroidale Neovaskularisation'),
            (r'\bLAExotropie\b', 'linkes Auge: Exotropie'),
            (r'\bRAExotropie\b', 'rechtes Auge: Exotropie'),
            (r'\bRAFd\b', 'rechtes Auge Fundus'),
            (r'\bLAFd\b', 'linkes Auge Fundus'),
            (r'\bRAwesentlich\b', 'rechtes Auge: wesentlich'),
            (r'\bLAwesentlich\b', 'linkes Auge: wesentlich'),
            (r'\b2xOSKeratopathie\b', '2x linkes Auge: Keratopathie'),
            (r'\b2xODKeratopathie\b', '2x rechtes Auge: Keratopathie'),
            (r'\bPEDund\b', 'Pigmentepithelabhebungen und'),
            (r'\bPEDwesentlich\b', 'Pigmentepithelabhebungen wesentlich'),
            (r'\bFibrosePED\b', 'Fibrose Pigmentepithelabhebung'),
            (r'\bDPEDs\b', 'Drusenoide Pigmentepithelabhebungen'),
            (r'\bNPDRPFd\.?\b', 'nicht-proliferative diabetische Retinopathie Fundus'),
            (r'\bNPDRP\b', 'nicht-proliferative diabetische Retinopathie'),
            (r'\bPDRP\b', 'proliferative diabetische Retinopathie'),
            (r'\bDRP\b', 'diabetische Retinopathie'),
            (r'YAG\s+Laser\s+Kapsulotomie', 'YAG LKT'),
            (r'\bObservanzDzt\b', 'Observanz derzeit'),
            (r'\bNP\s+Diabetische\s+Retinopathie\b', 'nicht-proliferative diabetische Retinopathie'),
            (r'\bNPDR\b', 'nicht-proliferative diabetische Retinopathie'),
            (r'\bCMEodNPDRP\b', 'cystoides Makulaödem, rechtes Auge nicht-proliferative diabetische Retinopathie'),
            (r'\bINjektion\b', 'Injektion'),
            (r'\bbbei\b', 'bei'),


            # --- GRAMMAR FIXES ---
            (r'mit linkes Auge', 'mit dem linken Auge'),
            (r'mit rechtes Auge', 'mit dem rechten Auge'),
            (r'\bfoveal\b(?=\s+Defekt)', 'fovealer'),
            (r'\bflach\b(?=\s+(sub|intra)retinale)', 'flache'),
            (r'\.\s+(?=Atrophie)', ' '),   # removes stray period before Atrophie
            (r'\btemporal\.', 'temporal'),
            (r'\bparafoveale\.', 'parafoveale'),
            # Drusen MP -> Drusen Makulapolaris
            (r'\bDrusen\s+MP\b', 'Drusen Makulapolaris'),
            (r'\bderzeit\.\s*', 'derzeit '),
            (r'Katarakt-OPOS', 'Katarakt-Operation; linkes Auge'),
            (r'Katarakt-OPOD', 'Katarakt-Operation; rechtes Auge'),
            (r'ChorioretinitisODFunkt\.', 'Chorioretinitis rechtes Auge funktionell'),
            (r'\bNHAODMyopie\b', 'Netzhautabhebung rechtes Auge Myopie'),
            (r'\bNHAOSMyopie\b', 'Netzhautabhebung linkes Auge Myopie'),
            (r'\bRAweitgehend\b', 'rechtes Auge weitgehend'),
            (r'\bLAweitgehend\b', 'linkes Auge weitgehend'),
            (r'\bRADie\b', 'rechtes Auge: Die'),
            (r'\bLADie\b', 'linkes Auge: Die'),
            (r'\bLACataracta\b', 'linkes Auge: Katarakt'),
            (r'\bRACataracta\b', 'rechtes Auge: Katarakt'),
            (r'\bLA\b', 'linkes Auge'),
            (r'\bRA\b', 'rechtes Auge'),
            (r'\bLACat\b', 'linkes Auge: Katarakt'),
            (r'\bRACat\b', 'rechtes Auge: Katarakt'),
            (r'\bRAund\b', 'rechtes Auge und'),
            (r'\bLAund\b', 'linkes Auge und'),
            (r'\bEDund\b', 'ED und'),
            (r'\bPFCLund\b', 'PFCL und'),
            (r'\bLAPseud\b', 'linkes Auge Pseudophakie'),
            (r'\bRAPseud\b', 'rechtes Auge Pseudophakie'),
            (r'\bFLAwesentlich\b', 'Fluoreszeinangiographie wesentlich'),
            (r'\bLAAMD\b', 'linkes Auge: altersbedingte Makuladegeneration'),
            (r'\bRAAMD\b', 'rechtes Auge: altersbedingte Makuladegeneration'),
            (r'\bMakulaTeleangiektasien\b', 'makuläre Teleangiektasien'),
            (r'\bERMund\b', 'epiretinale Membran und'),
            (r'\bAugeRezidiv\b', 'Auge Rezidiv'),
            (r'\bKataraktOperation\b', 'Katarakt-Operation'),
            (r'\bAUge\b', 'Auge'),
            (r'\bAUgenfacharzt\b', 'Augenfacharzt'),
            (r'\bMaku\b', 'Makula'),
            (r'\bAUF\b', 'auf'),
            (r'\bAUsschluss\b', 'Ausschluss'),
            (r'\bOS2006\b', 'linkes Auge, 2006'),
            (r'\bPDTund\b', 'Photodynamische Therapie und'),
            (r'\bTriam\b', 'Triamcinolon'),
            (r'\bTA\b', 'Triamcinolon-Applikation'),
            (r'\bMonatewesentlich\b', 'Monate wesentlich'),
            (r'\bAFAFundus\b', 'Autofluorescence Fundus'),
            (r'\bAFA\b', 'Autofluorescence'),
            (r'\bFAF\b', 'Autofluorescence'),
            (r'CNV-Rezidiv', 'choroidale Neovaskularisation Rezidiv'),
            (r'\bOPs\b', 'Operationen'),
            (r'\bo\.?\s*u\b', 'beidseits'),
            (r'\bERMwesentlich\b', 'epiretinale Membran wesentlich'),
            # Catches the fusion and expands MP
            (r'Katarakt-OPund\s+MP', 'Katarakt-Operation und Membranpeeling'),
            (r'\bCNVosPseudophakie\b', 'choroidale Neovaskularisation linkes Auge Pseudophakie'),
            (r'\bAugeCat\b', 'Auge Katarakt'),
            (r'\bPCVAMD\b', 'polypoidale choroidale Vaskulopathie, altersbedingte Makuladegeneration'),
            (r'\bCNVPatient\b', 'choroidale Neovaskularisation Patient'),
            (r'\bstabil\s+J\.\s*Kuhnt\b', 'stabil Junius-Kuhnt'),
            (r'\bYAG\-KT\b', 'YAG LKT'),
            (r'\bCNCV\b', 'choroidale Neovaskularisation'),
            (r'\bCNC\b', 'choroidale Neovaskularisation'),
            (r'\brfPDT\b', 'reduzierte Photodynamische Therapie'),
            (r'\brTPAund\b', 'rTPA und'),
            (r'\bGasInj\.?\b', 'Gas-Injektion'),
            (r'\bJunius\s+Kuhnt\b', 'Junius-Kuhnt'),
            (r'\bPatient\s*wurde\s*über\s*Art\b', ''),
            (r'\bMaculadegeneration\b', 'Makuladegeneration'),


            # --- FINAL CLEANUP ---
            # Adds space if a letter follows the closing bracket
            (r'\)\s*(?=[A-Za-zÄÖÜäöü])', ') '),
            # Replace ., by ,
            (r'\.,', ','),
            # Add always space after comma if it is not followed by space
            (r',(?=\S)', ', '),
            # Same for .
            (r'\.(?=\S)', '. '),
            # Same for ;
            (r';(?=\S)', '; '),
            # Remove spaces before .
            (r'\s+\.', '.'),
            # Same for ,
            (r'\s+,', ','),
            # Remove spaces before ;
            (r'\s+;', ';'),
            # Replace multiple spaces by only one space
            (r'\s+', ' '),
            # Same with () with only . or space inside
            (r'\(\s*[.\s]*\s*\)', ''),
            # Remove repeated ;
            (r'(;)\1+', r'\1'),
            # Remove repeated .
            (r'(\.)\1+', r'\1'),
            # Remove repeated ,
            (r'(,)\1+', r'\1'),
        ]
    ]

    for pattern, replacement in _patterns:
        text = pattern.sub(replacement, text)

    try:
        text = text.strip()
        if text.endswith(','):
            text = text[:-1] + '.'
        elif text[-1] not in '.!?':
            text += '.'
        return text[0].upper() + text[1:]
    except IndexError:
        return "k.A."


unique_words = set()
for fsid, data in diagnostic_data.items():
    oct_findings = data['oct_findings']
    clinical_term = data['clinical_term']
    diagnoses = data['diagnoses']
    unique_words.update(oct_findings.split())
    unique_words.update(clinical_term.split())
    unique_words.update(diagnoses.split())

print(f'Unique words in clinical findings: {len(unique_words)}')
# print(f'Example unique words: {list(unique_words)[:200]}')
# Print only words with '.'
unique_words_with_dot = [word for word in unique_words if '.' in word]
print(f'Unique words with ".": {len(unique_words_with_dot)}')
# print(f'Example unique words with ".": {unique_words_with_dot}')

save_dir = Path('/home/morano/SW/MIRAGEv2/MIRAGEv2.git/main/cfgs/')

with open(save_dir / 'reports.json', 'w') as f:
    json.dump(diagnostic_data, f, indent=2, ensure_ascii=False)


unique = set()
for fsid, data in tqdm(diagnostic_data.items()):
    data['oct_findings'] = medical_cleaner(data['oct_findings'])
    data['clinical_term'] = medical_cleaner(data['clinical_term'])
    data['diagnoses'] = medical_cleaner(data['diagnoses'])
    data['visual_acuities'] = round(float(data['visual_acuities']), 4)
    # if i > 4000:
    #     break
    # f.write(f"{fsid[:16]}\n{data['oct_findings']}\n{data['clinical_term']}\n{data['diagnoses']}\n\n")
    unique.add(f"{data['oct_findings']}\n{data['clinical_term']}\n{data['diagnoses']}\n\n")

with open(save_dir / 'reports_only.txt', 'w') as f:
    for item in unique:
        f.write(item)

with open(save_dir / 'reports_clean.json', 'w') as f:
    json.dump(diagnostic_data, f, indent=2, ensure_ascii=False)


# Split data at the patient level
# Example:
# "7346848-25-...+QQSEYU": {
#   "patient_id": 1,
#   "oct_findings": "Pigmentepithel-Irregularität, Fibrose, wenig Ödem.",
#   "visual_acuities": 0.8239,
#   "age": 78,
#   "sex": "Female",
#   "diagnoses": ...,
#   "clinical_term": ...,
# },
patients = set()
for fsid, meta in diagnostic_data.items():
    patients.add(meta['patient_id'])
patients = list(patients)
random.shuffle(patients)

num_patients = len(patients)
num_train = int(0.98 * num_patients)
train_patients = patients[:num_train]
val_patients = patients[num_train:]
print(f"Total patients: {num_patients}, Train: {len(train_patients)}, Val: {len(val_patients)}")

new_reports = {"train": {}, "val": {}}
for fsid, meta in diagnostic_data.items():
    if meta['patient_id'] in train_patients:
        new_reports["train"][fsid] = meta
    else:
        new_reports["val"][fsid] = meta

with open(save_dir / "reports_clean_split.json", "w") as f:
    json.dump(new_reports, f, indent=2)
