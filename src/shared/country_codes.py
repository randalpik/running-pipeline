"""English country name -> ISO 3166-1 alpha-2 code, for display only.

The pipeline stores geocode-correct city_states ('Berlin, Germany'); this
collapses the foreign ones to the 2-letter display form ('Berlin, DE') at
render time. US/CA city_states already carry a 2-letter subdivision code
('Chicago, IL') so they pass through unchanged, as do any countries not in
the table (they keep their full name rather than guess).

Kept as a static table (not a dependency) to match the repo's stdlib-only
posture; covers the common race destinations, extend as needed.
"""
from __future__ import annotations

_NAME_TO_ALPHA2 = {
    'afghanistan': 'AF', 'albania': 'AL', 'algeria': 'DZ', 'andorra': 'AD',
    'angola': 'AO', 'argentina': 'AR', 'armenia': 'AM', 'australia': 'AU',
    'austria': 'AT', 'azerbaijan': 'AZ', 'bahamas': 'BS', 'bahrain': 'BH',
    'bangladesh': 'BD', 'barbados': 'BB', 'belarus': 'BY', 'belgium': 'BE',
    'belize': 'BZ', 'benin': 'BJ', 'bhutan': 'BT', 'bolivia': 'BO',
    'bosnia and herzegovina': 'BA', 'botswana': 'BW', 'brazil': 'BR',
    'brunei': 'BN', 'bulgaria': 'BG', 'burkina faso': 'BF', 'burundi': 'BI',
    'cambodia': 'KH', 'cameroon': 'CM', 'canada': 'CA', 'chile': 'CL',
    'china': 'CN', 'colombia': 'CO', 'costa rica': 'CR', 'croatia': 'HR',
    'cuba': 'CU', 'cyprus': 'CY', 'czechia': 'CZ', 'czech republic': 'CZ',
    'denmark': 'DK', 'dominican republic': 'DO', 'ecuador': 'EC', 'egypt': 'EG',
    'el salvador': 'SV', 'estonia': 'EE', 'ethiopia': 'ET', 'fiji': 'FJ',
    'finland': 'FI', 'france': 'FR', 'georgia': 'GE', 'germany': 'DE',
    'ghana': 'GH', 'greece': 'GR', 'guatemala': 'GT', 'honduras': 'HN',
    'hong kong': 'HK', 'hungary': 'HU', 'iceland': 'IS', 'india': 'IN',
    'indonesia': 'ID', 'iran': 'IR', 'iraq': 'IQ', 'ireland': 'IE',
    'israel': 'IL', 'italy': 'IT', 'ivory coast': 'CI', 'jamaica': 'JM',
    'japan': 'JP', 'jordan': 'JO', 'kazakhstan': 'KZ', 'kenya': 'KE',
    'kuwait': 'KW', 'kyrgyzstan': 'KG', 'laos': 'LA', 'latvia': 'LV',
    'lebanon': 'LB', 'libya': 'LY', 'liechtenstein': 'LI', 'lithuania': 'LT',
    'luxembourg': 'LU', 'macau': 'MO', 'madagascar': 'MG', 'malaysia': 'MY',
    'maldives': 'MV', 'malta': 'MT', 'mexico': 'MX', 'moldova': 'MD',
    'monaco': 'MC', 'mongolia': 'MN', 'montenegro': 'ME', 'morocco': 'MA',
    'mozambique': 'MZ', 'myanmar': 'MM', 'nepal': 'NP', 'netherlands': 'NL',
    'new zealand': 'NZ', 'nicaragua': 'NI', 'nigeria': 'NG', 'north korea': 'KP',
    'north macedonia': 'MK', 'norway': 'NO', 'oman': 'OM', 'pakistan': 'PK',
    'panama': 'PA', 'paraguay': 'PY', 'peru': 'PE', 'philippines': 'PH',
    'poland': 'PL', 'portugal': 'PT', 'qatar': 'QA', 'romania': 'RO',
    'russia': 'RU', 'rwanda': 'RW', 'saudi arabia': 'SA', 'senegal': 'SN',
    'serbia': 'RS', 'singapore': 'SG', 'slovakia': 'SK', 'slovenia': 'SI',
    'south africa': 'ZA', 'south korea': 'KR', 'spain': 'ES', 'sri lanka': 'LK',
    'sweden': 'SE', 'switzerland': 'CH', 'taiwan': 'TW', 'tanzania': 'TZ',
    'thailand': 'TH', 'tunisia': 'TN', 'turkey': 'TR', 'türkiye': 'TR',
    'uganda': 'UG', 'ukraine': 'UA', 'united arab emirates': 'AE',
    'united kingdom': 'GB', 'united states': 'US', 'uruguay': 'UY',
    'uzbekistan': 'UZ', 'venezuela': 'VE', 'vietnam': 'VN', 'zambia': 'ZM',
    'zimbabwe': 'ZW',
}


def country_abbrev(city_state):
    """'City, <English country>' -> 'City, <alpha-2>' for display.

    Pass-through for US/CA city_states (region is already a 2-letter code, not
    a country name) and for any country missing from the table.
    """
    if not isinstance(city_state, str) or ',' not in city_state:
        return city_state
    city, _, region = city_state.rpartition(',')
    code = _NAME_TO_ALPHA2.get(region.strip().lower())
    return f'{city.strip()}, {code}' if code else city_state
