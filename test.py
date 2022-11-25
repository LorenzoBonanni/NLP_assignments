text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's " \
       "standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make " \
       "a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, " \
       "remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing " \
       "Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions " \
       "of Lorem Ipsum."

# DIVIDE INTO SENTENCES
sentences = [
    'Lorem Ipsum is simply dummy text of the printing and typesetting industry.',
    "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.",
    'It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.',
    'It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.'
]

# DIVIDE EACH SENTENCE INTO TOKENS
tokens = [
    ['lorem', 'ipsum', 'is', 'simply', 'dummy', 'text', 'of', 'the', 'printing', 'and', 'typesetting', 'industry', '.'],
    ['lorem', 'ipsum', 'has', 'been', 'the', 'industry', "'s", 'standard', 'dummy', 'text', 'ever', 'since', 'the',
     '1500s', ',', 'when', 'an', 'unknown', 'printer', 'took', 'a', 'galley', 'of', 'type', 'and', 'scrambled', 'it',
     'to', 'make', 'a', 'type', 'specimen', 'book', '.'],
    ['it', 'has', 'survived', 'not', 'only', 'five', 'centuries', ',', 'but', 'also', 'the', 'leap', 'into',
     'electronic', 'typesetting', ',', 'remaining', 'essentially', 'unchanged', '.'],
    ['it', 'was', 'popularised', 'in', 'the', '1960s', 'with', 'the', 'release', 'of', 'letraset', 'sheets',
     'containing', 'lorem', 'ipsum', 'passages', ',', 'and', 'more', 'recently', 'with', 'desktop', 'publishing',
     'software', 'like', 'aldus', 'pagemaker', 'including', 'versions', 'of', 'lorem', 'ipsum', '.']
]

# ....
# PREPROCESSING
# ....


# WE CANNOT GIVE VECTORS OF DIFFERENT FEATURE SIZE SO WE NEED TO "NORMALIZE" EACH VECTOR
# COMPUTE THE FREQUENCY OF EACH WORD

freqs = {'lorem': 4, 'ipsum': 4, 'is': 1, 'simply': 1, 'dummy': 2, 'text': 2, 'of': 4, 'the': 6, 'printing': 1,
         'and': 3, 'typesetting': 2, 'industry': 2, '.': 4, 'has': 2, 'been': 1, "'s": 1, 'standard': 1, 'ever': 1,
         'since': 1, '1500s': 1, ',': 4, 'when': 1, 'an': 1, 'unknown': 1, 'printer': 1, 'took': 1, 'a': 2, 'galley': 1,
         'type': 2, 'scrambled': 1, 'it': 3, 'to': 1, 'make': 1, 'specimen': 1, 'book': 1, 'survived': 1, 'not': 1,
         'only': 1, 'five': 1, 'centuries': 1, 'but': 1, 'also': 1, 'leap': 1, 'into': 1, 'electronic': 1,
         'remaining': 1, 'essentially': 1, 'unchanged': 1, 'was': 1, 'popularised': 1, 'in': 1, '1960s': 1, 'with': 2,
         'release': 1, 'letraset': 1, 'sheets': 1, 'containing': 1, 'passages': 1, 'more': 1, 'recently': 1,
         'desktop': 1, 'publishing': 1, 'software': 1, 'like': 1, 'aldus': 1, 'pagemaker': 1, 'including': 1,
         'versions': 1}

# COMPUTE `N` MOST COMMON WORDS
most_ten_common = ['the', 'lorem', 'ipsum', 'of', '.', ',', 'and', 'it', 'dummy', 'text']

# TRANSFORM EACH SENTENCE VECTOR INTO A VECTOR THAT STATES IF THE WORD IS PRESENT INTO THE VECTOR
featuresets = [
    {'contains(the)': True, 'contains(lorem)': True, 'contains(ipsum)': True, 'contains(of)': True, 'contains(.)': True,
     'contains(,)': False, 'contains(and)': True, 'contains(it)': False, 'contains(dummy)': True,
     'contains(text)': True},
    {'contains(the)': True, 'contains(lorem)': True, 'contains(ipsum)': True, 'contains(of)': True, 'contains(.)': True,
     'contains(,)': True, 'contains(and)': True, 'contains(it)': True, 'contains(dummy)': True, 'contains(text)': True},
    {'contains(the)': True, 'contains(lorem)': False, 'contains(ipsum)': False, 'contains(of)': False,
     'contains(.)': True, 'contains(,)': True, 'contains(and)': False, 'contains(it)': True, 'contains(dummy)': False,
     'contains(text)': False},
    {'contains(the)': True, 'contains(lorem)': True, 'contains(ipsum)': True, 'contains(of)': True, 'contains(.)': True,
     'contains(,)': True, 'contains(and)': True, 'contains(it)': True, 'contains(dummy)': False,
     'contains(text)': False}]
