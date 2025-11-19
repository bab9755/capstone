import ssl, certifi, nltk

# Patch the SSL context to use certifi's CA bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Now download tokenizers safely
nltk.download("punkt")
nltk.download("punkt_tab")
