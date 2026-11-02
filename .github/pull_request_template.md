## What and why

## Checklist

- [ ] `python tasks.py lint` and `python tasks.py test` pass
- [ ] A new signal or feature passes the truncation (causality) test pattern of `tests/test_causality.py`
- [ ] New parameters live in `config.py` with a comment and a default
- [ ] Out-of-sample claims are pre-registered (`fmflow register`) and reported whatever the result
- [ ] Docs updated (README, METHODOLOGY or DATA as relevant) and `python tasks.py docs` re-run if the API changed
