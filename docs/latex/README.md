# LaTeX sources of the paper and the tear sheet

`paper/` builds `docs/paper.pdf` and `tearsheet/` builds `docs/Tearsheet.pdf`. The figures in `fig/` are vector crops of the published charts.

```bash
cd docs/latex/paper && latexmk -pdf main.tex          # pdfLaTeX, Palatino (mathpazo)
cd docs/latex/tearsheet && latexmk -xelatex tearsheet.tex   # XeLaTeX, Inter font
```
